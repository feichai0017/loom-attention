"""Calibrate a checkpoint for static FP8 attention and KV-cache scales.

This utility follows vLLM's llm-compressor calibration path while keeping
checkpoint and corpus provenance explicit. It intentionally refuses to
overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time
from typing import Any


WEIGHT_PATTERNS = ("*.safetensors", "*.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Pinned upstream revision recorded alongside the checkpoint digest.",
    )
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention-target",
        required=True,
        help="Exact Transformers attention class, for example Qwen2Attention.",
    )
    parser.add_argument(
        "--strategy",
        choices=("tensor", "attn_head"),
        default="attn_head",
    )
    parser.add_argument(
        "--observer",
        choices=("static_minmax", "minmax", "mse"),
        required=True,
        help=(
            "Stateful observer selected for this workload; the system-quality "
            "gate, not this tool, decides whether its scales are acceptable."
        ),
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--messages-column", default="messages")
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.max_seq_len <= 1:
        parser.error("--max-seq-len must be greater than one")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def weight_files(root: Path) -> list[Path]:
    files = {
        path
        for pattern in WEIGHT_PATTERNS
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not files:
        raise ValueError(f"no checkpoint weights found under {root}")
    return sorted(files)


def file_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checkpoint_manifest(root: Path) -> dict[str, Any]:
    files = []
    digest = hashlib.sha256()
    for path in weight_files(root):
        relative_path = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": file_digest,
            }
        )
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\0")
    return {
        "digest": f"sha256:{digest.hexdigest()}",
        "bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def package_versions() -> dict[str, str]:
    names = (
        "compressed-tensors",
        "datasets",
        "llmcompressor",
        "safetensors",
        "torch",
        "transformers",
    )
    return {name: metadata.version(name) for name in names}


def scale_manifest(root: Path) -> dict[str, Any]:
    from safetensors import safe_open

    scales = []
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.endswith("_scale"):
                    scales.append(
                        {
                            "file": path.name,
                            "key": key,
                            "shape": list(checkpoint.get_slice(key).get_shape()),
                        }
                    )
    return {
        "count": len(scales),
        "q": sum(item["key"].endswith("q_scale") for item in scales),
        "k": sum(item["key"].endswith("k_scale") for item in scales),
        "v": sum(item["key"].endswith("v_scale") for item in scales),
        "tensors": scales,
    }


def prepare_dataset(
    parquet_path: Path,
    tokenizer: Any,
    *,
    messages_column: str,
    samples: int,
    max_seq_len: int,
    seed: int,
) -> tuple[Any, list[int]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    if messages_column not in dataset.column_names:
        raise ValueError(
            f"dataset has no {messages_column!r} column: {dataset.column_names}"
        )
    if samples > len(dataset):
        raise ValueError(
            f"requested {samples} calibration samples from a {len(dataset)} row dataset"
        )
    source_row_column = "__loom_source_row"
    dataset = dataset.add_column(source_row_column, list(range(len(dataset))))
    dataset = dataset.shuffle(seed=seed).select(range(samples))
    source_rows = list(dataset[source_row_column])

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = tokenizer.apply_chat_template(
            example[messages_column],
            tokenize=False,
        )
        return tokenizer(
            text,
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenized, source_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model.resolve(strict=True)
    dataset_path = args.dataset_parquet.resolve(strict=True)
    output_path = args.output.resolve()
    if model_path == output_path:
        raise ValueError("--output must differ from --model")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    source_checkpoint = checkpoint_manifest(model_path)
    dataset_sha256 = sha256_file(dataset_path)
    started_at = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        local_files_only=True,
    )
    attention_modules = [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ == args.attention_target
    ]
    if not attention_modules:
        available = sorted(
            {
                module.__class__.__name__
                for module in model.modules()
                if module.__class__.__name__.endswith("Attention")
            }
        )
        raise ValueError(
            f"no {args.attention_target!r} modules found; available: {available}"
        )

    dataset, source_rows = prepare_dataset(
        dataset_path,
        tokenizer,
        messages_column=args.messages_column,
        samples=args.samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    fp8_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=args.strategy,
        observer=args.observer,
    )
    recipe = QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[args.attention_target],
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
    )

    output_path.mkdir(parents=True)
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    config = json.loads((output_path / "config.json").read_text())
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError("calibrated checkpoint has no quantization_config")
    kv_cache_scheme = quantization_config.get("kv_cache_scheme")
    if not isinstance(kv_cache_scheme, dict):
        raise RuntimeError("calibrated checkpoint has no kv_cache_scheme")
    expected_scheme = {
        "num_bits": 8,
        "type": "float",
        "strategy": args.strategy,
        "observer": args.observer,
        "dynamic": False,
        "symmetric": True,
    }
    for field, expected in expected_scheme.items():
        if kv_cache_scheme.get(field) != expected:
            raise RuntimeError(
                f"calibrated kv_cache_scheme has {field}="
                f"{kv_cache_scheme.get(field)!r}; expected {expected!r}"
            )
    scales = scale_manifest(output_path)
    expected_scale_count = len(attention_modules)
    for kind in ("q", "k", "v"):
        if scales[kind] != expected_scale_count:
            raise RuntimeError(
                f"calibrated checkpoint has {scales[kind]} {kind}_scale tensors; "
                f"expected {expected_scale_count}"
            )

    result = {
        "schema_version": 1,
        "tool": file_manifest(Path(__file__)),
        "calibration": {
            "attention_target": args.attention_target,
            "attention_module_count": len(attention_modules),
            "strategy": args.strategy,
            "observer": args.observer,
            "samples": args.samples,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "messages_column": args.messages_column,
            "source_rows": source_rows,
            "elapsed_seconds": time.time() - started_at,
        },
        "source": {
            "model": str(model_path),
            "model_revision": args.model_revision,
            "checkpoint": source_checkpoint,
            "config": file_manifest(model_path / "config.json"),
            "tokenizer": file_manifest(model_path / "tokenizer.json"),
            "dataset": {
                "path": str(dataset_path),
                "bytes": dataset_path.stat().st_size,
                "sha256": dataset_sha256,
            },
        },
        "output": {
            "path": str(output_path),
            "checkpoint": checkpoint_manifest(output_path),
            "config": file_manifest(output_path / "config.json"),
            "tokenizer": file_manifest(output_path / "tokenizer.json"),
            "kv_cache_scheme": kv_cache_scheme,
            "scales": scales,
        },
        "packages": package_versions(),
    }
    (output_path / "loom-calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
