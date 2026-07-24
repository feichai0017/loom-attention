"""Select a deterministic language-model quality corpus from Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-column")
    source.add_argument("--messages-column")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="Optional loom-calibration.json whose exact source rows are excluded.",
    )
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sequences <= 0:
        parser.error("--sequences must be positive")
    if args.min_tokens < 2:
        parser.error("--min-tokens must be at least two")
    if args.max_tokens < args.min_tokens:
        parser.error("--max-tokens must be at least --min-tokens")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def deterministic_rank(seed: int, row: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(str(seed).encode())
    digest.update(b"\0")
    digest.update(str(row).encode())
    return digest.digest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model_path = args.model.resolve(strict=True)
    dataset_path = args.dataset_parquet.resolve(strict=True)
    output_path = args.output.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_path),
        split="train",
    )
    dataset_rows = len(dataset)
    source_column = args.text_column or args.messages_column
    if source_column not in dataset.column_names:
        raise ValueError(
            f"dataset has no {source_column!r} column: {dataset.column_names}"
        )

    dataset_sha256 = sha256_file(dataset_path)
    tokenizer_json = model_path / "tokenizer.json"
    tokenizer_sha256 = (
        sha256_file(tokenizer_json) if tokenizer_json.is_file() else None
    )
    calibration = None
    excluded_rows: set[int] = set()
    if args.calibration_manifest is not None:
        calibration_path = args.calibration_manifest.resolve(strict=True)
        calibration = json.loads(calibration_path.read_text())
        calibrated_dataset_sha256 = calibration["source"]["dataset"]["sha256"]
        if calibrated_dataset_sha256 != dataset_sha256:
            raise ValueError(
                "calibration manifest and quality source have different SHA-256"
            )
        calibrated_tokenizer_sha256 = calibration["source"]["tokenizer"]["sha256"]
        if calibrated_tokenizer_sha256 != tokenizer_sha256:
            raise ValueError(
                "calibration manifest and quality tokenizer have different SHA-256"
            )
        source_rows = calibration["calibration"].get("source_rows")
        if not isinstance(source_rows, list) or not all(
            type(row) is int for row in source_rows
        ):
            raise ValueError("calibration manifest does not record source rows")
        if len(source_rows) != len(set(source_rows)):
            raise ValueError("calibration manifest records duplicate source rows")
        if any(row < 0 or row >= dataset_rows for row in source_rows):
            raise ValueError("calibration manifest records an out-of-range source row")
        excluded_rows = set(source_rows)

    ranked_rows = sorted(
        (
            (deterministic_rank(args.seed, row), row)
            for row in range(dataset_rows)
            if row not in excluded_rows
        ),
        key=lambda item: item[0],
    )
    selected = []
    rows_examined = 0
    for _rank, row in ranked_rows:
        rows_examined += 1
        value = dataset[row][source_column]
        if args.messages_column is not None:
            text = tokenizer.apply_chat_template(value, tokenize=False)
        else:
            text = value
        if not isinstance(text, str) or not text.strip():
            continue
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count < args.min_tokens:
            continue
        selected.append(
            {
                "row": row,
                "text": text,
                "original_tokens": token_count,
                "scored_tokens": min(token_count, args.max_tokens),
            }
        )
        if len(selected) == args.sequences:
            break
    if len(selected) < args.sequences:
        raise ValueError(
            f"only {len(selected)} rows contain at least "
            f"{args.min_tokens} tokens; requested {args.sequences}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        for item in selected:
            output.write(
                json.dumps(
                    {
                        "text": item["text"],
                        "source_row": item["row"],
                        "original_tokens": item["original_tokens"],
                        "scored_tokens": item["scored_tokens"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    result = {
        "schema_version": 1,
        "tool": file_manifest(Path(__file__)),
        "source": {
            "dataset": str(dataset_path),
            "dataset_bytes": dataset_path.stat().st_size,
            "dataset_sha256": dataset_sha256,
            "tokenizer": str(model_path),
            "tokenizer_json_sha256": tokenizer_sha256,
            "calibration_manifest": (
                {
                    "path": str(args.calibration_manifest.resolve(strict=True)),
                    "sha256": sha256_file(
                        args.calibration_manifest.resolve(strict=True)
                    ),
                }
                if calibration is not None
                else None
            ),
        },
        "selection": {
            "source_kind": (
                "messages" if args.messages_column is not None else "text"
            ),
            "source_column": source_column,
            "seed": args.seed,
            "rank": "sha256(seed,row)",
            "excluded_calibration_rows": len(excluded_rows),
            "available_rows": len(ranked_rows),
            "rows_examined": rows_examined,
            "sequences": args.sequences,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "scored_tokens": sum(item["scored_tokens"] for item in selected),
            "source_rows": [item["row"] for item in selected],
            "sequence_token_counts": [
                item["scored_tokens"] for item in selected
            ],
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
