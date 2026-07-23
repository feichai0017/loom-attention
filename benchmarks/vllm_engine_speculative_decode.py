"""Run a process-isolated real-engine speculative-decoding comparison.

The controller launches three independent providers:

* target-only vLLM;
* native vLLM draft-model speculative decoding;
* the same speculative configuration with Loom's deterministic greedy
  verification boundary registered before engine construction.

Every provider receives identical prompt token IDs, generation settings, and
cache isolation. Correctness and path evidence are acceptance gates; a
performance improvement is reported, never assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


PROVIDERS = (
    "target-only",
    "vllm-speculative",
    "loom-speculative",
)
SPECULATIVE_PROVIDERS = frozenset(PROVIDERS[1:])
NATURAL_PROMPT_TEMPLATES = (
    (
        "Explain how memory bandwidth, arithmetic intensity, and kernel launch "
        "overhead interact during large language model decoding."
    ),
    (
        "Compare paged key-value caches with contiguous caches, including "
        "their effects on scheduling, fragmentation, and memory movement."
    ),
    (
        "Write a concise engineering note about deterministic GPU sampling, "
        "reproducibility, and counter-based random number generation."
    ),
    (
        "Describe a safe zero-copy boundary between a Rust CUDA library and a "
        "Python inference engine that owns tensors and CUDA streams."
    ),
    (
        "Analyze when speculative decoding improves token latency and when "
        "draft-model cost or low acceptance can make it slower."
    ),
    (
        "Summarize the responsibilities of vendor GEMM libraries and the "
        "memory-bound fused operators that surround matrix multiplication."
    ),
    (
        "Discuss how prefix caching and preemptive scheduling create gather, "
        "scatter, compact, and remap work for a paged key-value cache."
    ),
    (
        "Explain why an engine benchmark should isolate providers, reverse "
        "execution order, preserve tokens, and record exact path evidence."
    ),
)


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    input_len: int
    output_len: int

    @property
    def label(self) -> str:
        return f"b{self.batch_size}-in{self.input_len}-out{self.output_len}"

    @property
    def argument(self) -> str:
        return f"{self.batch_size}x{self.input_len}x{self.output_len}"


def parse_case(value: str) -> BenchmarkCase:
    try:
        dimensions = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be BATCHxINPUTxOUTPUT"
        ) from error
    if len(dimensions) != 3 or min(dimensions) <= 0:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT")
    return BenchmarkCase(*dimensions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--tested-revision", default="worktree")
    parser.add_argument("--target-revision", default="")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-revision", default="")
    parser.add_argument("--spec-tokens", type=int, default=4)
    parser.add_argument(
        "--prompt-mode",
        choices=("natural", "synthetic"),
        default="natural",
    )
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--boundary-profile-repeats", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--provider-order",
        choices=("native-first", "loom-first"),
        default="native-first",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-provider",
        choices=PROVIDERS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-result",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-cache-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 128, 128),
            BenchmarkCase(8, 128, 128),
        ]
    if args.spec_tokens <= 0:
        parser.error("spec-tokens must be positive")
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if args.boundary_profile_repeats < 0:
        parser.error("boundary-profile-repeats must be non-negative")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error("internal runs require result and cache paths")
    return args


def summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def make_prompts(
    case: BenchmarkCase,
    prompt_mode: str,
    tokenizer: Any | None,
) -> tuple[list[dict[str, list[int]]], str]:
    prompt_token_ids: list[list[int]] = []
    for batch_index in range(case.batch_size):
        if prompt_mode == "synthetic":
            tokens = [
                3 + ((batch_index * 17 + position * 13) % 1000)
                for position in range(case.input_len)
            ]
        else:
            if tokenizer is None:
                raise RuntimeError("natural prompts require a tokenizer")
            text = (
                NATURAL_PROMPT_TEMPLATES[
                    batch_index % len(NATURAL_PROMPT_TEMPLATES)
                ]
                + f"\nDeterministic request index: {batch_index}."
            )
            unit = list(
                tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
            if not unit:
                raise RuntimeError("tokenizer returned an empty natural prompt")
            repetitions = (case.input_len + len(unit) - 1) // len(unit)
            tokens = (unit * repetitions)[: case.input_len]
        prompt_token_ids.append(tokens)

    serialized = json.dumps(
        prompt_token_ids,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(serialized).hexdigest()
    return (
        [
            {"prompt_token_ids": tokens}
            for tokens in prompt_token_ids
        ],
        fingerprint,
    )


def request_metrics(
    outputs: list[Any],
) -> tuple[list[float], list[float], list[float]]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    e2e_ms: list[float] = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            continue
        if metrics.first_token_latency > 0.0:
            ttft = metrics.first_token_latency * 1000.0
            ttft_ms.append(ttft)
        else:
            ttft = 0.0
        generated = metrics.num_generation_tokens
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        if generated > 1 and decode_seconds >= 0.0:
            tpot_ms.append(decode_seconds * 1000.0 / (generated - 1))
        if ttft > 0.0 and decode_seconds >= 0.0:
            e2e_ms.append(ttft + decode_seconds * 1000.0)
    return ttft_ms, tpot_ms, e2e_ms


class RejectionCollector:
    """Collect path and acceptance evidence without synchronizing timed calls."""

    def __init__(self) -> None:
        self._installed = False
        self._cuda_timing_enabled = False
        self.reset()

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("rejection collector is already installed")
        from vllm.v1.sample import rejection_sampler

        original = rejection_sampler.rejection_sample

        @wraps(original)
        def tracked(*args: Any, **kwargs: Any) -> Any:
            event_pair = None
            if self._cuda_timing_enabled:
                import torch

                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                event_pair = (started, finished)
            output = original(*args, **kwargs)
            if event_pair is not None:
                event_pair[1].record()
                self.cuda_event_pairs.append(event_pair)
            if len(args) >= 2:
                draft_lengths = args[1]
            else:
                draft_lengths = kwargs["num_draft_tokens"]
            self.calls += 1
            self.draft_lengths.extend(int(length) for length in draft_lengths)
            # Retaining these small output tensors avoids a device
            # synchronization inside the timed rejection boundary.
            self.outputs.append(output.detach())
            return output

        rejection_sampler.rejection_sample = tracked
        self._installed = True

    def reset(self) -> None:
        self.calls = 0
        self.draft_lengths: list[int] = []
        self.outputs: list[Any] = []
        self.cuda_event_pairs: list[tuple[Any, Any]] = []

    def enable_cuda_timing(self, enabled: bool) -> None:
        self._cuda_timing_enabled = enabled

    def cuda_event_count(self) -> int:
        return len(self.cuda_event_pairs)

    def cuda_elapsed_ms_since(self, start: int) -> float:
        return sum(
            started.elapsed_time(finished)
            for started, finished in self.cuda_event_pairs[start:]
        )

    def report(self, spec_tokens: int) -> dict[str, Any]:
        accepted_lengths: list[int] = []
        for output in self.outputs:
            valid_lengths = (output != -1).sum(dim=1)
            accepted_lengths.extend(
                int(length)
                for length in (valid_lengths - 1).clamp_min(0).cpu().tolist()
            )

        requests = len(self.draft_lengths)
        proposed = sum(self.draft_lengths)
        accepted = sum(accepted_lengths)
        drafted_by_position = [
            sum(length > position for length in self.draft_lengths)
            for position in range(spec_tokens)
        ]
        accepted_by_position = [
            sum(length > position for length in accepted_lengths)
            for position in range(spec_tokens)
        ]
        per_position_rates = [
            (
                accepted_count / drafted_count
                if drafted_count > 0
                else None
            )
            for accepted_count, drafted_count in zip(
                accepted_by_position,
                drafted_by_position,
                strict=True,
            )
        ]
        return {
            "rejection_calls": self.calls,
            "draft_requests": requests,
            "proposed_draft_tokens": proposed,
            "accepted_draft_tokens": accepted,
            "draft_acceptance_rate": (
                accepted / proposed if proposed > 0 else None
            ),
            "mean_acceptance_length_including_bonus": (
                1.0 + accepted / requests if requests > 0 else None
            ),
            "drafted_tokens_per_position": drafted_by_position,
            "accepted_tokens_per_position": accepted_by_position,
            "acceptance_rate_per_position": per_position_rates,
            "counter_method": (
                "the rejection wrapper retains output tensors during timed "
                "calls and reads accepted lengths only after CUDA synchronize"
            ),
        }


def run_case(
    engine: Any,
    sampling_type: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
    collector: RejectionCollector,
    launch_count_fn: Callable[[], int],
    tokenizer: Any | None,
    boundary_profile_repeats: int,
) -> dict[str, Any]:
    import torch

    prompts, prompt_fingerprint = make_prompts(
        case,
        args.prompt_mode,
        tokenizer,
    )
    sampling = sampling_type(
        temperature=0.0,
        max_tokens=case.output_len,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        engine.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()

    collector.reset()
    launches_before = launch_count_fn()
    latency_ms: list[float] = []
    throughput: list[float] = []
    all_ttft_ms: list[float] = []
    all_tpot_ms: list[float] = []
    all_e2e_ms: list[float] = []
    token_ids: list[list[int]] | None = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)
        throughput.append(
            case.batch_size * case.output_len / (elapsed_ms / 1000.0)
        )
        ttft_ms, tpot_ms, e2e_ms = request_metrics(outputs)
        all_ttft_ms.extend(ttft_ms)
        all_tpot_ms.extend(tpot_ms)
        all_e2e_ms.extend(e2e_ms)
        current_token_ids = [
            list(request.outputs[0].token_ids) for request in outputs
        ]
        if any(len(tokens) != case.output_len for tokens in current_token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")
        if token_ids is not None and token_ids != current_token_ids:
            raise RuntimeError("fixed-seed greedy output changed across repeats")
        token_ids = current_token_ids

    launches_after = launch_count_fn()
    speculative_stats = collector.report(args.spec_tokens)
    boundary_profile = run_boundary_profile(
        engine,
        sampling,
        prompts,
        boundary_profile_repeats,
        collector,
        args.spec_tokens,
        token_ids,
    )
    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "prompt_mode": args.prompt_mode,
        "prompt_token_ids_sha256": prompt_fingerprint,
        "batch_latency_ms": summary(latency_ms),
        "request_ttft_ms": summary(all_ttft_ms),
        "request_tpot_ms": summary(all_tpot_ms),
        "request_e2e_ms": summary(all_e2e_ms),
        "output_tokens_per_second": summary(throughput),
        "token_ids": token_ids,
        "speculative_stats": speculative_stats,
        "boundary_profile": boundary_profile,
        "measured_loom_host_launches": launches_after - launches_before,
    }


def run_boundary_profile(
    engine: Any,
    sampling: Any,
    prompts: list[dict[str, list[int]]],
    repeats: int,
    collector: RejectionCollector,
    spec_tokens: int,
    expected_token_ids: list[list[int]] | None,
) -> dict[str, Any] | None:
    if repeats == 0:
        return None

    import torch

    collector.reset()
    collector.enable_cuda_timing(True)
    batch_latency_ms: list[float] = []
    rejection_cuda_ms: list[float] = []
    rejection_share: list[float] = []
    calls_per_generate: list[int] = []
    for _ in range(repeats):
        event_start = collector.cuda_event_count()
        calls_before = collector.calls
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        current_token_ids = [
            list(request.outputs[0].token_ids) for request in outputs
        ]
        if (
            expected_token_ids is not None
            and current_token_ids != expected_token_ids
        ):
            raise RuntimeError("boundary profile changed fixed greedy output")
        boundary_ms = collector.cuda_elapsed_ms_since(event_start)
        batch_latency_ms.append(elapsed_ms)
        rejection_cuda_ms.append(boundary_ms)
        rejection_share.append(
            boundary_ms / elapsed_ms if elapsed_ms > 0.0 else 0.0
        )
        calls_per_generate.append(collector.calls - calls_before)
    collector.enable_cuda_timing(False)
    return {
        "repeats": repeats,
        "batch_latency_ms": summary(batch_latency_ms),
        "rejection_boundary_cuda_ms": summary(rejection_cuda_ms),
        "rejection_boundary_share_of_batch_latency": summary(
            rejection_share
        ),
        "rejection_calls_per_generate": calls_per_generate,
        "speculative_stats": collector.report(spec_tokens),
        "method": (
            "CUDA events bracket the process-local rejection_sample function "
            "after primary timing; event instrumentation is not part of the "
            "reported benchmark latency samples"
        ),
    }


def prepare_environment(cache_root: Path) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    required = [
        str(Path(sys.executable).absolute().parent),
        str(cuda_home / "bin"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required if entry not in current_entries]
        + current_entries
    )


def resolve_model(value: str) -> tuple[str, str]:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve()), "local-checkpoint"
    return value, "huggingface"


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None and args.internal_cache_root is not None
    prepare_environment(args.internal_cache_root.resolve())

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import (
        provider_metadata,
        register_vllm_greedy_speculative_verify,
    )

    explicit_registration = None
    if provider == "loom-speculative":
        explicit_registration = register_vllm_greedy_speculative_verify()
        if explicit_registration is None:
            raise RuntimeError(
                "Loom greedy speculative verification registration failed"
            )

    collector = RejectionCollector()
    collector.install()
    operator = Operator.GREEDY_SPECULATIVE_VERIFY
    reset_launch_count(operator)
    launch_count_fn = lambda: launch_count(operator)

    target_model, target_kind = resolve_model(args.target_model)
    draft_model, draft_kind = resolve_model(args.draft_model)
    tokenizer = None
    if args.prompt_mode == "natural":
        from transformers import AutoTokenizer

        tokenizer_arguments: dict[str, Any] = {
            "local_files_only": target_kind == "local-checkpoint",
        }
        if args.target_revision and target_kind == "huggingface":
            tokenizer_arguments["revision"] = args.target_revision
        tokenizer = AutoTokenizer.from_pretrained(
            target_model,
            **tokenizer_arguments,
        )
    max_model_len = max(
        case.input_len + case.output_len for case in args.cases
    )
    max_num_seqs = max(case.batch_size for case in args.cases)
    max_num_batched_tokens = (
        max(case.batch_size * case.input_len for case in args.cases)
        + args.spec_tokens * max_num_seqs
    )
    engine_arguments: dict[str, Any] = {
        "model": target_model,
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
    }
    if provider in SPECULATIVE_PROVIDERS:
        engine_arguments.update(
            spec_method="draft_model",
            spec_model=draft_model,
            spec_tokens=args.spec_tokens,
        )
    engine = LLM(**engine_arguments)
    launches_after_engine_init = launch_count_fn()
    effective_profile_repeats = (
        0
        if provider == "target-only"
        else args.boundary_profile_repeats
    )
    cases = [
        run_case(
            engine,
            SamplingParams,
            case,
            args,
            collector,
            launch_count_fn,
            tokenizer,
            effective_profile_repeats,
        )
        for case in args.cases
    ]
    total_host_launch_count = launch_count_fn()
    measured_host_launch_count = sum(
        case["measured_loom_host_launches"] for case in cases
    )
    report = {
        "provider": provider,
        "tested_revision": args.tested_revision,
        "target_model": target_model,
        "target_model_source": args.target_model,
        "target_model_kind": target_kind,
        "target_revision": args.target_revision,
        "draft_model": (
            draft_model if provider in SPECULATIVE_PROVIDERS else None
        ),
        "draft_model_source": (
            args.draft_model if provider in SPECULATIVE_PROVIDERS else None
        ),
        "draft_model_kind": (
            draft_kind if provider in SPECULATIVE_PROVIDERS else None
        ),
        "draft_revision": (
            args.draft_revision if provider in SPECULATIVE_PROVIDERS else None
        ),
        "speculative_config": (
            {
                "method": "draft_model",
                "num_speculative_tokens": args.spec_tokens,
                "draft_sample_method": "greedy",
                "rejection_sample_method": "standard",
            }
            if provider in SPECULATIVE_PROVIDERS
            else None
        ),
        "sampling": (
            "temperature=0, max_tokens=case.output_len, "
            f"engine_seed={args.seed}, ignore_eos=true"
        ),
        "prompt_mode": args.prompt_mode,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "boundary_profile_repeats": effective_profile_repeats,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "cases": cases,
        "loom_path": {
            "explicit_registration": explicit_registration,
            "launches_after_engine_init": launches_after_engine_init,
            "measured_host_launch_count": measured_host_launch_count,
            "total_host_launch_count": total_host_launch_count,
            "provider_metadata": provider_metadata(),
            "counter_semantics": (
                "total includes warmup, measured, and optional profile "
                "submissions; measured count excludes warmup and profile by "
                "taking its delta first"
            ),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "cuda_home": os.environ["CUDA_HOME"],
            "v1_multiprocessing": os.environ[
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ],
            "vllm_cache_root": os.environ["VLLM_CACHE_ROOT"],
        },
    }
    assert args.internal_result is not None
    args.internal_result.parent.mkdir(parents=True, exist_ok=True)
    args.internal_result.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    rejection_calls = sum(
        case["speculative_stats"]["rejection_calls"] for case in cases
    )
    print(
        f"provider={provider} "
        f"rejection_calls={rejection_calls} "
        f"loom_measured_launches={measured_host_launch_count}",
        file=sys.stderr,
    )
    return report


def child_command(
    args: argparse.Namespace,
    provider: str,
    result: Path,
    cache_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--target-model",
        args.target_model,
        "--tested-revision",
        args.tested_revision,
        "--target-revision",
        args.target_revision,
        "--draft-model",
        args.draft_model,
        "--draft-revision",
        args.draft_revision,
        "--spec-tokens",
        str(args.spec_tokens),
        "--prompt-mode",
        args.prompt_mode,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--boundary-profile-repeats",
        str(args.boundary_profile_repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--internal-provider",
        provider,
        "--internal-result",
        str(result),
        "--internal-cache-root",
        str(cache_root),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    for case in args.cases:
        command.extend(("--case", case.argument))
    return command


def ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def median_metric(case: dict[str, Any], name: str) -> float | None:
    metric = case[name]
    return None if metric is None else float(metric["median"])


def median_profile_metric(
    case: dict[str, Any],
    name: str,
) -> float | None:
    profile = case["boundary_profile"]
    if profile is None:
        return None
    metric = profile[name]
    return None if metric is None else float(metric["median"])


def compare_reports(
    reports: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, bool, bool]:
    comparisons: list[dict[str, Any]] = []
    all_tokens_match = True
    all_prompt_ids_match = True
    speculative_stats_match = True
    provider_cases = [
        reports[provider]["cases"] for provider in PROVIDERS
    ]
    for target, native, loom in zip(*provider_cases, strict=True):
        if not (target["case"] == native["case"] == loom["case"]):
            raise RuntimeError("provider reports contain different cases")
        prompt_ids_match = (
            target["prompt_token_ids_sha256"]
            == native["prompt_token_ids_sha256"]
            == loom["prompt_token_ids_sha256"]
        )
        all_prompt_ids_match = all_prompt_ids_match and prompt_ids_match
        target_vs_native = target["token_ids"] == native["token_ids"]
        target_vs_loom = target["token_ids"] == loom["token_ids"]
        native_vs_loom = native["token_ids"] == loom["token_ids"]
        case_tokens_match = (
            target_vs_native and target_vs_loom and native_vs_loom
        )
        all_tokens_match = all_tokens_match and case_tokens_match

        native_stats = native["speculative_stats"]
        loom_stats = loom["speculative_stats"]
        case_stats_match = all(
            native_stats[name] == loom_stats[name]
            for name in (
                "draft_requests",
                "proposed_draft_tokens",
                "accepted_draft_tokens",
                "drafted_tokens_per_position",
                "accepted_tokens_per_position",
            )
        )
        speculative_stats_match = (
            speculative_stats_match and case_stats_match
        )

        target_batch = median_metric(target, "batch_latency_ms")
        native_batch = median_metric(native, "batch_latency_ms")
        loom_batch = median_metric(loom, "batch_latency_ms")
        target_tpot = median_metric(target, "request_tpot_ms")
        native_tpot = median_metric(native, "request_tpot_ms")
        loom_tpot = median_metric(loom, "request_tpot_ms")
        target_throughput = median_metric(
            target,
            "output_tokens_per_second",
        )
        native_throughput = median_metric(
            native,
            "output_tokens_per_second",
        )
        loom_throughput = median_metric(
            loom,
            "output_tokens_per_second",
        )
        native_rejection_ms = median_profile_metric(
            native,
            "rejection_boundary_cuda_ms",
        )
        loom_rejection_ms = median_profile_metric(
            loom,
            "rejection_boundary_cuda_ms",
        )
        native_rejection_share = median_profile_metric(
            native,
            "rejection_boundary_share_of_batch_latency",
        )
        loom_rejection_share = median_profile_metric(
            loom,
            "rejection_boundary_share_of_batch_latency",
        )
        comparisons.append(
            {
                "case": target["case"],
                "prompt_token_ids_match": prompt_ids_match,
                "prompt_token_ids_sha256": target[
                    "prompt_token_ids_sha256"
                ],
                "token_ids": {
                    "target_vs_native_match": target_vs_native,
                    "target_vs_loom_match": target_vs_loom,
                    "native_vs_loom_match": native_vs_loom,
                },
                "speculative_stats_match": case_stats_match,
                "target_over_native_batch_latency": ratio(
                    target_batch,
                    native_batch,
                ),
                "target_over_loom_batch_latency": ratio(
                    target_batch,
                    loom_batch,
                ),
                "native_over_loom_batch_latency": ratio(
                    native_batch,
                    loom_batch,
                ),
                "target_over_native_tpot": ratio(
                    target_tpot,
                    native_tpot,
                ),
                "target_over_loom_tpot": ratio(
                    target_tpot,
                    loom_tpot,
                ),
                "native_over_loom_tpot": ratio(
                    native_tpot,
                    loom_tpot,
                ),
                "native_over_target_output_throughput": ratio(
                    native_throughput,
                    target_throughput,
                ),
                "loom_over_target_output_throughput": ratio(
                    loom_throughput,
                    target_throughput,
                ),
                "loom_over_native_output_throughput": ratio(
                    loom_throughput,
                    native_throughput,
                ),
                "native_rejection_boundary_cuda_ms": native_rejection_ms,
                "loom_rejection_boundary_cuda_ms": loom_rejection_ms,
                "native_over_loom_rejection_boundary_cuda_time": ratio(
                    native_rejection_ms,
                    loom_rejection_ms,
                ),
                "native_rejection_share_of_batch_latency": (
                    native_rejection_share
                ),
                "loom_rejection_share_of_batch_latency": (
                    loom_rejection_share
                ),
                "performance_is_acceptance_gate": False,
            }
        )
    return (
        comparisons,
        all_tokens_match,
        all_prompt_ids_match,
        speculative_stats_match,
    )


def total_rejection_calls(report: dict[str, Any]) -> int:
    return sum(
        case["speculative_stats"]["rejection_calls"]
        for case in report["cases"]
    )


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    order = (
        ["target-only", "vllm-speculative", "loom-speculative"]
        if args.provider_order == "native-first"
        else ["target-only", "loom-speculative", "vllm-speculative"]
    )
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix="loom-vllm-speculative-"
    ) as directory:
        root = Path(directory)
        for provider in order:
            result = root / f"{provider}.json"
            cache_root = root / f"{provider}-cache"
            subprocess.run(
                child_command(args, provider, result, cache_root),
                check=True,
            )
            reports[provider] = json.loads(
                result.read_text(encoding="utf-8")
            )

    (
        comparisons,
        tokens_match,
        prompt_ids_match,
        speculative_stats_match,
    ) = compare_reports(reports)
    target = reports["target-only"]
    native = reports["vllm-speculative"]
    loom = reports["loom-speculative"]
    target_has_no_speculative_calls = total_rejection_calls(target) == 0
    native_speculative_path_observed = total_rejection_calls(native) > 0
    loom_speculative_path_observed = total_rejection_calls(loom) > 0
    native_uses_vllm_verifier = not native["loom_path"][
        "provider_metadata"
    ]["greedy_speculative_verify_override"]
    loom_registered = (
        loom["loom_path"]["explicit_registration"]
        == "greedy_speculative_verify"
        and loom["loom_path"]["provider_metadata"][
            "greedy_speculative_verify_override"
        ]
    )
    loom_contract_observed = (
        loom["loom_path"]["provider_metadata"][
            "greedy_speculative_verify_first_contract"
        ]
        is not None
    )
    loom_no_lifetime_contract_rejection = (
        loom["loom_path"]["provider_metadata"][
            "greedy_speculative_verify_first_rejection"
        ]
        is None
    )
    target_host_launches = target["loom_path"][
        "measured_host_launch_count"
    ]
    native_host_launches = native["loom_path"][
        "measured_host_launch_count"
    ]
    loom_host_launches = loom["loom_path"][
        "measured_host_launch_count"
    ]
    loom_measured_full_coverage = (
        loom_host_launches == total_rejection_calls(loom)
    )
    provider_isolation = (
        target_host_launches == 0
        and native_host_launches == 0
        and loom_host_launches > 0
    )
    accepted = all(
        (
            tokens_match,
            prompt_ids_match,
            speculative_stats_match,
            target_has_no_speculative_calls,
            native_speculative_path_observed,
            loom_speculative_path_observed,
            native_uses_vllm_verifier,
            loom_registered,
            loom_contract_observed,
            loom_measured_full_coverage,
            provider_isolation,
        )
    )
    report = {
        "schema_version": 1,
        "benchmark": "vllm_real_model_speculative_decode",
        "tested_revision": args.tested_revision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_model": args.target_model,
        "target_revision": args.target_revision,
        "draft_model": args.draft_model,
        "draft_revision": args.draft_revision,
        "spec_tokens": args.spec_tokens,
        "prompt_mode": args.prompt_mode,
        "boundary_profile_repeats": args.boundary_profile_repeats,
        "provider_order": order,
        "acceptance": {
            "passed": accepted,
            "token_ids_match_across_all_providers": tokens_match,
            "prompt_token_ids_match_across_all_providers": (
                prompt_ids_match
            ),
            "native_and_loom_speculative_stats_match": (
                speculative_stats_match
            ),
            "target_has_no_speculative_calls": (
                target_has_no_speculative_calls
            ),
            "native_speculative_path_observed": (
                native_speculative_path_observed
            ),
            "loom_speculative_path_observed": (
                loom_speculative_path_observed
            ),
            "native_uses_vllm_verifier": native_uses_vllm_verifier,
            "loom_registered": loom_registered,
            "loom_contract_observed": loom_contract_observed,
            "loom_measured_full_coverage": loom_measured_full_coverage,
            "loom_no_lifetime_contract_rejection": (
                loom_no_lifetime_contract_rejection
            ),
            "lifetime_fallback_is_acceptance_gate": False,
            "provider_isolation": provider_isolation,
            "target_measured_loom_host_launches": target_host_launches,
            "native_measured_loom_host_launches": native_host_launches,
            "loom_measured_host_launches": loom_host_launches,
            "performance_is_acceptance_gate": False,
        },
        "comparisons": comparisons,
        "providers": reports,
    }
    rendered = json.dumps(report, indent=2)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not accepted:
        raise SystemExit("real-model speculative-decode gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is not None:
        run_provider(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
