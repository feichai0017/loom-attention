#!/usr/bin/env python3
"""Compare Loom's greedy speculative verifier with vLLM's Triton kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import (
    bridge_abi_version,
    greedy_speculative_verify,
)
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    rejection_greedy_sample_kernel,
)


def parse_positive_list(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not values or min(values) <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batches", default="1,8,32,128,256")
    parser.add_argument("--draft-lengths", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.samples) <= 0:
        parser.error("warmup, iterations, and samples must be positive")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def elapsed_microseconds(
    operation: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iterations)


def benchmark_case(
    batch: int,
    draft_length: int,
    warmup: int,
    iterations: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed + batch * 1009 + draft_length)
    draft = torch.randint(
        0,
        151936,
        (batch * draft_length,),
        dtype=torch.int32,
        device="cuda",
    )
    target = draft.to(torch.int64)
    # Keep two thirds of requests fully accepted and reject the rest at the
    # second token (or the first token when K=1).
    mismatch_position = min(1, draft_length - 1)
    mismatch_requests = (
        torch.arange(2, batch, 3, device="cuda")
        if batch > 2
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    if mismatch_requests.numel() > 0:
        mismatch_indices = mismatch_requests * draft_length + mismatch_position
        target[mismatch_indices] = (
            target[mismatch_indices] + 1
        ) % 151936
    bonus = torch.randint(
        0, 151936, (batch, 1), dtype=torch.int32, device="cuda"
    )
    cumulative = (
        torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
        * draft_length
    )

    def baseline() -> torch.Tensor:
        output = torch.full(
            (batch, draft_length + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device="cuda",
        )
        rejection_greedy_sample_kernel[(batch,)](
            output,
            cumulative,
            draft,
            target,
            bonus,
            None,
            draft_length,
            None,
            None,
            SYNTHETIC_MODE=False,
        )
        return output

    def loom() -> torch.Tensor:
        return greedy_speculative_verify(
            draft,
            target,
            bonus,
            cumulative,
            draft_length,
        )[0]

    expected = baseline()
    actual = loom()
    torch.cuda.synchronize()
    if not torch.equal(actual, expected):
        raise AssertionError("Loom speculative output differs from vLLM")

    baseline_samples: list[float] = []
    loom_samples: list[float] = []
    for sample in range(samples):
        if sample % 2 == 0:
            baseline_samples.append(
                elapsed_microseconds(
                    baseline, warmup=warmup, iterations=iterations
                )
            )
            loom_samples.append(
                elapsed_microseconds(
                    loom, warmup=warmup, iterations=iterations
                )
            )
        else:
            loom_samples.append(
                elapsed_microseconds(
                    loom, warmup=warmup, iterations=iterations
                )
            )
            baseline_samples.append(
                elapsed_microseconds(
                    baseline, warmup=warmup, iterations=iterations
                )
            )

    baseline_median = statistics.median(baseline_samples)
    loom_median = statistics.median(loom_samples)
    return {
        "batch": batch,
        "draft_length": draft_length,
        "draft_tokens": batch * draft_length,
        "baseline_us": baseline_median,
        "loom_us": loom_median,
        "speedup": baseline_median / loom_median,
        "latency_reduction_percent": (
            (baseline_median - loom_median) / baseline_median * 100.0
        ),
        "baseline_samples_us": baseline_samples,
        "loom_samples_us": loom_samples,
        "baseline_p90_us": percentile(baseline_samples, 0.9),
        "loom_p90_us": percentile(loom_samples, 0.9),
        "fully_accepted_requests": batch - len(range(2, batch, 3)),
        "mismatched_requests": len(range(2, batch, 3)),
        "baseline_output_bytes": batch * (draft_length + 1) * 4,
        "loom_combined_output_and_metadata_bytes": (
            batch * (draft_length + 3) * 4
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm

    batches = parse_positive_list(args.batches, "batches")
    draft_lengths = parse_positive_list(
        args.draft_lengths, "draft-lengths"
    )
    results = [
        benchmark_case(
            batch,
            draft_length,
            args.warmup,
            args.iterations,
            args.samples,
            args.seed,
        )
        for draft_length in draft_lengths
        for batch in batches
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "deterministic greedy speculative verify and compact",
        "baseline": (
            f"vLLM {vllm.__version__} rejection_greedy_sample_kernel "
            "with torch.full output"
        ),
        "candidate": (
            "Loom greedy_speculative_verify with one combined caller-owned "
            "output allocation"
        ),
        "contract": {
            "draft_token_ids": "flattened contiguous int32",
            "target_token_ids": "flattened contiguous int64 argmax IDs",
            "bonus_token_ids": "contiguous int32 [batch, 1]",
            "cumulative_draft_lengths": "inclusive contiguous int32 [batch]",
            "output_token_ids": "contiguous int32 [batch, K + 1], -1 padded",
        },
        "scope": {
            "batches": batches,
            "draft_lengths": draft_lengths,
            "tested_cases": len(results),
            "acceptance_pattern": (
                "two of every three requests fully accepted; remaining "
                "requests mismatch at position min(1, K - 1)"
            ),
        },
        "timing": {
            "warmup": args.warmup,
            "iterations_per_sample": args.iterations,
            "samples": args.samples,
            "provider_order": "alternated for every sample",
            "method": (
                "CUDA events over Python-dispatched calls including output "
                "allocation/fill and kernel execution"
            ),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi_version": bridge_abi_version(),
        },
        "acceptance": {
            "passed": True,
            "output": "bit exact",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
