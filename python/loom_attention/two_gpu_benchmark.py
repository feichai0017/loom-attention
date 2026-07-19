"""Two-GPU NCCL benchmark runtime for the Route-Q data path.

The remote rank owns a sealed KV prefix. The engine rank sends only Q, receives
an output-plus-LSE attention state, merges it with its local active-tail state,
and compares the result with full attention. A Stage-KV baseline transfers the
remote prefix in the opposite direction under the same workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Sequence

from .attention_state import (
    ATTENTION_BACKENDS,
    compute_attention_state,
    merge_attention_states,
)


DTYPE_BYTES = {"float16": 2, "bfloat16": 2}
DTYPE_TOLERANCES = {
    "float16": (2e-3, 2e-3),
    "bfloat16": (2e-2, 2e-2),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    prefix_tokens: int = 4096
    tail_tokens: int = 16
    rows: int = 1
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"
    attention_backend: str = "reference"
    warmup: int = 5
    iterations: int = 20
    seed: int = 7
    atol: float | None = None
    rtol: float | None = None
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if self.dtype in DTYPE_TOLERANCES:
            default_atol, default_rtol = DTYPE_TOLERANCES[self.dtype]
            if self.atol is None:
                object.__setattr__(self, "atol", default_atol)
            if self.rtol is None:
                object.__setattr__(self, "rtol", default_rtol)

    def validate(self) -> None:
        positive = {
            "prefix_tokens": self.prefix_tokens,
            "rows": self.rows,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "iterations": self.iterations,
            "timeout_seconds": self.timeout_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"values must be positive: {', '.join(invalid)}")
        if self.tail_tokens < 0 or self.warmup < 0:
            raise ValueError("tail_tokens and warmup must be non-negative")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("kv_heads must divide query_heads")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"unsupported attention backend: {self.attention_backend}"
            )
        if self.atol is None or self.rtol is None:
            raise ValueError("atol and rtol must be configured")
        if self.atol < 0.0 or self.rtol < 0.0:
            raise ValueError("atol and rtol must be non-negative")


def projected_transfer_bytes(config: BenchmarkConfig) -> dict[str, int]:
    config.validate()
    element_bytes = DTYPE_BYTES[config.dtype]
    query = config.rows * config.query_heads * config.head_dim * element_bytes
    output = config.rows * config.query_heads * config.head_dim * element_bytes
    logsumexp = config.rows * config.query_heads * 4
    attention_state = output + logsumexp
    staged_kv = (
        2
        * config.prefix_tokens
        * config.kv_heads
        * config.head_dim
        * element_bytes
    )
    return {
        "query": query,
        "output": output,
        "logsumexp": logsumexp,
        "attention_state": attention_state,
        "route_query_total": query + attention_state,
        "stage_kv_total": staged_kv,
    }


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": float(median(samples)),
        "p99_ms": percentile(samples, 0.99),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def _torch_dtype(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _make_inputs(torch: Any, config: BenchmarkConfig, device: Any) -> tuple[Any, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    dtype = _torch_dtype(torch, config.dtype)

    def sample(shape: tuple[int, ...]) -> Any:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .to(dtype=dtype)
            .to(device=device)
            .contiguous()
        )

    query = sample((config.rows, config.query_heads, config.head_dim))
    prefix_key = sample((config.prefix_tokens, config.kv_heads, config.head_dim))
    prefix_value = sample((config.prefix_tokens, config.kv_heads, config.head_dim))
    tail_key = sample((config.tail_tokens, config.kv_heads, config.head_dim))
    tail_value = sample((config.tail_tokens, config.kv_heads, config.head_dim))
    return query, prefix_key, prefix_value, tail_key, tail_value


def _batch_send(dist: Any, tensors: Sequence[Any], destination: int) -> None:
    operations = [dist.P2POp(dist.isend, tensor, destination) for tensor in tensors]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _batch_receive(dist: Any, tensors: Sequence[Any], source: int) -> None:
    operations = [dist.P2POp(dist.irecv, tensor, source) for tensor in tensors]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _route_engine(
    torch: Any,
    dist: Any,
    query: Any,
    remote_buffers: tuple[Any, Any],
    local_tail: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    dist.send(query, dst=1)
    _batch_receive(dist, remote_buffers, source=1)
    states = [remote_buffers]
    if config.tail_tokens:
        states.append(
            compute_attention_state(
                torch,
                query,
                local_tail[0],
                local_tail[1],
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
                backend=config.attention_backend,
            )
        )
    return merge_attention_states(
        torch, states, backend=config.attention_backend
    )[0]


def _route_worker(
    torch: Any,
    dist: Any,
    query_buffer: Any,
    prefix: tuple[Any, Any],
    config: BenchmarkConfig,
) -> None:
    dist.recv(query_buffer, src=0)
    state = compute_attention_state(
        torch,
        query_buffer,
        prefix[0],
        prefix[1],
        kv_heads=config.kv_heads,
        scale=config.head_dim**-0.5,
        backend=config.attention_backend,
    )
    _batch_send(dist, state, destination=0)


def _stage_engine(
    torch: Any,
    dist: Any,
    query: Any,
    receive_buffers: tuple[Any, Any],
    full_buffers: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    _batch_receive(dist, receive_buffers, source=1)
    return compute_attention_state(
        torch,
        query,
        full_buffers[0],
        full_buffers[1],
        kv_heads=config.kv_heads,
        scale=config.head_dim**-0.5,
        backend=config.attention_backend,
    )[0]


def _stage_worker(dist: Any, prefix: tuple[Any, Any]) -> None:
    _batch_send(dist, prefix, destination=0)


def _full_attention(
    torch: Any,
    query: Any,
    prefix: tuple[Any, Any],
    local_tail: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    if config.tail_tokens:
        key = torch.cat((prefix[0], local_tail[0]), dim=0)
        value = torch.cat((prefix[1], local_tail[1]), dim=0)
    else:
        key, value = prefix
    return compute_attention_state(
        torch,
        query,
        key,
        value,
        kv_heads=config.kv_heads,
        scale=config.head_dim**-0.5,
        backend="reference",
    )[0]


def _timed_cuda(torch: Any, operation: Any) -> float:
    torch.cuda.synchronize()
    started = perf_counter()
    operation()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1_000.0


def _environment(torch: Any) -> dict[str, Any]:
    devices = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl_version = ".".join(str(part) for part in nccl_version)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": nccl_version,
        "device_peer_access": torch.cuda.can_device_access_peer(0, 1),
        "devices": devices,
    }


def _write_report(
    torch: Any,
    config: BenchmarkConfig,
    route_output: Any,
    stage_output: Any,
    expected: Any,
    route_samples: Sequence[float],
    stage_samples: Sequence[float],
    report_path: str,
) -> None:
    def correctness(output: Any) -> dict[str, Any]:
        difference = (output - expected).abs()
        return {
            "passed": bool(
                torch.allclose(output, expected, atol=config.atol, rtol=config.rtol)
            ),
            "max_absolute_error": float(difference.max().item()),
            "max_relative_error": float(
                (difference / expected.abs().clamp_min(1e-6)).max().item()
            ),
        }

    route_correctness = correctness(route_output)
    stage_correctness = correctness(stage_output)
    passed = route_correctness["passed"] and stage_correctness["passed"]
    route = _latency_summary(route_samples)
    stage = _latency_summary(stage_samples)
    route["payload_bytes"] = projected_transfer_bytes(config)["route_query_total"]
    stage["payload_bytes"] = projected_transfer_bytes(config)["stage_kv_total"]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "implementation": {
            "transport": "torch.distributed NCCL point-to-point",
            "attention_backend": config.attention_backend,
            "attention_kernel": (
                "FlashInfer single_decode_with_kv_cache"
                if config.attention_backend == "flashinfer"
                else "PyTorch CUDA einsum output-plus-LSE reference"
            ),
            "merge_kernel": (
                "FlashInfer merge_states"
                if config.attention_backend == "flashinfer"
                else "PyTorch output-plus-LSE merge"
            ),
            "kv_layout": "contiguous NHD",
            "production_kernel": False,
        },
        "environment": _environment(torch),
        "workload": asdict(config),
        "correctness": {
            "atol": config.atol,
            "rtol": config.rtol,
            "route_query": route_correctness,
            "stage_kv": stage_correctness,
        },
        "route_query": route,
        "stage_kv": stage,
        "stage_over_route_p50": stage["p50_ms"] / route["p50_ms"],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _run_rank(
    rank: int,
    config: BenchmarkConfig,
    init_method: str,
    report_path: str,
) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=config.timeout_seconds),
    )
    try:
        query, prefix_key, prefix_value, tail_key, tail_value = _make_inputs(
            torch, config, device
        )
        prefix = (prefix_key, prefix_value)
        local_tail = (tail_key, tail_value)

        if rank == 0:
            remote_buffers = (
                torch.empty(
                    (config.rows, config.query_heads, config.head_dim),
                    dtype=query.dtype,
                    device=device,
                ),
                torch.empty(
                    (config.rows, config.query_heads),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            full_tokens = config.prefix_tokens + config.tail_tokens
            stage_key = torch.empty(
                (full_tokens, config.kv_heads, config.head_dim),
                dtype=prefix_key.dtype,
                device=device,
            )
            stage_value = torch.empty_like(stage_key)
            stage_receive_buffers = (
                stage_key[: config.prefix_tokens],
                stage_value[: config.prefix_tokens],
            )
            stage_full_buffers = (stage_key, stage_value)
            if config.tail_tokens:
                stage_key[config.prefix_tokens :].copy_(tail_key)
                stage_value[config.prefix_tokens :].copy_(tail_value)
        else:
            query_buffer = torch.empty_like(query)

        dist.barrier()
        if rank == 0:
            route_output = _route_engine(
                torch, dist, query, remote_buffers, local_tail, config
            )
            expected = _full_attention(torch, query, prefix, local_tail, config)
        else:
            _route_worker(torch, dist, query_buffer, prefix, config)

        dist.barrier()
        route_samples = []
        for iteration in range(config.warmup + config.iterations):
            if rank == 0:
                elapsed = _timed_cuda(
                    torch,
                    lambda: _route_engine(
                        torch, dist, query, remote_buffers, local_tail, config
                    ),
                )
                if iteration >= config.warmup:
                    route_samples.append(elapsed)
            else:
                _route_worker(torch, dist, query_buffer, prefix, config)

        dist.barrier()
        if rank == 0:
            stage_output = _stage_engine(
                torch,
                dist,
                query,
                stage_receive_buffers,
                stage_full_buffers,
                config,
            )
        else:
            _stage_worker(dist, prefix)

        dist.barrier()
        stage_samples = []
        for iteration in range(config.warmup + config.iterations):
            if rank == 0:
                elapsed = _timed_cuda(
                    torch,
                    lambda: _stage_engine(
                        torch,
                        dist,
                        query,
                        stage_receive_buffers,
                        stage_full_buffers,
                        config,
                    ),
                )
                if iteration >= config.warmup:
                    stage_samples.append(elapsed)
            else:
                _stage_worker(dist, prefix)

        dist.barrier()
        if rank == 0:
            _write_report(
                torch,
                config,
                route_output,
                stage_output,
                expected,
                route_samples,
                stage_samples,
                report_path,
            )
    finally:
        dist.destroy_process_group()


def _require_cuda_environment() -> Any:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as error:
        raise RuntimeError("PyTorch is required; install './python[cuda]'") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA devices are required")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("a PyTorch build with torch.distributed NCCL is required")
    return torch


def run_benchmark(config: BenchmarkConfig, report_path: Path) -> dict[str, Any]:
    """Execute the two-rank benchmark and return its persisted report."""
    config.validate()
    torch = _require_cuda_environment()
    import torch.multiprocessing as multiprocessing

    with TemporaryDirectory(prefix="loom-two-gpu-") as directory:
        rendezvous = Path(directory) / "nccl-rendezvous"
        init_method = f"file://{rendezvous}"
        multiprocessing.spawn(
            _run_rank,
            args=(config, init_method, str(report_path)),
            nprocs=2,
            join=True,
        )
    return json.loads(report_path.read_text())
