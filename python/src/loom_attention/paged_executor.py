"""FlashInfer executor for generation-pinned paged KV views."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .attention_state import AttentionState


DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024
SUPPORTED_LAYOUTS = ("HND", "NHD")

__all__ = [
    "DEFAULT_WORKSPACE_BYTES",
    "FlashInferPagedExecutor",
    "PagedKvContractError",
    "PagedKvView",
    "SUPPORTED_LAYOUTS",
]


class PagedKvContractError(ValueError):
    """Raised when a paged KV view cannot be consumed safely."""


@dataclass(frozen=True)
class PagedKvView:
    """Device page table covered by leases and pinned to one generation.

    The owner must increment ``page_table_generation`` before reusing this object
    after changing any page-table tensor in place. Lease expiry is enforced by
    the runtime that resolves this view; the executor only requires the proof
    identifiers to be present in its plan identity.
    """

    table_id: str
    page_table_generation: int
    lease_ids: tuple[str, ...]
    indptr: Any
    indices: Any
    last_page_len: Any
    page_size: int
    layout: str = "NHD"

    def validate(self, query: Any, paged_kv_cache: Any, *, kv_heads: int) -> Any:
        """Validate metadata without reading device tensor values on the host."""
        if not self.table_id:
            raise PagedKvContractError("table_id must be non-empty")
        if self.page_table_generation <= 0:
            raise PagedKvContractError(
                "page_table_generation must be positive for historical KV"
            )
        if not self.lease_ids or any(not lease_id for lease_id in self.lease_ids):
            raise PagedKvContractError(
                "historical KV requires non-empty lease identifiers"
            )
        if self.page_size <= 0:
            raise PagedKvContractError("page_size must be positive")
        if self.layout not in SUPPORTED_LAYOUTS:
            raise PagedKvContractError(f"unsupported KV layout: {self.layout}")
        if kv_heads <= 0:
            raise PagedKvContractError("kv_heads must be positive")

        query_shape = _shape(query, "query")
        if len(query_shape) != 3:
            raise PagedKvContractError(
                "query must have shape [batch, query_heads, head_dim]"
            )
        batch_size, query_heads, head_dim = query_shape
        if batch_size <= 0 or query_heads <= 0 or head_dim <= 0:
            raise PagedKvContractError("query dimensions must be positive")
        if query_heads % kv_heads != 0:
            raise PagedKvContractError("kv_heads must divide query heads")

        query_device = _device(query, "query")
        if not query_device.startswith("cuda"):
            raise PagedKvContractError("paged attention requires CUDA tensors")
        query_dtype = _dtype(query, "query")
        if not _is_attention_dtype(query_dtype):
            raise PagedKvContractError(
                f"unsupported query dtype for paged attention: {query_dtype}"
            )
        _require_contiguous(query, "query")

        auxiliary = (
            ("indptr", self.indptr, batch_size + 1),
            ("indices", self.indices, None),
            ("last_page_len", self.last_page_len, batch_size),
        )
        for name, tensor, expected_length in auxiliary:
            shape = _shape(tensor, name)
            if len(shape) != 1:
                raise PagedKvContractError(f"{name} must be one-dimensional")
            if expected_length is not None and shape[0] != expected_length:
                raise PagedKvContractError(
                    f"{name} length must be {expected_length}, got {shape[0]}"
                )
            if name == "indices" and shape[0] == 0:
                raise PagedKvContractError("indices must reference at least one page")
            if not _is_int32(_dtype(tensor, name)):
                raise PagedKvContractError(f"{name} must use torch.int32")
            if _device(tensor, name) != query_device:
                raise PagedKvContractError(
                    f"{name} must be on the query device {query_device}"
                )
            _require_contiguous(tensor, name)

        cache_tensors = _cache_tensors(paged_kv_cache)
        cache_shape = _validate_cache_shapes(
            cache_tensors,
            layout=self.layout,
            page_size=self.page_size,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        cache_dtype = _dtype(cache_tensors[0], "paged_kv_cache")
        if not _is_attention_dtype(cache_dtype):
            raise PagedKvContractError(
                f"unsupported KV dtype for paged attention: {cache_dtype}"
            )
        for index, tensor in enumerate(cache_tensors):
            name = f"paged_kv_cache[{index}]"
            if _device(tensor, name) != query_device:
                raise PagedKvContractError(
                    f"{name} must be on the query device {query_device}"
                )
            if _dtype(tensor, name) != cache_dtype:
                raise PagedKvContractError("key and value cache dtypes must match")
            _require_contiguous(tensor, name)
        return cache_dtype, cache_shape


@dataclass(frozen=True)
class _PlanKey:
    table_id: str
    page_table_generation: int
    lease_ids: tuple[str, ...]
    batch_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    layout: str
    query_dtype: str
    kv_dtype: str
    device: str
    scale: float


class FlashInferPagedExecutor:
    """Own and reuse one FlashInfer paged-decode wrapper and workspace."""

    def __init__(
        self,
        *,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        backend: str = "auto",
        use_tensor_cores: bool = False,
        torch_module: Any | None = None,
        flashinfer_module: Any | None = None,
    ) -> None:
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive")
        self._workspace_bytes = workspace_bytes
        self._backend = backend
        self._use_tensor_cores = use_tensor_cores
        self._torch = torch_module
        self._flashinfer = flashinfer_module
        self._workspace: Any | None = None
        self._wrapper: Any | None = None
        self._wrapper_device: str | None = None
        self._wrapper_layout: str | None = None
        self._plan_key: _PlanKey | None = None
        self._plan_count = 0

    @property
    def plan_count(self) -> int:
        """Number of FlashInfer plans built by this executor."""
        return self._plan_count

    def invalidate(self) -> None:
        """Force the next execute call to rebuild its plan."""
        self._plan_key = None

    def execute(
        self,
        query: Any,
        paged_kv_cache: Any,
        view: PagedKvView,
        *,
        kv_heads: int,
        scale: float,
    ) -> AttentionState:
        """Run paged decode and return the mergeable output-plus-LSE state."""
        if not math.isfinite(scale) or scale <= 0.0:
            raise PagedKvContractError("attention scale must be finite and positive")
        kv_dtype, _ = view.validate(query, paged_kv_cache, kv_heads=kv_heads)
        torch = self._torch if self._torch is not None else _load_torch()
        flashinfer = (
            self._flashinfer
            if self._flashinfer is not None
            else _load_flashinfer()
        )

        query_shape = tuple(query.shape)
        device = str(query.device)
        key = _PlanKey(
            table_id=view.table_id,
            page_table_generation=view.page_table_generation,
            lease_ids=view.lease_ids,
            batch_size=query_shape[0],
            query_heads=query_shape[1],
            kv_heads=kv_heads,
            head_dim=query_shape[2],
            page_size=view.page_size,
            layout=view.layout,
            query_dtype=str(query.dtype),
            kv_dtype=str(kv_dtype),
            device=device,
            scale=float(scale),
        )
        self._ensure_wrapper(torch, flashinfer, device=device, layout=view.layout)
        if self._plan_key != key:
            self._wrapper.plan(
                view.indptr,
                view.indices,
                view.last_page_len,
                query_shape[1],
                kv_heads,
                query_shape[2],
                view.page_size,
                pos_encoding_mode="NONE",
                q_data_type=query.dtype,
                kv_data_type=kv_dtype,
                sm_scale=scale,
            )
            self._plan_key = key
            self._plan_count += 1

        output, logsumexp = self._wrapper.run(
            query, paged_kv_cache, return_lse=True
        )
        return output.contiguous(), logsumexp.float().contiguous()

    def _ensure_wrapper(
        self, torch: Any, flashinfer: Any, *, device: str, layout: str
    ) -> None:
        if (
            self._wrapper is not None
            and self._wrapper_device == device
            and self._wrapper_layout == layout
        ):
            return
        self._workspace = torch.zeros(
            self._workspace_bytes,
            dtype=torch.uint8,
            device=device,
        )
        decode = getattr(flashinfer, "decode", flashinfer)
        wrapper_type = getattr(decode, "BatchDecodeWithPagedKVCacheWrapper")
        self._wrapper = wrapper_type(
            self._workspace,
            layout,
            use_cuda_graph=False,
            use_tensor_cores=self._use_tensor_cores,
            backend=self._backend,
        )
        self._wrapper_device = device
        self._wrapper_layout = layout
        self._plan_key = None


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("paged attention requires PyTorch") from error
    return torch


def _load_flashinfer() -> Any:
    try:
        import flashinfer
    except ImportError as error:
        raise RuntimeError(
            "paged attention requires './python[flashinfer]'"
        ) from error
    return flashinfer


def _shape(tensor: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(tensor.shape)
    except (AttributeError, TypeError) as error:
        raise PagedKvContractError(f"{name} must expose a tensor shape") from error


def _device(tensor: Any, name: str) -> str:
    try:
        return str(tensor.device)
    except AttributeError as error:
        raise PagedKvContractError(f"{name} must expose a device") from error


def _dtype(tensor: Any, name: str) -> Any:
    try:
        return tensor.dtype
    except AttributeError as error:
        raise PagedKvContractError(f"{name} must expose a dtype") from error


def _require_contiguous(tensor: Any, name: str) -> None:
    try:
        contiguous = tensor.is_contiguous()
    except AttributeError as error:
        raise PagedKvContractError(
            f"{name} must expose is_contiguous()"
        ) from error
    if not contiguous:
        raise PagedKvContractError(f"{name} must be contiguous")


def _is_int32(dtype: Any) -> bool:
    return str(dtype) in {"int32", "torch.int32"}


def _is_attention_dtype(dtype: Any) -> bool:
    return str(dtype) in {
        "bfloat16",
        "float16",
        "torch.bfloat16",
        "torch.float16",
    }


def _cache_tensors(paged_kv_cache: Any) -> tuple[Any, ...]:
    if isinstance(paged_kv_cache, tuple):
        if len(paged_kv_cache) != 2:
            raise PagedKvContractError(
                "separate paged KV cache must be a (key, value) tuple"
            )
        return paged_kv_cache
    return (paged_kv_cache,)


def _validate_cache_shapes(
    cache_tensors: tuple[Any, ...],
    *,
    layout: str,
    page_size: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, ...]:
    shapes = tuple(_shape(tensor, "paged_kv_cache") for tensor in cache_tensors)
    if len(cache_tensors) == 1:
        expected = (
            (None, 2, page_size, kv_heads, head_dim)
            if layout == "NHD"
            else (None, 2, kv_heads, page_size, head_dim)
        )
    else:
        expected = (
            (None, page_size, kv_heads, head_dim)
            if layout == "NHD"
            else (None, kv_heads, page_size, head_dim)
        )
        if shapes[0] != shapes[1]:
            raise PagedKvContractError("key and value cache shapes must match")
    shape = shapes[0]
    if len(shape) != len(expected) or shape[0] <= 0:
        raise PagedKvContractError(
            f"paged KV cache has invalid {layout} shape: {shape}"
        )
    if any(actual != wanted for actual, wanted in zip(shape[1:], expected[1:])):
        raise PagedKvContractError(
            f"paged KV cache has invalid {layout} shape: {shape}"
        )
    return shape
