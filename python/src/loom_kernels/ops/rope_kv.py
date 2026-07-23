"""RoPE and paged-KV predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _rope_paged_kv_write
from ._common import _DTYPE_NAMES, _require_inference_tensors


def supports_rope_paged_kv_write(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    """Return whether tensors match the native/FP8 RoPE+paged-write ABI."""
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        return False
    if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
        return False
    if cos_sin_cache.dim() != 2:
        return False
    rotary_dim = cos_sin_cache.shape[1]
    return bool(
        query.device.type == "cuda"
        and key.device == query.device
        and value.device == query.device
        and positions.device == query.device
        and cos_sin_cache.device == query.device
        and kv_cache.device == query.device
        and key_scales.device == query.device
        and value_scales.device == query.device
        and slot_mapping.device == query.device
        and query.dtype in _DTYPE_NAMES
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and cos_sin_cache.dtype == query.dtype
        and kv_cache.dtype in (query.dtype, torch.uint8)
        and key_scales.dtype == torch.float32
        and value_scales.dtype == torch.float32
        and key_scales.numel() == value_scales.numel()
        and key_scales.numel() in (1, key.shape[1])
        and key_scales.is_contiguous()
        and value_scales.is_contiguous()
        and positions.dtype == torch.int64
        and slot_mapping.dtype == torch.int64
        and query.shape[0] == key.shape[0] == value.shape[0]
        and query.shape[2] == key.shape[2]
        and key.shape[1] == value.shape[1]
        and key.shape[2] == value.shape[2]
        and positions.dim() == 1
        and positions.numel() == query.shape[0]
        and slot_mapping.dim() == 1
        and slot_mapping.numel() <= query.shape[0]
        and cos_sin_cache.shape[0] > 0
        and rotary_dim > 0
        and rotary_dim % 2 == 0
        and rotary_dim <= query.shape[2]
        and kv_cache.dim() == 5
        and kv_cache.shape[0] > 0
        and kv_cache.shape[1] == 2
        and kv_cache.shape[2] > 0
        and kv_cache.shape[3:] == key.shape[1:]
        and query.stride(2) == 1
        and key.stride(2) == 1
        and value.stride(2) == 1
        and all(stride > 0 for stride in query.stride()[:2])
        and all(stride > 0 for stride in key.stride()[:2])
        and all(stride > 0 for stride in value.stride()[:2])
        and positions.is_contiguous()
        and cos_sin_cache.is_contiguous()
        and slot_mapping.is_contiguous()
        and kv_cache.stride(4) == 1
        and all(stride > 0 for stride in kv_cache.stride()[:4])
        and kv_cache.stride(1)
        >= (
            (kv_cache.shape[2] - 1) * kv_cache.stride(2)
            + (kv_cache.shape[3] - 1) * kv_cache.stride(3)
            + kv_cache.shape[4]
        )
        and kv_cache.stride(0)
        >= (
            kv_cache.stride(1)
            + (kv_cache.shape[2] - 1) * kv_cache.stride(2)
            + (kv_cache.shape[3] - 1) * kv_cache.stride(3)
            + kv_cache.shape[4]
        )
        and not query.requires_grad
        and not key.requires_grad
        and not value.requires_grad
        and not cos_sin_cache.requires_grad
    )


def rope_paged_kv_write_(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_neox: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate Q/K and write native or scaled FP8 E4M3 paged K/V caches."""
    _require_inference_tensors(
        query,
        key,
        value,
        cos_sin_cache,
        kv_cache,
        key_scales,
        value_scales,
    )
    _rope_paged_kv_write(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        kv_cache,
        key_scales,
        value_scales,
        slot_mapping,
        bool(is_neox),
    )
    return query, key, kv_cache
