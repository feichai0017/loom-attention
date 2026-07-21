"""vLLM IR provider registration for Loom Kernels."""

from __future__ import annotations

import os
from typing import Any

import torch

from ._native import native_available


DEFAULT_PROVIDER = "loom_cuda"
SILU_OVERRIDE_KEY = "SiluAndMul"
SILU_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_SILU_AND_MUL"
ACT_QUANT_OVERRIDE_KEY = "silu_and_mul_dynamic_fp8"
ACT_QUANT_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8"
ROPE_PAGED_KV_OVERRIDE_KEY = "rope_paged_kv"
_SILU_OVERRIDE_CLASS: type | None = None
_ACT_QUANT_OVERRIDE_REGISTERED = False
_ROPE_PAGED_KV_REGISTERED = False
_ROPE_PAGED_KV_FIRST_CONTRACT: dict[str, Any] | None = None


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _silu_override_requested() -> bool:
    return _env_enabled(SILU_OVERRIDE_ENV)


def _act_quant_override_requested() -> bool:
    return _env_enabled(ACT_QUANT_OVERRIDE_ENV)


def register_vllm_silu_and_mul() -> str | None:
    """Override vLLM's standard SwiGLU layer with the Loom CUDA operator."""
    global _SILU_OVERRIDE_CLASS
    if _SILU_OVERRIDE_CLASS is not None:
        return SILU_OVERRIDE_KEY
    if not native_available():
        return None

    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.activation import SiluAndMul

    from .torch_ops import _silu_and_mul_unchecked

    @CustomOp.register_oot(name=SILU_OVERRIDE_KEY)
    class LoomSiluAndMul(SiluAndMul):
        def __init__(self, *, compile_native: bool = True):
            # vLLM may globally disable CustomOp kernels while compiling its
            # native fallback.  An out-of-tree replacement must opt back in,
            # otherwise the registered class exists but never reaches Loom.
            del compile_native
            CustomOp.__init__(self, enforce_enable=True, compile_native=False)

        def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
            width = x.shape[-1] // 2
            output = torch.empty(
                (*x.shape[:-1], width), dtype=x.dtype, device=x.device
            )
            _silu_and_mul_unchecked(x, output)
            return output

    _SILU_OVERRIDE_CLASS = LoomSiluAndMul
    return SILU_OVERRIDE_KEY


def register_vllm_silu_and_mul_dynamic_fp8() -> str | None:
    """Route vLLM's 64/128-element activation-quant fusions to Loom."""
    global _ACT_QUANT_OVERRIDE_REGISTERED
    if _ACT_QUANT_OVERRIDE_REGISTERED:
        return ACT_QUANT_OVERRIDE_KEY
    if not native_available():
        return None

    from .torch_ops import adapter_backend

    if adapter_backend() != "cpp-dispatch":
        return None

    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    FUSED_OPS[kFp8Dynamic64Sym] = implementation
    FUSED_OPS[kFp8Dynamic128Sym] = implementation
    _ACT_QUANT_OVERRIDE_REGISTERED = True
    return ACT_QUANT_OVERRIDE_KEY


def register_vllm_rope_paged_kv() -> str | None:
    """Teach vLLM 0.24 CUDA attention backends to call Loom's fused op.

    Registration only installs the backend capability and implementation. Use
    :func:`configure_vllm_rope_paged_kv` before constructing ``vllm.LLM`` to
    opt the compilation graph into vLLM's existing RoPE+KV fusion pass.
    """
    global _ROPE_PAGED_KV_REGISTERED
    if _ROPE_PAGED_KV_REGISTERED:
        return ROPE_PAGED_KV_OVERRIDE_KEY
    if not native_available():
        return None

    from .torch_ops import adapter_backend

    if adapter_backend() != "cpp-dispatch":
        return None

    from importlib.metadata import version

    if not version("vllm").startswith("0.24."):
        return None

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    implementation = torch.ops.loom_kernels.rope_paged_kv_write_unchecked_.default
    native_cache_dtypes = {
        "auto",
        "float16",
        "half",
        "bfloat16",
        "float32",
        "float",
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }

    def supported(attention: Any) -> bool:
        return bool(
            getattr(attention, "kv_sharing_target_layer_name", None) is None
            and getattr(attention, "kv_cache_dtype", None) in native_cache_dtypes
            and getattr(attention, "attn_type", AttentionType.DECODER)
            == AttentionType.DECODER
        )

    def do_rope_and_kv_cache_update(
        attention: Any,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ) -> None:
        global _ROPE_PAGED_KV_FIRST_CONTRACT
        del attention, layer
        key_cache, value_cache = kv_cache.unbind(1)
        if _ROPE_PAGED_KV_FIRST_CONTRACT is None:
            _ROPE_PAGED_KV_FIRST_CONTRACT = {
                "query": {
                    "shape": list(query.shape),
                    "stride": list(query.stride()),
                    "dtype": str(query.dtype),
                },
                "key": {
                    "shape": list(key.shape),
                    "stride": list(key.stride()),
                    "dtype": str(key.dtype),
                },
                "value": {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                },
                "positions": list(positions.shape),
                "cos_sin_cache": {
                    "shape": list(cos_sin_cache.shape),
                    "dtype": str(cos_sin_cache.dtype),
                },
                "kv_cache": {
                    "shape": list(kv_cache.shape),
                    "stride": list(kv_cache.stride()),
                    "dtype": str(kv_cache.dtype),
                },
                "slot_mapping": list(layer_slot_mapping.shape),
                "is_neox": is_neox,
            }
        implementation(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            layer_slot_mapping,
            is_neox,
        )

    for implementation_class in (FlashAttentionImpl, FlashInferImpl):
        implementation_class.fused_rope_kvcache_supported = supported
        implementation_class.do_rope_and_kv_cache_update = (
            do_rope_and_kv_cache_update
        )

    _ROPE_PAGED_KV_REGISTERED = True
    return ROPE_PAGED_KV_OVERRIDE_KEY


def configure_vllm_rope_paged_kv(
    compilation_config: Any | None = None,
    *,
    max_token_num: int = 256,
) -> Any:
    """Return a vLLM compilation config with Loom RoPE+KV fusion enabled.

    vLLM 0.24 labels this pass ROCm-only during initial ``PassConfig``
    validation. Setting the flag after constructing ``CompilationConfig`` is
    intentional: Loom supplies the missing CUDA backend implementation.
    """
    if max_token_num <= 0:
        raise ValueError("max_token_num must be positive")
    if register_vllm_rope_paged_kv() is None:
        raise RuntimeError(
            "Loom RoPE+paged-KV requires vLLM 0.24 and the C++ dispatcher bridge"
        )

    from vllm.config import CompilationConfig

    if compilation_config is None:
        configured = CompilationConfig()
    elif isinstance(compilation_config, dict):
        configured = CompilationConfig(**compilation_config)
    elif isinstance(compilation_config, CompilationConfig):
        configured = compilation_config
    else:
        raise TypeError("compilation_config must be a dict or CompilationConfig")

    if "+rotary_embedding" not in configured.custom_ops:
        configured.custom_ops.append("+rotary_embedding")
    if configured.splitting_ops is None:
        configured.splitting_ops = []
    configured.pass_config.fuse_rope_kvcache = True
    configured.pass_config.rope_kvcache_fusion_max_token_num = max_token_num
    return configured


def register_vllm_ir(provider: str = DEFAULT_PROVIDER) -> str:
    """Register Loom as an in-place fused_add_rms_norm IR provider."""
    from vllm import ir
    import vllm.ir.ops.layernorm  # noqa: F401 - registers the IR operation

    from .torch_ops import (
        _add_rms_norm_mut_unchecked,
        adapter_backend,
        supports_vllm_add_rms_norm,
    )

    if _silu_override_requested():
        register_vllm_silu_and_mul()
    if _act_quant_override_requested():
        register_vllm_silu_and_mul_dynamic_fp8()

    operation = ir.ops.fused_add_rms_norm
    implementations = getattr(operation, "impls", {})
    if provider in implementations:
        return provider

    def implementation(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight is None or variance_size is not None:
            raise ValueError("unsupported Loom Add+RMSNorm contract reached dispatch")
        _add_rms_norm_mut_unchecked(x, x_residual, weight, epsilon)
        return x, x_residual

    def supports(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> bool:
        return supports_vllm_add_rms_norm(
            x, x_residual, weight, epsilon, variance_size
        )

    decorator = operation.register_impl(
        provider,
        supported=native_available(),
        supports_args=supports,
        inplace=True,
    )
    decorator(implementation)
    operation.impls[provider].adapter_backend = adapter_backend()
    return provider


def provider_metadata() -> dict[str, Any]:
    from .torch_ops import adapter_backend

    return {
        "provider": DEFAULT_PROVIDER,
        "native_available": native_available(),
        "operator": "fused_add_rms_norm",
        "inplace": True,
        "adapter_backend": adapter_backend(),
        "silu_and_mul_override_requested": _silu_override_requested(),
        "silu_and_mul_override": _SILU_OVERRIDE_CLASS is not None,
        "silu_and_mul_fp8_override_requested": _act_quant_override_requested(),
        "silu_and_mul_fp8_override": _ACT_QUANT_OVERRIDE_REGISTERED,
        "rope_paged_kv_override": _ROPE_PAGED_KV_REGISTERED,
        "rope_paged_kv_first_contract": _ROPE_PAGED_KV_FIRST_CONTRACT,
    }


__all__ = [
    "ACT_QUANT_OVERRIDE_ENV",
    "ACT_QUANT_OVERRIDE_KEY",
    "DEFAULT_PROVIDER",
    "ROPE_PAGED_KV_OVERRIDE_KEY",
    "SILU_OVERRIDE_ENV",
    "SILU_OVERRIDE_KEY",
    "provider_metadata",
    "configure_vllm_rope_paged_kv",
    "register_vllm_ir",
    "register_vllm_rope_paged_kv",
    "register_vllm_silu_and_mul",
    "register_vllm_silu_and_mul_dynamic_fp8",
]
