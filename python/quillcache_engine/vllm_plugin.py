"""Out-of-tree vLLM attention backend that delegates to FlashAttention.

The plugin registers `AttentionBackendEnum.CUSTOM`. M1 keeps all tensor math in
vLLM's native FlashAttention implementation while QuillCache validates the
local contract and records forward telemetry. Remote execution is introduced by
later backends, not by changing this delegate's behavior.
"""

from __future__ import annotations

import os
from typing import Any

from .local_delegate import LocalForwardObserver

_REGISTERED = False


def register() -> None:
    """Register the QuillCache backend with vLLM's documented OOT registry."""

    global _REGISTERED
    global QuillCacheFlashAttentionBackend
    global QuillCacheFlashAttentionImpl

    if _REGISTERED:
        return
    delegate = os.environ.get("QUILLCACHE_VLLM_DELEGATE", "flash_attn")
    if delegate != "flash_attn":
        raise RuntimeError(
            "M1 supports only QUILLCACHE_VLLM_DELEGATE=flash_attn; "
            f"got {delegate!r}"
        )

    try:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
            FlashAttentionImpl,
        )
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError as error:
        raise RuntimeError(
            "QuillCache's vLLM plugin requires vLLM 0.20.x with the V1 "
            "attention backend registry"
        ) from error

    class _QuillCacheFlashAttentionImpl(FlashAttentionImpl):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._quillcache_observer = _observer_from_init(args, kwargs)

        @property
        def quillcache_observer(self) -> LocalForwardObserver:
            return self._quillcache_observer

        def forward(
            self,
            layer: Any,
            query: Any,
            key: Any,
            value: Any,
            kv_cache: Any,
            attn_metadata: Any,
            output: Any,
            output_scale: Any = None,
            output_block_scale: Any = None,
        ) -> Any:
            token = self._quillcache_observer.before_forward(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                output=output,
            )
            try:
                result = super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                )
            except BaseException:
                self._quillcache_observer.after_forward(token, failed=True)
                raise
            self._quillcache_observer.after_forward(token)
            return result

    class _QuillCacheFlashAttentionBackend(FlashAttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "QUILLCACHE_FLASH_ATTN"

        @staticmethod
        def get_impl_cls() -> type[_QuillCacheFlashAttentionImpl]:
            return _QuillCacheFlashAttentionImpl

    # vLLM resolves registered classes by module-qualified name. Publish the
    # dynamic subclasses under stable names before updating the registry.
    _QuillCacheFlashAttentionImpl.__name__ = "QuillCacheFlashAttentionImpl"
    _QuillCacheFlashAttentionImpl.__qualname__ = "QuillCacheFlashAttentionImpl"
    _QuillCacheFlashAttentionImpl.__module__ = __name__
    QuillCacheFlashAttentionImpl = _QuillCacheFlashAttentionImpl

    _QuillCacheFlashAttentionBackend.__name__ = "QuillCacheFlashAttentionBackend"
    _QuillCacheFlashAttentionBackend.__qualname__ = "QuillCacheFlashAttentionBackend"
    _QuillCacheFlashAttentionBackend.__module__ = __name__
    QuillCacheFlashAttentionBackend = _QuillCacheFlashAttentionBackend

    register_backend(
        AttentionBackendEnum.CUSTOM,
        class_path=f"{__name__}.QuillCacheFlashAttentionBackend",
    )
    _REGISTERED = True


def _observer_from_init(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> LocalForwardObserver:
    def argument(name: str, index: int, default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        if index < len(args):
            return args[index]
        return default

    num_heads = argument("num_heads", 0)
    head_size = argument("head_size", 1)
    if num_heads is None or head_size is None:
        raise RuntimeError(
            "vLLM did not provide num_heads/head_size to the attention implementation"
        )
    return LocalForwardObserver(
        num_heads=int(num_heads),
        head_size=int(head_size),
        num_kv_heads=argument("num_kv_heads", 3),
        kv_cache_dtype=str(argument("kv_cache_dtype", 6, "auto")),
        attention_type=str(argument("attn_type", 8, "decoder")),
        validate_every_call=_boolean_environment(
            "QUILLCACHE_VALIDATE_EVERY_FORWARD", default=False
        ),
    )


def _boolean_environment(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value, got {value!r}")
