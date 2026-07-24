"""Extensions to vLLM's RoPE plus KV-cache compiler pattern."""

from __future__ import annotations

from typing import Any

import torch


_PER_HEAD_PATTERN_INSTALLED = False


def install_per_head_rope_kv_pattern() -> None:
    """Register the per-head static-Q form alongside vLLM's tensor form.

    vLLM 0.24/0.25 models per-head FP8 query quantization as one static
    quantization group per KV head. Its built-in RoPE+KV pattern only provides
    a scalar scale example, so that graph does not match. Loom retains the
    built-in registration and adds the missing vector-scale pattern.
    """

    global _PER_HEAD_PATTERN_INSTALLED
    if _PER_HEAD_PATTERN_INSTALLED:
        return

    from vllm.compilation.passes.fusion import rope_kvcache_fusion as fusion

    original_pattern = fusion.RopeStaticQQuantKVCachePattern
    if getattr(original_pattern, "_loom_registers_per_head", False):
        _PER_HEAD_PATTERN_INSTALLED = True
        return

    class PerHeadRopeStaticQQuantKVCachePattern(original_pattern):
        def get_inputs(self) -> list[Any]:
            token_count = 5
            positions = fusion.empty_i64(token_count)
            qkv = fusion.empty_bf16(
                token_count,
                self.q_size + self.k_size + self.v_size,
            )
            cos_sin_cache = fusion.empty_bf16(4096, self.head_size)
            q_scale = torch.empty(
                self.num_kv_heads,
                dtype=torch.float32,
                device=qkv.device,
            )
            inputs: list[Any] = [qkv, positions, cos_sin_cache, q_scale]
            if fusion._USE_LAYERNAME:
                inputs.append(fusion._encode_layer_name(self.layer_name))
            return inputs

        @property
        def q_group_shape(self) -> tuple[int, int]:
            return (-1, self.q_size // self.num_kv_heads)

        def _pattern(
            self,
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            q_scale: torch.Tensor,
            layer_name: Any,
        ):
            q, k, v = qkv.split(
                [self.q_size, self.k_size, self.v_size],
                dim=-1,
            )
            q, k = self.rope_matcher(positions, q, k, cos_sin_cache)
            q_fp8 = torch.empty(
                q.shape,
                device=q.device,
                dtype=fusion.FP8_DTYPE,
            )
            _, q_fp8 = fusion.auto_functionalized(
                torch.ops._C.static_scaled_fp8_quant.default,
                result=q_fp8,
                input=q,
                scale=q_scale,
                group_shape=self.q_group_shape,
            )
            q_view = q_fp8.view(-1, self.num_heads, self.head_size)
            k_view = k.view(-1, self.num_kv_heads, self.head_size)
            v_view = v.view(-1, self.num_kv_heads, self.head_size_v)
            cache_dependency = torch.ops.vllm.unified_kv_cache_update(
                k_view,
                v_view,
                layer_name,
            )
            return cache_dependency, q_view, k_view, v_view

        def _replacement(
            self,
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            q_scale: torch.Tensor,
            layer_name: Any,
        ):
            q, k, v = qkv.split(
                [self.q_size, self.k_size, self.v_size],
                dim=-1,
            )
            q_view = q.view(-1, self.num_heads, self.head_size)
            k_view = k.view(-1, self.num_kv_heads, self.head_size)
            v_view = v.view(-1, self.num_kv_heads, self.head_size_v)
            rope_kv_results = fusion.auto_functionalized(
                self.FUSED_ROPE_KV_OP,
                query=q_view,
                key=k_view,
                value=v_view,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=layer_name,
            )
            q_after = rope_kv_results[1].view(-1, self.q_size)
            q_fp8 = torch.empty(
                q_after.shape,
                device=q_after.device,
                dtype=fusion.FP8_DTYPE,
            )
            _, q_fp8 = fusion.auto_functionalized(
                torch.ops._C.static_scaled_fp8_quant.default,
                result=q_fp8,
                input=q_after,
                scale=q_scale,
                group_shape=self.q_group_shape,
            )
            return (
                rope_kv_results[0],
                q_fp8.view(-1, self.num_heads, self.head_size),
                rope_kv_results[2],
                v_view,
            )

        def _mk_pattern_with_layer_name_input(self, _layer_name: Any):
            def pattern(qkv, positions, cos_sin_cache, q_scale, layer_name):
                return self._pattern(
                    qkv,
                    positions,
                    cos_sin_cache,
                    q_scale,
                    layer_name,
                )

            def replacement(qkv, positions, cos_sin_cache, q_scale, layer_name):
                return self._replacement(
                    qkv,
                    positions,
                    cos_sin_cache,
                    q_scale,
                    layer_name,
                )

            return pattern, replacement

        def _mk_pattern_with_layer_name_closure(self, layer_name: Any):
            def pattern(qkv, positions, cos_sin_cache, q_scale):
                return self._pattern(
                    qkv,
                    positions,
                    cos_sin_cache,
                    q_scale,
                    layer_name,
                )

            def replacement(qkv, positions, cos_sin_cache, q_scale):
                return self._replacement(
                    qkv,
                    positions,
                    cos_sin_cache,
                    q_scale,
                    layer_name,
                )

            return pattern, replacement

    class TensorAndPerHeadRopeStaticQQuantKVCachePattern(original_pattern):
        _loom_registers_per_head = True

        def __init__(self, layer: Any, is_neox: bool) -> None:
            self._loom_layer = layer
            super().__init__(layer, is_neox)

        def register(self, matcher_pass: Any) -> None:
            original_pattern.register(self, matcher_pass)
            per_head = PerHeadRopeStaticQQuantKVCachePattern(
                self._loom_layer,
                self.is_neox,
            )
            original_pattern.register(per_head, matcher_pass)

    fusion.RopeStaticQQuantKVCachePattern = (
        TensorAndPerHeadRopeStaticQQuantKVCachePattern
    )
    _PER_HEAD_PATTERN_INSTALLED = True


def per_head_rope_kv_pattern_installed() -> bool:
    return _PER_HEAD_PATTERN_INSTALLED
