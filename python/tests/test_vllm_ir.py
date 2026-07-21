from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from loom_kernels.vllm import (
    ACT_QUANT_OVERRIDE_ENV,
    ACT_QUANT_OVERRIDE_KEY,
    DEFAULT_PROVIDER,
    ROPE_PAGED_KV_OVERRIDE_KEY,
    SILU_OVERRIDE_ENV,
    SILU_OVERRIDE_KEY,
    configure_vllm_rope_paged_kv,
    provider_metadata,
    register_vllm_ir,
    register_vllm_rope_paged_kv,
    register_vllm_silu_and_mul,
    register_vllm_silu_and_mul_dynamic_fp8,
)


def test_configures_vllm_rope_paged_kv_fusion():
    from vllm.config import CompilationConfig
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    assert register_vllm_rope_paged_kv() == ROPE_PAGED_KV_OVERRIDE_KEY
    config = configure_vllm_rope_paged_kv(max_token_num=128)

    assert isinstance(config, CompilationConfig)
    assert config.pass_config.fuse_rope_kvcache is True
    assert config.pass_config.rope_kvcache_fusion_max_token_num == 128
    assert config.splitting_ops == []
    assert "+rotary_embedding" in config.custom_ops
    assert FlashAttentionImpl.fused_rope_kvcache_supported.__module__ == (
        "loom_kernels.vllm"
    )
    assert FlashInferImpl.fused_rope_kvcache_supported.__module__ == (
        "loom_kernels.vllm"
    )
    assert provider_metadata()["rope_paged_kv_override"] is True


def test_registers_inplace_fused_add_rms_norm_provider():
    from vllm import ir

    assert register_vllm_ir() == DEFAULT_PROVIDER
    assert DEFAULT_PROVIDER in ir.ops.fused_add_rms_norm.impls
    implementation = ir.ops.fused_add_rms_norm.impls[DEFAULT_PROVIDER]
    assert implementation.inplace is True


def test_registers_vllm_silu_and_mul_override():
    from vllm.model_executor.custom_op import op_registry_oot

    assert register_vllm_silu_and_mul() == SILU_OVERRIDE_KEY
    assert SILU_OVERRIDE_KEY in op_registry_oot


def test_silu_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(SILU_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_override_requested"] is False
    monkeypatch.setenv(SILU_OVERRIDE_ENV, "true")
    assert provider_metadata()["silu_and_mul_override_requested"] is True


def test_registers_vllm_silu_and_mul_dynamic_fp8_fusion():
    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    assert (
        register_vllm_silu_and_mul_dynamic_fp8() == ACT_QUANT_OVERRIDE_KEY
    )
    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    assert FUSED_OPS[kFp8Dynamic64Sym] == implementation
    assert FUSED_OPS[kFp8Dynamic128Sym] == implementation


def test_act_quant_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(ACT_QUANT_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_fp8_override_requested"] is False
    monkeypatch.setenv(ACT_QUANT_OVERRIDE_ENV, "on")
    assert provider_metadata()["silu_and_mul_fp8_override_requested"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_activation_quant_pattern_rewrites_to_loom():
    from vllm.compilation.passes.fusion.act_quant_fusion import (
        SiluMulBlockQuantPattern,
    )
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
        enable_fake_mode,
    )
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
    )

    config = VllmConfig()
    with set_current_vllm_config(config):
        register_vllm_silu_and_mul_dynamic_fp8()
        pattern = SiluMulBlockQuantPattern(kFp8Dynamic128Sym)
        fusion_pass = VllmFusionPatternMatcherPass(
            config, "loom_activation_quant_test"
        )
        fusion_pass.register(pattern)

        @enable_fake_mode
        def trace_official_pattern():
            return fusion_pass._trace_fn(pattern.pattern, pattern.get_inputs())

        graph_module = trace_official_pattern()
        fusion_pass(graph_module.graph)

    loom_operator = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    loom_target_present = any(
        node.op == "call_function"
        and node.args
        and node.args[0] == loom_operator
        for node in graph_module.graph.nodes
    )
    assert fusion_pass.matched_count == 1
    assert loom_target_present


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_silu_layer_dispatches_to_loom():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul

    register_vllm_silu_and_mul()
    with set_current_vllm_config(VllmConfig()):
        activation = SiluAndMul()
    assert type(activation).__name__ == "LoomSiluAndMul"

    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(expected, input_tensor)
    actual = activation(input_tensor)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_ir_dispatches_to_loom_provider():
    from vllm import ir
    from vllm.platforms import current_platform

    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    input_tensor = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    expected_residual = (input_tensor.float() + residual.float()).to(torch.bfloat16)

    with operation.set_priority([DEFAULT_PROVIDER, "native"]):
        assert (
            operation.dispatch(input_tensor, residual, weight, 1.0e-5).provider
            == DEFAULT_PROVIDER
        )
        output, residual_output = operation.maybe_inplace(
            input_tensor, residual, weight, 1.0e-5
        )
    torch.cuda.synchronize()

    assert output is input_tensor
    assert residual_output is residual
    torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 4096), (8, 4096), (128, 4096), (8, 8192)])
def test_loom_is_bitwise_equal_to_vllm_cuda_provider(shape):
    from vllm import ir
    from vllm.platforms import current_platform

    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    if "vllm_c" not in operation.impls or not operation.impls["vllm_c"].supported:
        pytest.skip("vLLM CUDA provider is unavailable")

    torch.manual_seed(20260721)
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    outputs = {}
    for provider in (DEFAULT_PROVIDER, "vllm_c"):
        provider_input = input_tensor.clone()
        provider_residual = residual.clone()
        with operation.set_priority([provider, "native"]):
            outputs[provider] = operation.maybe_inplace(
                provider_input, provider_residual, weight, 1.0e-5
            )
    torch.cuda.synchronize()

    loom_output, loom_residual = outputs[DEFAULT_PROVIDER]
    vllm_output, vllm_residual = outputs["vllm_c"]
    assert torch.equal(loom_output, vllm_output)
    assert torch.equal(loom_residual, vllm_residual)
