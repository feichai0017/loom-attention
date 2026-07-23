from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    rope_paged_kv_write_,
)


def make_cos_sin_cache(
    max_position: int, rotary_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    frequencies = torch.outer(positions, inverse_frequency)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(dtype)


def make_cache(
    num_blocks: int,
    block_size: int,
    kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_blocks, 2, block_size, kv_heads, head_size)
    if layout == "NHD":
        combined = torch.empty(shape, device="cuda", dtype=dtype)
    elif layout == "HND":
        block_stride = 2 * block_size * kv_heads * head_size
        kv_stride = block_size * kv_heads * head_size
        combined = torch.empty_strided(
            shape,
            (
                block_stride,
                kv_stride,
                head_size,
                block_size * head_size,
                1,
            ),
            device="cuda",
            dtype=dtype,
        )
    else:
        raise ValueError(layout)
    combined.fill_(0xA5 if dtype == torch.uint8 else -7.0)
    key_cache, value_cache = combined.unbind(1)
    return combined, key_cache, value_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_rope_paged_kv_matches_vllm(dtype, is_neox, layout):
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401 - registers vLLM dispatcher ops

    torch.manual_seed(73)
    tokens = 5
    query_heads = 4
    kv_heads = 2
    head_size = 16
    value_head_size = 16
    rotary_dim = 8
    max_position = 32
    num_blocks = 3
    block_size = 8

    query = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=dtype
    )
    key = torch.randn((tokens, kv_heads, head_size), device="cuda", dtype=dtype)
    value = torch.randn(
        (tokens, kv_heads, value_head_size), device="cuda", dtype=dtype
    )
    positions = torch.tensor([0, 3, 5, 7, 11], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 7, -1, 15, 22], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(max_position, rotary_dim, dtype)

    expected_query = query.clone()
    expected_key = key.clone()
    expected_combined, expected_key_cache, expected_value_cache = make_cache(
        num_blocks, block_size, kv_heads, head_size, dtype, layout
    )
    actual_query = query.clone()
    actual_key = key.clone()
    actual_combined, _, _ = make_cache(
        num_blocks, block_size, kv_heads, head_size, dtype, layout
    )

    torch.ops._C.rotary_embedding(
        positions,
        expected_query,
        expected_key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        expected_key,
        value,
        expected_key_cache,
        expected_value_cache,
        slots,
        "auto",
        scale,
        scale,
    )

    returned = rope_paged_kv_write_(
        actual_query,
        actual_key,
        value,
        positions,
        cos_sin_cache,
        actual_combined,
        scale,
        scale,
        slots,
        is_neox,
    )
    torch.cuda.synchronize()

    assert all(
        actual is expected
        for actual, expected in zip(
            returned,
            (actual_query, actual_key, actual_combined),
            strict=True,
        )
    )
    tolerance = {
        torch.float32: (1.0e-5, 1.0e-6),
        torch.float16: (1.0e-3, 1.0e-3),
        torch.bfloat16: (1.0e-2, 1.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual_query, expected_query, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(
        actual_key, expected_key, rtol=tolerance[0], atol=tolerance[1]
    )
    # Compare the combined physical allocations so untouched padding and both
    # logical NHD/HND stride paths are covered as well.
    torch.testing.assert_close(
        actual_combined, expected_combined, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale_mode", ["per_tensor", "per_head"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("is_neox", [True, False])
def test_rope_paged_kv_fp8_matches_vllm(dtype, scale_mode, layout, is_neox):
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401 - registers vLLM dispatcher ops

    torch.manual_seed(89)
    tokens, query_heads, kv_heads = 5, 4, 2
    head_size, rotary_dim = 16, 8
    num_blocks, block_size = 3, 8
    query = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=dtype
    )
    key = torch.randn((tokens, kv_heads, head_size), device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    positions = torch.tensor([0, 3, 5, 7, 11], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 7, -1, 15, 22], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(32, rotary_dim, dtype)
    if scale_mode == "per_tensor":
        key_scales = torch.tensor([0.25], device="cuda", dtype=torch.float32)
        value_scales = torch.tensor([0.375], device="cuda", dtype=torch.float32)
    else:
        key_scales = torch.tensor([0.25, 0.5], device="cuda", dtype=torch.float32)
        value_scales = torch.tensor(
            [0.375, 0.75], device="cuda", dtype=torch.float32
        )

    expected_query = query.clone()
    expected_key = key.clone()
    expected_combined, expected_key_cache, expected_value_cache = make_cache(
        num_blocks, block_size, kv_heads, head_size, torch.uint8, layout
    )
    actual_query = query.clone()
    actual_key = key.clone()
    actual_combined, _, _ = make_cache(
        num_blocks, block_size, kv_heads, head_size, torch.uint8, layout
    )

    torch.ops._C.rotary_embedding(
        positions,
        expected_query,
        expected_key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        expected_key,
        value,
        expected_key_cache,
        expected_value_cache,
        slots,
        "fp8",
        key_scales,
        value_scales,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rope_paged_kv_write_(
            actual_query,
            actual_key,
            value,
            positions,
            cos_sin_cache,
            actual_combined,
            key_scales,
            value_scales,
            slots,
            is_neox,
        )
    stream.synchronize()

    tolerance = (1.0e-3, 1.0e-3) if dtype == torch.float16 else (1.0e-2, 1.0e-2)
    torch.testing.assert_close(
        actual_query, expected_query, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(
        actual_key, expected_key, rtol=tolerance[0], atol=tolerance[1]
    )
    assert torch.equal(actual_combined, expected_combined)


def make_small_fp8_case():
    tokens, query_heads, kv_heads, head_size = 2, 2, 1, 8
    query = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(
        (tokens, kv_heads, head_size), device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    positions = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 5], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(8, head_size, torch.bfloat16)
    combined, _, _ = make_cache(
        1, 8, kv_heads, head_size, torch.uint8, "NHD"
    )
    key_scales = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    value_scales = torch.tensor([0.375], device="cuda", dtype=torch.float32)
    arguments = (
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        combined,
        key_scales,
        value_scales,
        slots,
        True,
    )
    return arguments, combined


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_fp8_schema_and_fake_tensor_contract():
    arguments, _ = make_small_fp8_case()
    torch.library.opcheck(
        torch.ops.loom_kernels.rope_paged_kv_write_.default,
        arguments,
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_fp8_launch_counter_proves_bridge_submission():
    arguments, _ = make_small_fp8_case()

    reset_launch_count(Operator.ROPE_PAGED_KV_WRITE)
    rope_paged_kv_write_(*arguments)
    torch.cuda.synchronize()
    assert launch_count(Operator.ROPE_PAGED_KV_WRITE) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_rejects_bad_scale_length_before_bridge_launch():
    arguments, _ = make_small_fp8_case()
    invalid_scales = torch.ones(2, device="cuda", dtype=torch.float32)
    arguments = (*arguments[:6], invalid_scales, invalid_scales, *arguments[8:])

    reset_launch_count(Operator.ROPE_PAGED_KV_WRITE)
    with pytest.raises(
        RuntimeError,
        match="one element or one element per KV head",
    ):
        rope_paged_kv_write_(*arguments)
    assert launch_count(Operator.ROPE_PAGED_KV_WRITE) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_rejects_overlapping_packed_cache_partitions():
    arguments, _ = make_small_fp8_case()
    storage = torch.empty(128, device="cuda", dtype=torch.uint8)
    overlapping_cache = torch.as_strided(
        storage,
        (1, 2, 8, 1, 8),
        (128, 1, 8, 8, 1),
    )
    arguments = (*arguments[:5], overlapping_cache, *arguments[6:])

    reset_launch_count(Operator.ROPE_PAGED_KV_WRITE)
    with pytest.raises(RuntimeError, match="partitions must not overlap"):
        rope_paged_kv_write_(*arguments)
    assert launch_count(Operator.ROPE_PAGED_KV_WRITE) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_fp8_survives_torch_compile():
    torch.manual_seed(101)
    expected_arguments, expected_combined = make_small_fp8_case()
    torch.manual_seed(101)
    actual_arguments, actual_combined = make_small_fp8_case()

    rope_paged_kv_write_(*expected_arguments)

    def compiled_target(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        kv_cache,
        key_scales,
        value_scales,
        slots,
        is_neox,
    ):
        torch.ops.loom_kernels.rope_paged_kv_write_.default(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            kv_cache,
            key_scales,
            value_scales,
            slots,
            is_neox,
        )
        return query, key, kv_cache

    compiled = torch.compile(compiled_target, fullgraph=True)
    actual_query, actual_key, _ = compiled(*actual_arguments)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_query, expected_arguments[0])
    torch.testing.assert_close(actual_key, expected_arguments[1])
    assert torch.equal(actual_combined, expected_combined)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_fp8_can_be_captured_and_replayed():
    torch.manual_seed(103)
    expected_arguments, expected_combined = make_small_fp8_case()
    torch.manual_seed(103)
    actual_arguments, actual_combined = make_small_fp8_case()
    original_query = actual_arguments[0].clone()
    original_key = actual_arguments[1].clone()

    rope_paged_kv_write_(*expected_arguments)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rope_paged_kv_write_(*actual_arguments)
    actual_arguments[0].copy_(original_query)
    actual_arguments[1].copy_(original_key)
    actual_combined.fill_(0xA5)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_arguments[0], expected_arguments[0])
    torch.testing.assert_close(actual_arguments[1], expected_arguments[1])
    assert torch.equal(actual_combined, expected_combined)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cache_dtype", ["native", "fp8"])
def test_rope_paged_kv_accepts_packed_qkv_and_short_slot_mapping(cache_dtype):
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401 - registers vLLM dispatcher ops

    torch.manual_seed(97)
    tokens, query_heads, kv_heads, head_size = 5, 4, 2, 16
    q_width = query_heads * head_size
    kv_width = kv_heads * head_size
    packed = torch.randn(
        (tokens, q_width + 2 * kv_width),
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int64)
    # vLLM leaves Q/K/V padded while the cache update only receives real slots.
    slots = torch.tensor([0, 7, 9], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(16, head_size, torch.bfloat16)

    expected_packed = packed.clone()
    expected_query, expected_key, expected_value = expected_packed.split(
        (q_width, kv_width, kv_width), dim=-1
    )
    expected_query = expected_query.view(tokens, query_heads, head_size)
    expected_key = expected_key.view(tokens, kv_heads, head_size)
    expected_value = expected_value.view(tokens, kv_heads, head_size)
    actual_packed = packed.clone()
    actual_query, actual_key, actual_value = actual_packed.split(
        (q_width, kv_width, kv_width), dim=-1
    )
    actual_query = actual_query.view(tokens, query_heads, head_size)
    actual_key = actual_key.view(tokens, kv_heads, head_size)
    actual_value = actual_value.view(tokens, kv_heads, head_size)
    assert not actual_query.is_contiguous()
    assert actual_query.stride(0) == actual_key.stride(0) == actual_value.stride(0)

    cache_torch_dtype = torch.bfloat16 if cache_dtype == "native" else torch.uint8
    vllm_cache_dtype = "auto" if cache_dtype == "native" else "fp8"
    expected_combined, expected_key_cache, expected_value_cache = make_cache(
        2, 8, kv_heads, head_size, cache_torch_dtype, "NHD"
    )
    actual_combined, _, _ = make_cache(
        2, 8, kv_heads, head_size, cache_torch_dtype, "NHD"
    )
    torch.ops._C.rotary_embedding(
        positions,
        expected_query,
        expected_key,
        head_size,
        cos_sin_cache,
        True,
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        expected_key,
        expected_value,
        expected_key_cache,
        expected_value_cache,
        slots,
        vllm_cache_dtype,
        scale,
        scale,
    )

    rope_paged_kv_write_(
        actual_query,
        actual_key,
        actual_value,
        positions,
        cos_sin_cache,
        actual_combined,
        scale,
        scale,
        slots,
        True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_packed, expected_packed, rtol=1.0e-2, atol=1.0e-2)
    if cache_dtype == "fp8":
        assert torch.equal(actual_combined, expected_combined)
    else:
        torch.testing.assert_close(
            actual_combined, expected_combined, rtol=1.0e-2, atol=1.0e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_uses_the_current_external_stream():
    tokens, query_heads, kv_heads, head_size = 2, 2, 1, 8
    query = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=torch.float16
    )
    key = torch.randn((tokens, kv_heads, head_size), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    positions = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 5], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(8, head_size, torch.float16)
    kv_cache, key_cache, value_cache = make_cache(
        1, 8, kv_heads, head_size, torch.float16, "NHD"
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rope_paged_kv_write_(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            kv_cache,
            scale,
            scale,
            slots,
            True,
        )
    stream.synchronize()

    assert torch.isfinite(query).all()
    assert torch.isfinite(key).all()
    torch.testing.assert_close(key_cache[0, 0], key[0])
    torch.testing.assert_close(key_cache[0, 5], key[1])
    torch.testing.assert_close(value_cache[0, 0], value[0])
    torch.testing.assert_close(value_cache[0, 5], value[1])
