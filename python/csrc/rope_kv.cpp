#include "common.h"

namespace loom_kernels::torch_adapter {

uint64_t kv_cache_span_elements(
    const Tensor& kv_cache, const int64_t* dimensions,
    std::size_t dimension_count) {
  uint64_t span = 1;
  for (std::size_t index = 0; index < dimension_count; ++index) {
    const auto dimension = dimensions[index];
    const auto extent =
        static_cast<uint64_t>(kv_cache.size(dimension) - 1);
    const auto stride =
        static_cast<uint64_t>(kv_cache.stride(dimension));
    STD_TORCH_CHECK(
        extent == 0 ||
            stride <=
                (std::numeric_limits<uint64_t>::max() - span) / extent,
        "Loom packed K/V cache span exceeds the bridge ABI");
    span += extent * stride;
  }
  return span;
}

uint64_t kv_cache_partition_span_elements(const Tensor& kv_cache) {
  const int64_t dimensions[] = {2, 3, 4};
  return kv_cache_span_elements(
      kv_cache, dimensions, sizeof(dimensions) / sizeof(dimensions[0]));
}

uint64_t kv_cache_view_span_elements(const Tensor& kv_cache) {
  const int64_t dimensions[] = {0, 2, 3, 4};
  return kv_cache_span_elements(
      kv_cache, dimensions, sizeof(dimensions) / sizeof(dimensions[0]));
}

void check_rope_paged_kv_write_contract(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    const Tensor& kv_cache, const Tensor& key_scales,
    const Tensor& value_scales, const Tensor& slot_mapping) {
  STD_TORCH_CHECK(query.is_cuda(), "Loom RoPE+paged-KV query must be CUDA");
  STD_TORCH_CHECK(key.device() == query.device() &&
                  value.device() == query.device() &&
                  positions.device() == query.device() &&
                  cos_sin_cache.device() == query.device() &&
                  kv_cache.device() == query.device() &&
                  key_scales.device() == query.device() &&
                  value_scales.device() == query.device() &&
                  slot_mapping.device() == query.device(),
              "Loom RoPE+paged-KV tensors must be on one CUDA device");
  STD_TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == cos_sin_cache.scalar_type(),
              "Loom RoPE+paged-KV Q/K/V and cos/sin cache must share a dtype");
  STD_TORCH_CHECK(query.scalar_type() == ScalarType::Float ||
                  query.scalar_type() == ScalarType::Half ||
                  query.scalar_type() == ScalarType::BFloat16,
              "Loom RoPE+paged-KV supports F32, FP16, and BF16 sources");
  STD_TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "Loom RoPE+paged-KV Q/K/V must have rank 3");
  const bool native_cache =
      kv_cache.scalar_type() == query.scalar_type();
  const bool fp8_cache = kv_cache.scalar_type() == ScalarType::Byte;
  STD_TORCH_CHECK(native_cache || fp8_cache,
              "Loom paged K/V cache must use the source dtype or uint8 "
              "FP8 E4M3 storage");
  STD_TORCH_CHECK(key_scales.scalar_type() == ScalarType::Float &&
                  value_scales.scalar_type() == ScalarType::Float &&
                  key_scales.is_contiguous() && value_scales.is_contiguous() &&
                  key_scales.numel() == value_scales.numel() &&
                  (key_scales.numel() == 1 ||
                   key_scales.numel() == key.size(1)),
              "Loom paged K/V cache scales must be contiguous float32 "
              "tensors with one element or one element per KV head");
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Long &&
                  slot_mapping.scalar_type() == ScalarType::Long,
              "Loom RoPE+paged-KV positions and slot mapping must be int64");
  STD_TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
              "Loom RoPE+paged-KV query must be non-empty");
  STD_TORCH_CHECK(key.size(0) == query.size(0) &&
                  value.size(0) == query.size(0),
              "Loom RoPE+paged-KV Q/K/V token counts must match");
  STD_TORCH_CHECK(key.size(1) > 0 && key.size(1) == value.size(1),
              "Loom RoPE+paged-KV K/V head counts must match");
  STD_TORCH_CHECK(key.size(2) == query.size(2),
              "Loom RoPE+paged-KV Q/K head sizes must match");
  STD_TORCH_CHECK(value.size(2) > 0,
              "Loom RoPE+paged-KV value head size must be positive");
  STD_TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1 &&
                  value.stride(2) == 1 && query.stride(0) > 0 &&
                  query.stride(1) > 0 && key.stride(0) > 0 &&
                  key.stride(1) > 0 && value.stride(0) > 0 &&
                  value.stride(1) > 0 && positions.is_contiguous() &&
                  cos_sin_cache.is_contiguous() &&
                  slot_mapping.is_contiguous(),
              "Loom RoPE+paged-KV sources require unit dim stride and positive "
              "token/head strides; metadata must be contiguous");
  STD_TORCH_CHECK(positions.dim() == 1 &&
                  positions.numel() == query.size(0) &&
                  slot_mapping.dim() == 1 &&
                  slot_mapping.numel() <= query.size(0),
              "Loom RoPE positions must cover every token and slot_mapping "
              "must not exceed the padded token count");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(0) > 0 &&
                  cos_sin_cache.size(1) > 0 &&
                  cos_sin_cache.size(1) % 2 == 0 &&
                  cos_sin_cache.size(1) <= query.size(2),
              "Loom RoPE+paged-KV cos/sin cache must be "
              "[max_position, even rotary_dim <= head_size]");
  STD_TORCH_CHECK(kv_cache.dim() == 5 && kv_cache.size(1) == 2,
              "Loom paged K/V cache must have shape "
              "[blocks, 2, block_size, kv_heads, head_size]");
  STD_TORCH_CHECK(kv_cache.size(0) > 0 && kv_cache.size(2) > 0 &&
                  kv_cache.size(3) == key.size(1) &&
                  kv_cache.size(4) == key.size(2) &&
                  value.size(1) == key.size(1) &&
                  value.size(2) == key.size(2),
              "Loom packed K/V cache and K/V sources must share "
              "head count and head size");
  STD_TORCH_CHECK(kv_cache.stride(4) == 1 &&
                  kv_cache.stride(0) > 0 && kv_cache.stride(1) > 0 &&
                  kv_cache.stride(2) > 0 && kv_cache.stride(3) > 0,
              "Loom packed K/V cache requires unit element stride and "
              "positive block/KV/page/head strides");
  const auto partition_span = kv_cache_partition_span_elements(kv_cache);
  const auto kv_stride = static_cast<uint64_t>(kv_cache.stride(1));
  const auto block_stride = static_cast<uint64_t>(kv_cache.stride(0));
  STD_TORCH_CHECK(kv_stride >= partition_span &&
                  block_stride >= kv_stride &&
                  block_stride - kv_stride >= partition_span,
              "Loom packed K/V cache partitions must not overlap");
}

void launch_rope_paged_kv_write(
    Tensor query, Tensor key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    Tensor kv_cache, const Tensor& key_scales,
    const Tensor& value_scales, const Tensor& slot_mapping,
    bool is_neox) {
  const int64_t limits[] = {
      query.size(0),       query.size(1),       key.size(1),
      query.size(2),       value.size(2),       cos_sin_cache.size(1),
      cos_sin_cache.size(0), kv_cache.size(0),  kv_cache.size(2),
  };
  for (const int64_t value_to_check : limits) {
    STD_TORCH_CHECK(value_to_check > 0 &&
                    value_to_check <= std::numeric_limits<uint32_t>::max(),
                "Loom RoPE+paged-KV shape exceeds the CUDA ABI");
  }

  const CudaDeviceGuard device_guard(query.device());
  const auto stream = current_cuda_stream(query.device().index());
  const auto tokens = static_cast<uint32_t>(query.size(0));
  const auto cache_tokens = static_cast<uint32_t>(slot_mapping.numel());
  const auto query_heads = static_cast<uint32_t>(query.size(1));
  const auto kv_heads = static_cast<uint32_t>(key.size(1));
  const auto head_size = static_cast<uint32_t>(query.size(2));
  const auto value_head_size = static_cast<uint32_t>(value.size(2));
  const auto rotary_dim = static_cast<uint32_t>(cos_sin_cache.size(1));
  const auto max_position = static_cast<uint32_t>(cos_sin_cache.size(0));
  const auto num_blocks = static_cast<uint32_t>(kv_cache.size(0));
  const auto block_size = static_cast<uint32_t>(kv_cache.size(2));
  const auto query_token_stride = static_cast<uint64_t>(query.stride(0));
  const auto query_head_stride = static_cast<uint64_t>(query.stride(1));
  const auto key_token_stride = static_cast<uint64_t>(key.stride(0));
  const auto key_head_stride = static_cast<uint64_t>(key.stride(1));
  const auto value_token_stride = static_cast<uint64_t>(value.stride(0));
  const auto value_head_stride = static_cast<uint64_t>(value.stride(1));
  const auto key_block_stride = static_cast<uint64_t>(kv_cache.stride(0));
  const auto key_page_stride = static_cast<uint64_t>(kv_cache.stride(2));
  const auto key_cache_head_stride =
      static_cast<uint64_t>(kv_cache.stride(3));
  const auto value_block_stride = key_block_stride;
  const auto value_page_stride = key_page_stride;
  const auto value_cache_head_stride =
      key_cache_head_stride;
  const uint32_t cache_encoding =
      kv_cache.scalar_type() == ScalarType::Byte
          ? LOOM_CUDA_BRIDGE_KV_CACHE_FP8_E4M3
          : LOOM_CUDA_BRIDGE_KV_CACHE_NATIVE;
  const auto cache_element_size =
      static_cast<uint64_t>(kv_cache.element_size());
  const auto value_cache_offset =
      static_cast<uint64_t>(kv_cache.stride(1));
  STD_TORCH_CHECK(
      cache_element_size == 0 ||
          value_cache_offset <=
              std::numeric_limits<uint64_t>::max() / cache_element_size,
      "Loom packed K/V cache byte offset exceeds uint64");
  auto* key_cache_data =
      static_cast<uint8_t*>(kv_cache.mutable_data_ptr());
  auto* value_cache_data =
      key_cache_data + value_cache_offset * cache_element_size;
  const auto cache_view_span = kv_cache_view_span_elements(kv_cache);

  const int status = loom_cuda_bridge_rope_paged_kv_write(
      bridge_dtype(query), cache_encoding, query.mutable_data_ptr(),
      storage_span_elements(query), key.mutable_data_ptr(),
      storage_span_elements(key), value.const_data_ptr(),
      storage_span_elements(value), positions.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(positions.numel()),
      cos_sin_cache.const_data_ptr(),
      static_cast<uint64_t>(cos_sin_cache.numel()),
      key_cache_data, cache_view_span, value_cache_data, cache_view_span,
      key_scales.const_data_ptr<float>(),
      static_cast<uint64_t>(key_scales.numel()),
      value_scales.const_data_ptr<float>(),
      static_cast<uint64_t>(value_scales.numel()),
      slot_mapping.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(slot_mapping.numel()), tokens, cache_tokens,
      query_heads, kv_heads, head_size, value_head_size, rotary_dim,
      max_position, num_blocks, block_size, query_token_stride,
      query_head_stride, key_token_stride, key_head_stride,
      value_token_stride, value_head_stride, key_block_stride,
      key_page_stride, key_cache_head_stride, value_block_stride,
      value_page_stride, value_cache_head_stride, is_neox ? 1U : 0U,
      stream.stream());
  check_bridge_status(status, "RoPE+paged-KV");
}

void rope_paged_kv_write(
    Tensor query, Tensor key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    Tensor kv_cache, const Tensor& key_scales,
    const Tensor& value_scales, const Tensor& slot_mapping,
    bool is_neox) {
  check_rope_paged_kv_write_contract(query, key, value, positions,
                                     cos_sin_cache, kv_cache, key_scales,
                                     value_scales, slot_mapping);
  launch_rope_paged_kv_write(query, key, value, positions, cos_sin_cache,
                             kv_cache, key_scales, value_scales, slot_mapping,
                             is_neox);
}


}  // namespace loom_kernels::torch_adapter

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl(
      "rope_paged_kv_write_",
      TORCH_BOX(&loom_kernels::torch_adapter::rope_paged_kv_write));
}
