#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

struct FloatOps {
  using Scalar = float;

  __device__ static float to_float(Scalar value) { return value; }
  __device__ static Scalar from_float(float value) { return value; }
};

struct HalfOps {
  using Scalar = __half;

  __device__ static float to_float(Scalar value) {
    return __half2float(value);
  }
  __device__ static Scalar from_float(float value) {
    return __float2half_rn(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;

  __device__ static float to_float(Scalar value) {
    return __bfloat162float(value);
  }
  __device__ static Scalar from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <typename Ops>
__global__ __launch_bounds__(kThreads) void rope_paged_kv_write_kernel(
    typename Ops::Scalar* query, typename Ops::Scalar* key,
    const typename Ops::Scalar* value, const int64_t* positions,
    const typename Ops::Scalar* cos_sin_cache,
    typename Ops::Scalar* key_cache, typename Ops::Scalar* value_cache,
    const int64_t* slot_mapping, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t rotary_dim,
    uint32_t max_position, uint32_t cache_tokens, uint32_t num_blocks,
    uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride,
    uint64_t key_cache_block_stride, uint64_t key_cache_page_stride,
    uint64_t key_cache_head_stride, uint64_t value_cache_block_stride,
    uint64_t value_cache_page_stride, uint64_t value_cache_head_stride,
    bool is_neox) {
  using Scalar = typename Ops::Scalar;

  const uint32_t token = blockIdx.x;
  const int64_t position = positions[token];
  if (position < 0 || static_cast<uint64_t>(position) >= max_position) {
    // Engine metadata is a trusted precondition. Keep the guard to prevent an
    // invalid position from becoming an out-of-bounds device read.
    return;
  }

  const uint32_t half_rotary_dim = rotary_dim / 2U;
  const Scalar* cos_sin =
      cos_sin_cache + static_cast<size_t>(position) * rotary_dim;

  const uint64_t query_pairs =
      static_cast<uint64_t>(query_heads) * half_rotary_dim;
  const uint64_t query_token_offset =
      static_cast<uint64_t>(token) * query_token_stride;
  for (uint64_t linear_pair = threadIdx.x; linear_pair < query_pairs;
       linear_pair += blockDim.x) {
    const uint32_t head = linear_pair / half_rotary_dim;
    const uint32_t pair = linear_pair % half_rotary_dim;
    const uint32_t first_dim = is_neox ? pair : pair * 2U;
    const uint32_t second_dim =
        is_neox ? pair + half_rotary_dim : pair * 2U + 1U;
    const uint64_t head_offset =
        query_token_offset + static_cast<uint64_t>(head) * query_head_stride;
    const float first = Ops::to_float(query[head_offset + first_dim]);
    const float second = Ops::to_float(query[head_offset + second_dim]);
    const float cosine = Ops::to_float(cos_sin[pair]);
    const float sine = Ops::to_float(cos_sin[half_rotary_dim + pair]);
    query[head_offset + first_dim] =
        Ops::from_float(first * cosine - second * sine);
    query[head_offset + second_dim] =
        Ops::from_float(second * cosine + first * sine);
  }

  const uint64_t slot_capacity =
      static_cast<uint64_t>(num_blocks) * block_size;
  const bool has_cache_slot = token < cache_tokens;
  const int64_t slot = has_cache_slot ? slot_mapping[token] : -1;
  const bool write_cache =
      slot >= 0 && static_cast<uint64_t>(slot) < slot_capacity;
  uint64_t cache_block = 0;
  uint64_t cache_page = 0;
  if (write_cache) {
    cache_block = static_cast<uint64_t>(slot) / block_size;
    cache_page = static_cast<uint64_t>(slot) % block_size;
  }

  const uint64_t key_pairs =
      static_cast<uint64_t>(kv_heads) * half_rotary_dim;
  const uint64_t key_token_offset =
      static_cast<uint64_t>(token) * key_token_stride;
  for (uint64_t linear_pair = threadIdx.x; linear_pair < key_pairs;
       linear_pair += blockDim.x) {
    const uint32_t head = linear_pair / half_rotary_dim;
    const uint32_t pair = linear_pair % half_rotary_dim;
    const uint32_t first_dim = is_neox ? pair : pair * 2U;
    const uint32_t second_dim =
        is_neox ? pair + half_rotary_dim : pair * 2U + 1U;
    const uint64_t source_head_offset =
        key_token_offset + static_cast<uint64_t>(head) * key_head_stride;
    const float first = Ops::to_float(key[source_head_offset + first_dim]);
    const float second = Ops::to_float(key[source_head_offset + second_dim]);
    const float cosine = Ops::to_float(cos_sin[pair]);
    const float sine = Ops::to_float(cos_sin[half_rotary_dim + pair]);
    const Scalar rotated_first =
        Ops::from_float(first * cosine - second * sine);
    const Scalar rotated_second =
        Ops::from_float(second * cosine + first * sine);
    key[source_head_offset + first_dim] = rotated_first;
    key[source_head_offset + second_dim] = rotated_second;

    if (write_cache) {
      const uint64_t target_head_offset =
          cache_block * key_cache_block_stride +
          cache_page * key_cache_page_stride +
          static_cast<uint64_t>(head) * key_cache_head_stride;
      key_cache[target_head_offset + first_dim] = rotated_first;
      key_cache[target_head_offset + second_dim] = rotated_second;
    }
  }

  if (write_cache && rotary_dim < head_size) {
    const uint64_t key_tail_elements =
        static_cast<uint64_t>(kv_heads) * (head_size - rotary_dim);
    for (uint64_t linear_tail = threadIdx.x;
         linear_tail < key_tail_elements; linear_tail += blockDim.x) {
      const uint32_t head = linear_tail / (head_size - rotary_dim);
      const uint32_t dim =
          rotary_dim + linear_tail % (head_size - rotary_dim);
      const uint64_t source =
          key_token_offset + static_cast<uint64_t>(head) * key_head_stride +
          dim;
      const uint64_t target = cache_block * key_cache_block_stride +
                              cache_page * key_cache_page_stride +
                              static_cast<uint64_t>(head) *
                                  key_cache_head_stride +
                              dim;
      key_cache[target] = key[source];
    }
  }

  if (write_cache) {
    const uint64_t value_elements =
        static_cast<uint64_t>(kv_heads) * value_head_size;
    const uint64_t value_token_offset =
        static_cast<uint64_t>(token) * value_token_stride;
    for (uint64_t linear_value = threadIdx.x;
         linear_value < value_elements; linear_value += blockDim.x) {
      const uint32_t head = linear_value / value_head_size;
      const uint32_t dim = linear_value % value_head_size;
      const uint64_t source =
          value_token_offset + static_cast<uint64_t>(head) * value_head_stride +
          dim;
      const uint64_t target = cache_block * value_cache_block_stride +
                              cache_page * value_cache_page_stride +
                              static_cast<uint64_t>(head) *
                                  value_cache_head_stride +
                              dim;
      value_cache[target] = value[source];
    }
  }
}

template <typename Ops>
int launch_rope_paged_kv_write(
    typename Ops::Scalar* query, typename Ops::Scalar* key,
    const typename Ops::Scalar* value, const int64_t* positions,
    const typename Ops::Scalar* cos_sin_cache,
    typename Ops::Scalar* key_cache, typename Ops::Scalar* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      positions == nullptr || cos_sin_cache == nullptr ||
      key_cache == nullptr || value_cache == nullptr || tokens == 0 ||
      (cache_tokens > 0 && slot_mapping == nullptr) || cache_tokens > tokens ||
      query_heads == 0 || kv_heads == 0 || head_size == 0 ||
      value_head_size == 0 ||
      rotary_dim == 0 || (rotary_dim & 1U) != 0U ||
      rotary_dim > head_size || max_position == 0 || num_blocks == 0 ||
      block_size == 0 || query_token_stride == 0 ||
      query_head_stride == 0 || key_token_stride == 0 ||
      key_head_stride == 0 || value_token_stride == 0 ||
      value_head_stride == 0 || key_cache_block_stride == 0 ||
      key_cache_page_stride == 0 || key_cache_head_stride == 0 ||
      value_cache_block_stride == 0 || value_cache_page_stride == 0 ||
      value_cache_head_stride == 0 || is_neox > 1U ||
      tokens > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  rope_paged_kv_write_kernel<Ops><<<tokens, kThreads, 0,
                                    static_cast<cudaStream_t>(stream)>>>(
      query, key, value, positions, cos_sin_cache, key_cache, value_cache,
      slot_mapping, query_heads, kv_heads, head_size, value_head_size,
      rotary_dim, max_position, cache_tokens, num_blocks, block_size,
      query_token_stride, query_head_stride, key_token_stride, key_head_stride,
      value_token_stride, value_head_stride, key_cache_block_stride,
      key_cache_page_stride, key_cache_head_stride,
      value_cache_block_stride, value_cache_page_stride,
      value_cache_head_stride, is_neox != 0U);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_rope_paged_kv_write_f32(
    float* query, float* key, const float* value, const int64_t* positions,
    const float* cos_sin_cache, float* key_cache, float* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream) {
  return launch_rope_paged_kv_write<FloatOps>(
      query, key, value, positions, cos_sin_cache, key_cache, value_cache,
      slot_mapping, tokens, cache_tokens, query_heads, kv_heads, head_size,
      value_head_size, rotary_dim, max_position, num_blocks, block_size,
      query_token_stride, query_head_stride, key_token_stride, key_head_stride,
      value_token_stride, value_head_stride, key_cache_block_stride,
      key_cache_page_stride, key_cache_head_stride,
      value_cache_block_stride, value_cache_page_stride,
      value_cache_head_stride, is_neox, stream);
}

extern "C" int loom_cuda_rope_paged_kv_write_f16(
    uint16_t* query, uint16_t* key, const uint16_t* value,
    const int64_t* positions, const uint16_t* cos_sin_cache,
    uint16_t* key_cache, uint16_t* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream) {
  return launch_rope_paged_kv_write<HalfOps>(
      reinterpret_cast<__half*>(query), reinterpret_cast<__half*>(key),
      reinterpret_cast<const __half*>(value), positions,
      reinterpret_cast<const __half*>(cos_sin_cache),
      reinterpret_cast<__half*>(key_cache),
      reinterpret_cast<__half*>(value_cache), slot_mapping, tokens,
      cache_tokens, query_heads, kv_heads, head_size, value_head_size,
      rotary_dim, max_position, num_blocks, block_size, query_token_stride,
      query_head_stride, key_token_stride, key_head_stride, value_token_stride,
      value_head_stride, key_cache_block_stride, key_cache_page_stride,
      key_cache_head_stride, value_cache_block_stride,
      value_cache_page_stride, value_cache_head_stride, is_neox, stream);
}

extern "C" int loom_cuda_rope_paged_kv_write_bf16(
    uint16_t* query, uint16_t* key, const uint16_t* value,
    const int64_t* positions, const uint16_t* cos_sin_cache,
    uint16_t* key_cache, uint16_t* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream) {
  return launch_rope_paged_kv_write<Bfloat16Ops>(
      reinterpret_cast<__nv_bfloat16*>(query),
      reinterpret_cast<__nv_bfloat16*>(key),
      reinterpret_cast<const __nv_bfloat16*>(value), positions,
      reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache),
      reinterpret_cast<__nv_bfloat16*>(key_cache),
      reinterpret_cast<__nv_bfloat16*>(value_cache), slot_mapping, tokens,
      cache_tokens, query_heads, kv_heads, head_size, value_head_size,
      rotary_dim, max_position, num_blocks, block_size, query_token_stride,
      query_head_stride, key_token_stride, key_head_stride, value_token_stride,
      value_head_stride, key_cache_block_stride, key_cache_page_stride,
      key_cache_head_stride, value_cache_block_stride,
      value_cache_page_stride, value_cache_head_stride, is_neox, stream);
}
