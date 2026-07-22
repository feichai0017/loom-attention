#include "loom_cuda.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kPackedThreads = 256;
constexpr uint32_t kPackedWarps = kPackedThreads / 32;
constexpr uint32_t kMaxContext = 1024;
constexpr uint32_t kSingleHeadMaximumContext = 16;
constexpr uint64_t kFourHeadMinimumKvWorkItems = 128;
constexpr uint64_t kPartialFourHeadMinimumPackedWorkItems = 256;

struct FloatOps {
  using Scalar = float;
  __device__ static float to_float(float value) { return value; }
  __device__ static float from_float(float value) { return value; }
  __device__ static float2 load_pair(const float* values) {
    return *reinterpret_cast<const float2*>(values);
  }
};

struct HalfOps {
  using Scalar = __half;
  __device__ static float to_float(__half value) { return __half2float(value); }
  __device__ static __half from_float(float value) {
    return __float2half_rn(value);
  }
  __device__ static float2 load_pair(const __half* values) {
    return __half22float2(*reinterpret_cast<const __half2*>(values));
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;
  __device__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }
  __device__ static float2 load_pair(const __nv_bfloat16* values) {
    return __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(values));
  }
};

struct Maximum {
  __device__ float operator()(float left, float right) const {
    return fmaxf(left, right);
  }
};

__device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

template <typename Ops>
__global__ void paged_decode_attention_kernel(
    const typename Ops::Scalar* query, const typename Ops::Scalar* key_cache,
    const typename Ops::Scalar* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, typename Ops::Scalar* output,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, float scale) {
  const uint32_t sequence_head = blockIdx.x;
  const uint32_t sequence = sequence_head / query_heads;
  const uint32_t query_head = sequence_head % query_heads;
  const uint32_t queries_per_kv = query_heads / kv_heads;
  const uint32_t kv_head = query_head / queries_per_kv;
  const uint32_t sequence_length =
      static_cast<uint32_t>(sequence_lengths[sequence]);

  extern __shared__ float scores[];
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  __shared__ float maximum;
  __shared__ float inverse_denominator;

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  const size_t query_offset =
      (static_cast<size_t>(sequence) * query_heads + query_head) * head_size;
  const size_t table_offset =
      static_cast<size_t>(sequence) * max_blocks_per_sequence;

  for (uint32_t position = warp; position < sequence_length;
       position += kWarps) {
    const uint32_t logical_block = position / block_size;
    const uint32_t block_offset = position % block_size;
    const uint32_t physical_block =
        static_cast<uint32_t>(block_tables[table_offset + logical_block]);
    const size_t key_offset =
        static_cast<size_t>(physical_block) * key_block_stride +
        (static_cast<size_t>(block_offset) * kv_heads + kv_head) * head_size;
    float partial = 0.0F;
    for (uint32_t dimension = lane; dimension < head_size;
         dimension += 32U) {
      partial += Ops::to_float(query[query_offset + dimension]) *
                 Ops::to_float(key_cache[key_offset + dimension]);
    }
    partial = warp_sum(partial);
    if (lane == 0U) {
      scores[position] = partial * scale;
    }
  }
  __syncthreads();

  float local_maximum = -CUDART_INF_F;
  for (uint32_t position = threadIdx.x; position < sequence_length;
       position += kThreads) {
    local_maximum = fmaxf(local_maximum, scores[position]);
  }
  const float reduced_maximum =
      BlockReduce(reduction_storage).Reduce(local_maximum, Maximum{});
  if (threadIdx.x == 0U) {
    maximum = reduced_maximum;
  }
  __syncthreads();

  float local_denominator = 0.0F;
  for (uint32_t position = threadIdx.x; position < sequence_length;
       position += kThreads) {
    const float weight = expf(scores[position] - maximum);
    scores[position] = weight;
    local_denominator += weight;
  }
  __syncthreads();
  const float denominator =
      BlockReduce(reduction_storage).Sum(local_denominator);
  if (threadIdx.x == 0U) {
    inverse_denominator = 1.0F / denominator;
  }
  __syncthreads();

  const size_t output_offset =
      (static_cast<size_t>(sequence) * query_heads + query_head) *
      value_head_size;
  for (uint32_t dimension = threadIdx.x; dimension < value_head_size;
       dimension += kThreads) {
    float accumulator = 0.0F;
    for (uint32_t position = 0; position < sequence_length; ++position) {
      const uint32_t logical_block = position / block_size;
      const uint32_t block_offset = position % block_size;
      const uint32_t physical_block =
          static_cast<uint32_t>(block_tables[table_offset + logical_block]);
      const size_t value_offset =
          static_cast<size_t>(physical_block) * value_block_stride +
          (static_cast<size_t>(block_offset) * kv_heads + kv_head) *
              value_head_size;
      accumulator += scores[position] *
                     Ops::to_float(value_cache[value_offset + dimension]);
    }
    output[output_offset + dimension] =
        Ops::from_float(accumulator * inverse_denominator);
  }
}

// Keep the established full-group kernel branch-free. Odd GQA ratios compile
// a separate guarded tail, while D64 compiles fixed per-lane Q pairs into
// registers instead of reloading the decode query for every cache position.
template <typename Ops, int PackedQueryHeads, bool CacheHeadSize64Query,
          bool AllowPartialGroup>
__global__ void paged_decode_attention_gqa_kernel(
    const typename Ops::Scalar* query, const typename Ops::Scalar* key_cache,
    const typename Ops::Scalar* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, typename Ops::Scalar* output,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale) {
  constexpr uint32_t packed_query_heads =
      static_cast<uint32_t>(PackedQueryHeads);
  const uint32_t queries_per_kv = query_heads / kv_heads;
  const uint32_t packed_groups_per_kv =
      AllowPartialGroup
          ? (queries_per_kv + packed_query_heads - 1U) / packed_query_heads
          : queries_per_kv / packed_query_heads;
  const uint32_t packed_groups_per_sequence =
      kv_heads * packed_groups_per_kv;
  const uint32_t sequence = blockIdx.x / packed_groups_per_sequence;
  const uint32_t packed_group = blockIdx.x % packed_groups_per_sequence;
  const uint32_t kv_head = packed_group / packed_groups_per_kv;
  const uint32_t subgroup = packed_group % packed_groups_per_kv;
  const uint32_t first_query_head =
      kv_head * queries_per_kv + subgroup * packed_query_heads;
  const uint32_t remaining_query_heads =
      queries_per_kv - subgroup * packed_query_heads;
  const uint32_t active_query_heads =
      remaining_query_heads < packed_query_heads ? remaining_query_heads
                                                 : packed_query_heads;
  const uint32_t sequence_length =
      static_cast<uint32_t>(sequence_lengths[sequence]);

  extern __shared__ unsigned char shared_bytes[];
  float* scores = reinterpret_cast<float*>(shared_bytes);
  const size_t score_bytes =
      static_cast<size_t>(PackedQueryHeads) * max_sequence_length *
      sizeof(float);
  const size_t block_id_start =
      (score_bytes + alignof(uint32_t) - 1U) & ~(alignof(uint32_t) - 1U);
  uint32_t* physical_blocks =
      reinterpret_cast<uint32_t*>(shared_bytes + block_id_start);
  using BlockReduce = cub::BlockReduce<float, kPackedThreads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  __shared__ float maximum;
  __shared__ float inverse_denominators[PackedQueryHeads];

  const size_t table_offset =
      static_cast<size_t>(sequence) * max_blocks_per_sequence;
  for (uint32_t position = threadIdx.x; position < sequence_length;
       position += kPackedThreads) {
    const uint32_t logical_block = position / block_size;
    const uint32_t physical_block =
        static_cast<uint32_t>(block_tables[table_offset + logical_block]);
    physical_blocks[position] = physical_block;
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  float2 cached_query_values[PackedQueryHeads] = {};
  if constexpr (CacheHeadSize64Query) {
#pragma unroll
    for (int packed_head = 0; packed_head < PackedQueryHeads;
         ++packed_head) {
      if (!AllowPartialGroup ||
          static_cast<uint32_t>(packed_head) < active_query_heads) {
        const uint32_t query_head =
            first_query_head + static_cast<uint32_t>(packed_head);
        const size_t query_offset =
            (static_cast<size_t>(sequence) * query_heads + query_head) *
            head_size;
        cached_query_values[packed_head] =
            Ops::load_pair(query + query_offset + lane * 2U);
      }
    }
  }
  for (uint32_t position = warp; position < sequence_length;
       position += kPackedWarps) {
    float partials[PackedQueryHeads] = {};
    const uint32_t block_offset = position % block_size;
    const size_t key_offset =
        static_cast<size_t>(physical_blocks[position]) * key_block_stride +
        (static_cast<size_t>(block_offset) * kv_heads + kv_head) * head_size;
    if constexpr (CacheHeadSize64Query) {
      const float2 key_values =
          Ops::load_pair(key_cache + key_offset + lane * 2U);
#pragma unroll
      for (int packed_head = 0; packed_head < PackedQueryHeads;
           ++packed_head) {
        if (!AllowPartialGroup ||
            static_cast<uint32_t>(packed_head) < active_query_heads) {
          partials[packed_head] =
              fmaf(cached_query_values[packed_head].x, key_values.x,
                   partials[packed_head]);
          partials[packed_head] =
              fmaf(cached_query_values[packed_head].y, key_values.y,
                   partials[packed_head]);
        }
      }
    } else if (head_size % 2U == 0U) {
      for (uint32_t dimension = lane * 2U; dimension < head_size;
           dimension += 64U) {
        const float2 key_values =
            Ops::load_pair(key_cache + key_offset + dimension);
#pragma unroll
        for (int packed_head = 0; packed_head < PackedQueryHeads;
             ++packed_head) {
          if (!AllowPartialGroup ||
              static_cast<uint32_t>(packed_head) < active_query_heads) {
            const uint32_t query_head =
                first_query_head + static_cast<uint32_t>(packed_head);
            const size_t query_offset =
                (static_cast<size_t>(sequence) * query_heads + query_head) *
                head_size;
            const float2 query_values =
                Ops::load_pair(query + query_offset + dimension);
            partials[packed_head] =
                fmaf(query_values.x, key_values.x, partials[packed_head]);
            partials[packed_head] =
                fmaf(query_values.y, key_values.y, partials[packed_head]);
          }
        }
      }
    } else {
      for (uint32_t dimension = lane; dimension < head_size;
           dimension += 32U) {
        const float key_value =
            Ops::to_float(key_cache[key_offset + dimension]);
#pragma unroll
        for (int packed_head = 0; packed_head < PackedQueryHeads;
             ++packed_head) {
          if (!AllowPartialGroup ||
              static_cast<uint32_t>(packed_head) < active_query_heads) {
            const uint32_t query_head =
                first_query_head + static_cast<uint32_t>(packed_head);
            const size_t query_offset =
                (static_cast<size_t>(sequence) * query_heads + query_head) *
                head_size;
            partials[packed_head] +=
                Ops::to_float(query[query_offset + dimension]) * key_value;
          }
        }
      }
    }
#pragma unroll
    for (int packed_head = 0; packed_head < PackedQueryHeads; ++packed_head) {
      if (!AllowPartialGroup ||
          static_cast<uint32_t>(packed_head) < active_query_heads) {
        partials[packed_head] = warp_sum(partials[packed_head]);
        if (lane == 0U) {
          scores[static_cast<size_t>(packed_head) * max_sequence_length +
                 position] = partials[packed_head] * scale;
        }
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int packed_head = 0; packed_head < PackedQueryHeads; ++packed_head) {
    if (!AllowPartialGroup ||
        static_cast<uint32_t>(packed_head) < active_query_heads) {
      float local_maximum = -CUDART_INF_F;
      float* head_scores =
          scores + static_cast<size_t>(packed_head) * max_sequence_length;
      for (uint32_t position = threadIdx.x; position < sequence_length;
           position += kPackedThreads) {
        local_maximum = fmaxf(local_maximum, head_scores[position]);
      }
      const float reduced_maximum =
          BlockReduce(reduction_storage).Reduce(local_maximum, Maximum{});
      if (threadIdx.x == 0U) {
        maximum = reduced_maximum;
      }
      __syncthreads();

      float local_denominator = 0.0F;
      for (uint32_t position = threadIdx.x; position < sequence_length;
           position += kPackedThreads) {
        const float weight = expf(head_scores[position] - maximum);
        head_scores[position] = weight;
        local_denominator += weight;
      }
      __syncthreads();
      const float denominator =
          BlockReduce(reduction_storage).Sum(local_denominator);
      if (threadIdx.x == 0U) {
        inverse_denominators[packed_head] = 1.0F / denominator;
      }
      __syncthreads();
    }
  }

  for (uint32_t dimension = threadIdx.x; dimension < value_head_size;
       dimension += kPackedThreads) {
    float accumulators[PackedQueryHeads] = {};
    for (uint32_t position = 0; position < sequence_length; ++position) {
      const uint32_t block_offset = position % block_size;
      const size_t value_offset =
          static_cast<size_t>(physical_blocks[position]) * value_block_stride +
          (static_cast<size_t>(block_offset) * kv_heads + kv_head) *
              value_head_size;
      const float value =
          Ops::to_float(value_cache[value_offset + dimension]);
#pragma unroll
      for (int packed_head = 0; packed_head < PackedQueryHeads;
           ++packed_head) {
        if (!AllowPartialGroup ||
            static_cast<uint32_t>(packed_head) < active_query_heads) {
          accumulators[packed_head] +=
              scores[static_cast<size_t>(packed_head) * max_sequence_length +
                     position] *
              value;
        }
      }
    }
#pragma unroll
    for (int packed_head = 0; packed_head < PackedQueryHeads; ++packed_head) {
      if (!AllowPartialGroup ||
          static_cast<uint32_t>(packed_head) < active_query_heads) {
        const uint32_t query_head =
            first_query_head + static_cast<uint32_t>(packed_head);
        const size_t output_offset =
            (static_cast<size_t>(sequence) * query_heads + query_head) *
            value_head_size;
        output[output_offset + dimension] = Ops::from_float(
            accumulators[packed_head] * inverse_denominators[packed_head]);
      }
    }
  }
}

template <typename Ops, int PackedQueryHeads>
void launch_paged_decode_attention_gqa(
    const typename Ops::Scalar* query, const typename Ops::Scalar* key_cache,
    const typename Ops::Scalar* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, typename Ops::Scalar* output,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, uint64_t packed_grid_size, size_t shared_bytes,
    cudaStream_t stream) {
  const bool allow_partial_group =
      (query_heads / kv_heads) % static_cast<uint32_t>(PackedQueryHeads) != 0U;
  if (head_size == 64U && allow_partial_group) {
    paged_decode_attention_gqa_kernel<Ops, PackedQueryHeads, true, true>
        <<<static_cast<uint32_t>(packed_grid_size), kPackedThreads,
           shared_bytes,
           stream>>>(query, key_cache, value_cache, block_tables,
                     sequence_lengths, output, query_heads, kv_heads,
                     head_size, value_head_size, block_size, key_block_stride,
                     value_block_stride, max_blocks_per_sequence,
                     max_sequence_length, scale);
  } else if (head_size == 64U) {
    paged_decode_attention_gqa_kernel<Ops, PackedQueryHeads, true, false>
        <<<static_cast<uint32_t>(packed_grid_size), kPackedThreads,
           shared_bytes,
           stream>>>(query, key_cache, value_cache, block_tables,
                     sequence_lengths, output, query_heads, kv_heads,
                     head_size, value_head_size, block_size, key_block_stride,
                     value_block_stride, max_blocks_per_sequence,
                     max_sequence_length, scale);
  } else if (allow_partial_group) {
    paged_decode_attention_gqa_kernel<Ops, PackedQueryHeads, false, true>
        <<<static_cast<uint32_t>(packed_grid_size), kPackedThreads,
           shared_bytes,
           stream>>>(query, key_cache, value_cache, block_tables,
                     sequence_lengths, output, query_heads, kv_heads,
                     head_size, value_head_size, block_size, key_block_stride,
                     value_block_stride, max_blocks_per_sequence,
                     max_sequence_length, scale);
  } else {
    paged_decode_attention_gqa_kernel<Ops, PackedQueryHeads, false, false>
        <<<static_cast<uint32_t>(packed_grid_size), kPackedThreads,
           shared_bytes,
           stream>>>(query, key_cache, value_cache, block_tables,
                     sequence_lengths, output, query_heads, kv_heads,
                     head_size, value_head_size, block_size, key_block_stride,
                     value_block_stride, max_blocks_per_sequence,
                     max_sequence_length, scale);
  }
}

bool checked_product(const uint32_t* dimensions, size_t count) {
  size_t product = 1;
  for (size_t index = 0; index < count; ++index) {
    if (dimensions[index] == 0U ||
        product > std::numeric_limits<size_t>::max() / dimensions[index]) {
      return false;
    }
    product *= dimensions[index];
  }
  return true;
}

template <typename Ops>
int launch_paged_decode_attention(
    const typename Ops::Scalar* query, const typename Ops::Scalar* key_cache,
    const typename Ops::Scalar* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, typename Ops::Scalar* output,
    uint32_t sequences, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t num_blocks,
    uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream) {
  const uint32_t query_dimensions[] = {sequences, query_heads, head_size};
  const uint32_t output_dimensions[] = {sequences, query_heads,
                                        value_head_size};
  const uint32_t key_dimensions[] = {num_blocks, block_size, kv_heads,
                                     head_size};
  const uint32_t value_dimensions[] = {num_blocks, block_size, kv_heads,
                                       value_head_size};
  const uint32_t table_dimensions[] = {sequences,
                                       max_blocks_per_sequence};
  const uint64_t context_capacity =
      static_cast<uint64_t>(block_size) * max_blocks_per_sequence;
  const uint64_t grid_size =
      static_cast<uint64_t>(sequences) * query_heads;
  const bool valid_products =
      checked_product(query_dimensions, 3) &&
      checked_product(output_dimensions, 3) &&
      checked_product(key_dimensions, 4) &&
      checked_product(value_dimensions, 4) &&
      checked_product(table_dimensions, 2);
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || sequence_lengths == nullptr ||
      output == nullptr || sequences == 0U || query_heads == 0U ||
      kv_heads == 0U || head_size == 0U || value_head_size == 0U ||
      num_blocks == 0U || block_size == 0U ||
      max_blocks_per_sequence == 0U || query_heads % kv_heads != 0U ||
      max_sequence_length == 0U || max_sequence_length > kMaxContext ||
      max_sequence_length > context_capacity || !isfinite(scale) ||
      scale <= 0.0F ||
      grid_size > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      !valid_products) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  const size_t key_block_elements =
      static_cast<size_t>(block_size) * kv_heads * head_size;
  const size_t value_block_elements =
      static_cast<size_t>(block_size) * kv_heads * value_head_size;
  const auto valid_block_stride = [num_blocks](uint64_t stride,
                                                size_t block_elements) {
    if (stride < block_elements ||
        stride > std::numeric_limits<size_t>::max()) {
      return false;
    }
    return num_blocks <= 1U ||
           stride <= (std::numeric_limits<size_t>::max() - block_elements) /
                         (num_blocks - 1U);
  };
  if (!valid_block_stride(key_block_stride, key_block_elements) ||
      !valid_block_stride(value_block_stride, value_block_elements)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  const uint32_t queries_per_kv = query_heads / kv_heads;
  const uint64_t kv_work_items =
      static_cast<uint64_t>(sequences) * kv_heads;
  constexpr uint32_t four_packed_query_heads = 4;
  const uint64_t four_head_packed_grid_size =
      kv_work_items *
      ((queries_per_kv + four_packed_query_heads - 1U) /
       four_packed_query_heads);
  // Packing four query heads cuts the grid by 4x. Keep enough independent
  // work on the H20-qualified path to preserve occupancy. Existing evenly
  // packed GQA shapes retain their original threshold; a partial tail group
  // uses its actual packed grid size because it adds another CUDA block.
  if (max_sequence_length > kSingleHeadMaximumContext &&
      queries_per_kv >= four_packed_query_heads &&
      ((queries_per_kv % four_packed_query_heads == 0U &&
       kv_work_items >= kFourHeadMinimumKvWorkItems) ||
       (queries_per_kv % four_packed_query_heads != 0U &&
        four_head_packed_grid_size >=
            kPartialFourHeadMinimumPackedWorkItems))) {
    constexpr uint32_t packed_query_heads = four_packed_query_heads;
    const uint64_t packed_grid_size = four_head_packed_grid_size;
    const size_t score_bytes =
        static_cast<size_t>(packed_query_heads) * max_sequence_length *
        sizeof(float);
    const size_t block_id_start =
        (score_bytes + alignof(uint32_t) - 1U) & ~(alignof(uint32_t) - 1U);
    const size_t shared_bytes =
        block_id_start +
        static_cast<size_t>(max_sequence_length) * sizeof(uint32_t);
    launch_paged_decode_attention_gqa<Ops, packed_query_heads>(
        query, key_cache, value_cache, block_tables, sequence_lengths, output,
        query_heads, kv_heads, head_size, value_head_size, block_size,
        key_block_stride, value_block_stride, max_blocks_per_sequence,
        max_sequence_length, scale, packed_grid_size, shared_bytes,
        cuda_stream);
  } else if (max_sequence_length > kSingleHeadMaximumContext &&
             queries_per_kv >= 2U) {
    constexpr uint32_t packed_query_heads = 2;
    const uint64_t packed_grid_size =
        kv_work_items *
        ((queries_per_kv + packed_query_heads - 1U) / packed_query_heads);
    const size_t score_bytes =
        static_cast<size_t>(packed_query_heads) * max_sequence_length *
        sizeof(float);
    const size_t block_id_start =
        (score_bytes + alignof(uint32_t) - 1U) & ~(alignof(uint32_t) - 1U);
    const size_t shared_bytes =
        block_id_start +
        static_cast<size_t>(max_sequence_length) * sizeof(uint32_t);
    launch_paged_decode_attention_gqa<Ops, packed_query_heads>(
        query, key_cache, value_cache, block_tables, sequence_lengths, output,
        query_heads, kv_heads, head_size, value_head_size, block_size,
        key_block_stride, value_block_stride, max_blocks_per_sequence,
        max_sequence_length, scale, packed_grid_size, shared_bytes,
        cuda_stream);
  } else {
    const size_t shared_bytes =
        static_cast<size_t>(max_sequence_length) * sizeof(float);
    paged_decode_attention_kernel<Ops>
        <<<static_cast<uint32_t>(grid_size), kThreads, shared_bytes,
           cuda_stream>>>(query, key_cache, value_cache, block_tables,
                          sequence_lengths, output, query_heads, kv_heads,
                          head_size, value_head_size, block_size,
                          key_block_stride, value_block_stride,
                          max_blocks_per_sequence, scale);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_paged_decode_attention_f32(
    const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* sequence_lengths,
    float* output, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream) {
  return launch_paged_decode_attention<FloatOps>(
      query, key_cache, value_cache, block_tables, sequence_lengths, output,
      sequences, query_heads, kv_heads, head_size, value_head_size, num_blocks,
      block_size, key_block_stride, value_block_stride,
      max_blocks_per_sequence, max_sequence_length, scale, stream);
}

extern "C" int loom_cuda_paged_decode_attention_f16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream) {
  return launch_paged_decode_attention<HalfOps>(
      reinterpret_cast<const __half*>(query),
      reinterpret_cast<const __half*>(key_cache),
      reinterpret_cast<const __half*>(value_cache), block_tables,
      sequence_lengths, reinterpret_cast<__half*>(output), sequences,
      query_heads, kv_heads, head_size, value_head_size, num_blocks, block_size,
      key_block_stride, value_block_stride, max_blocks_per_sequence,
      max_sequence_length, scale, stream);
}

extern "C" int loom_cuda_paged_decode_attention_bf16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream) {
  return launch_paged_decode_attention<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(query),
      reinterpret_cast<const __nv_bfloat16*>(key_cache),
      reinterpret_cast<const __nv_bfloat16*>(value_cache), block_tables,
      sequence_lengths, reinterpret_cast<__nv_bfloat16*>(output), sequences,
      query_heads, kv_heads, head_size, value_head_size, num_blocks, block_size,
      key_block_stride, value_block_stride, max_blocks_per_sequence,
      max_sequence_length, scale, stream);
}
