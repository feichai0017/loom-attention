#include "loom_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int32_t kPlaceholderTokenId = -1;
constexpr int kThreads = 32;

__global__ __launch_bounds__(kThreads) void greedy_speculative_verify_kernel(
    const int32_t* draft_token_ids, const int64_t* target_token_ids,
    const int32_t* bonus_token_ids,
    const int32_t* cumulative_draft_lengths, int32_t* output_token_ids,
    int32_t* accepted_lengths, int32_t* emitted_lengths,
    uint32_t draft_tokens, uint32_t max_draft_tokens) {
  const uint32_t request = blockIdx.x;
  const size_t output_width = static_cast<size_t>(max_draft_tokens) + 1U;
  const size_t output_offset = static_cast<size_t>(request) * output_width;

  for (uint32_t position = threadIdx.x; position <= max_draft_tokens;
       position += blockDim.x) {
    output_token_ids[output_offset + position] = kPlaceholderTokenId;
  }
  __syncwarp();

  const int32_t start =
      request == 0 ? 0 : cumulative_draft_lengths[request - 1U];
  const int32_t end = cumulative_draft_lengths[request];
  const bool valid =
      start >= 0 && end >= start &&
      static_cast<uint32_t>(end) <= draft_tokens &&
      static_cast<uint32_t>(end - start) <= max_draft_tokens;
  if (!valid) {
    if (threadIdx.x == 0) {
      accepted_lengths[request] = 0;
      emitted_lengths[request] = 0;
    }
    return;
  }

  const uint32_t request_draft_tokens = static_cast<uint32_t>(end - start);
  int32_t first_mismatch = static_cast<int32_t>(request_draft_tokens);
  for (uint32_t position = threadIdx.x; position < request_draft_tokens;
       position += blockDim.x) {
    const size_t token = static_cast<size_t>(start) + position;
    if (static_cast<int64_t>(draft_token_ids[token]) !=
        target_token_ids[token]) {
      first_mismatch =
          min(first_mismatch, static_cast<int32_t>(position));
    }
  }
  constexpr unsigned kFullWarp = 0xffffffffU;
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    first_mismatch =
        min(first_mismatch,
            __shfl_down_sync(kFullWarp, first_mismatch, offset));
  }

  const uint32_t accepted = static_cast<uint32_t>(
      __shfl_sync(kFullWarp, first_mismatch, 0));
  for (uint32_t position = threadIdx.x; position < accepted;
       position += blockDim.x) {
    output_token_ids[output_offset + position] =
        draft_token_ids[static_cast<size_t>(start) + position];
  }
  if (threadIdx.x == 0) {
    if (accepted < request_draft_tokens) {
      output_token_ids[output_offset + accepted] = static_cast<int32_t>(
          target_token_ids[static_cast<size_t>(start) + accepted]);
    } else {
      output_token_ids[output_offset + accepted] = bonus_token_ids[request];
    }
    accepted_lengths[request] = static_cast<int32_t>(accepted);
    emitted_lengths[request] = static_cast<int32_t>(accepted + 1U);
  }
}

}  // namespace

extern "C" int loom_cuda_greedy_speculative_verify(
    const int32_t* draft_token_ids, const int64_t* target_token_ids,
    const int32_t* bonus_token_ids,
    const int32_t* cumulative_draft_lengths, int32_t* output_token_ids,
    int32_t* accepted_lengths, int32_t* emitted_lengths, uint32_t requests,
    uint32_t draft_tokens, uint32_t max_draft_tokens, void* stream) {
  if (draft_token_ids == nullptr || target_token_ids == nullptr ||
      bonus_token_ids == nullptr || cumulative_draft_lengths == nullptr ||
      output_token_ids == nullptr || accepted_lengths == nullptr ||
      emitted_lengths == nullptr || requests == 0 || draft_tokens == 0 ||
      max_draft_tokens == 0 ||
      max_draft_tokens == std::numeric_limits<uint32_t>::max() ||
      requests > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      static_cast<uint64_t>(draft_tokens) >
          static_cast<uint64_t>(requests) * max_draft_tokens ||
      static_cast<uint64_t>(requests) *
              (static_cast<uint64_t>(max_draft_tokens) + 1U) >
          std::numeric_limits<size_t>::max()) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  greedy_speculative_verify_kernel<<<requests, kThreads, 0, cuda_stream>>>(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, output_token_ids, accepted_lengths,
      emitted_lengths, draft_tokens, max_draft_tokens);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}
