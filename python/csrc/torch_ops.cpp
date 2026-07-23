#include "common.h"

STABLE_TORCH_LIBRARY(loom_kernels, library) {
  library.def(
      "rms_norm(Tensor input_tensor, Tensor weight, Tensor(a!) output, "
      "float epsilon) -> ()");
  library.def(
      "add_rms_norm_mut(Tensor(a!) input_tensor, Tensor(b!) residual, "
      "Tensor weight, float epsilon) -> ()");
  library.def(
      "rms_norm_dynamic_fp8(Tensor input_tensor, Tensor weight, "
      "Tensor(a!) output, Tensor(b!) scales, float epsilon) -> ()");
  library.def(
      "silu_and_mul(Tensor input_tensor, Tensor(a!) output) -> ()");
  library.def(
      "silu_and_mul_dynamic_fp8(Tensor input_tensor, Tensor(a!) output, "
      "Tensor(b!) scales, int group_size) -> ()");
  library.def(
      "silu_and_mul_per_block_fp8(Tensor(a!) out, Tensor input, "
      "Tensor(b!) scales, int group_size, Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");
  library.def(
      "greedy_sample_logprobs(Tensor logits) -> (Tensor token_ids, Tensor "
      "logprobs, Tensor ranks)");
  library.def(
      "selected_token_logprobs(Tensor logits, Tensor token_ids) -> (Tensor "
      "logprobs, Tensor ranks)");
  library.def(
      "greedy_speculative_verify(Tensor draft_token_ids, Tensor "
      "target_token_ids, Tensor bonus_token_ids, Tensor "
      "cumulative_draft_lengths, Tensor(a!) output_token_ids, Tensor(b!) "
      "accepted_lengths, Tensor(c!) emitted_lengths, int "
      "max_draft_tokens) -> ()");
  library.def("min_p_filter_(Tensor(a!) logits, Tensor min_p) -> ()");
  library.def(
      "paged_decode_attention(Tensor query, Tensor key_cache, Tensor "
      "value_cache, Tensor block_tables, Tensor sequence_lengths, "
      "Tensor(a!) output, int max_sequence_length, float scale) -> ()");
  library.def(
      "rope_paged_kv_write_(Tensor(a!) query, Tensor(b!) key, Tensor value, "
      "Tensor positions, Tensor cos_sin_cache, Tensor(c!) kv_cache, Tensor "
      "key_scales, Tensor value_scales, Tensor slot_mapping, bool is_neox) "
      "-> ()");
  library.def("bridge_abi_version() -> int");
  library.def("bridge_launch_count(int operation) -> int");
  library.def("reset_bridge_launch_count(int operation) -> ()");
}
