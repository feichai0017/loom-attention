#include "loom_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace {

std::atomic<int64_t> vllm_silu_and_mul_per_block_fp8_launches{0};

bool byte_ranges_overlap(const at::Tensor& left, const at::Tensor& right) {
  const auto left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
  const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
  const auto left_end = left_begin + left.nbytes();
  const auto right_end = right_begin + right.nbytes();
  return left_begin < right_end && right_begin < left_end;
}

void check_contract(const at::Tensor& input, const at::Tensor& residual,
                    const at::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Loom Add+RMSNorm input must be CUDA");
  TORCH_CHECK(residual.device() == input.device() &&
                  weight.device() == input.device(),
              "Loom Add+RMSNorm tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(),
              "Loom Add+RMSNorm tensors must have matching dtypes");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom Add+RMSNorm supports F32, FP16, and BF16");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous(),
              "Loom Add+RMSNorm tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom Add+RMSNorm input must be non-empty");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "Loom Add+RMSNorm input/residual shapes must match");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom Add+RMSNorm weight must match the hidden dimension");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom Add+RMSNorm epsilon must be finite and positive");
  TORCH_CHECK(!byte_ranges_overlap(input, residual) &&
                  !byte_ranges_overlap(input, weight) &&
                  !byte_ranges_overlap(residual, weight),
              "Loom Add+RMSNorm tensor storage ranges must not overlap");
}

void launch_add_rms_norm(at::Tensor input, at::Tensor residual,
                         const at::Tensor& weight, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom Add+RMSNorm shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto epsilon_f32 = static_cast<float>(epsilon);
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kFloat) {
    status = loom_cuda_add_rms_norm_f32(
        input.data_ptr<float>(), residual.data_ptr<float>(),
        weight.data_ptr<float>(), rows, hidden_size, epsilon_f32, stream.stream());
  } else if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_add_rms_norm_f16(
        reinterpret_cast<uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::Half>()), rows,
        hidden_size, epsilon_f32, stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_add_rms_norm_bf16(
        reinterpret_cast<uint16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()), rows,
        hidden_size, epsilon_f32, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA Add+RMSNorm launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
}

void add_rms_norm_mut(at::Tensor input, at::Tensor residual,
                      const at::Tensor& weight, double epsilon) {
  check_contract(input, residual, weight, epsilon);
  launch_add_rms_norm(input, residual, weight, epsilon);
}

void check_dynamic_fp8_contract(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& output,
                                const at::Tensor& scales, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Loom RMSNorm+FP8 input must be CUDA");
  TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom RMSNorm+FP8 tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "Loom RMSNorm+FP8 input and weight dtypes must match");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom RMSNorm+FP8 supports F32, FP16, and BF16 inputs");
  TORCH_CHECK(output.scalar_type() == at::kFloat8_e4m3fn,
              "Loom RMSNorm+FP8 output must use torch.float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == at::kFloat,
              "Loom RMSNorm+FP8 scales must use F32");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous() && scales.is_contiguous(),
              "Loom RMSNorm+FP8 tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm+FP8 input must be non-empty");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm+FP8 weight must match the hidden dimension");
  TORCH_CHECK(output.sizes() == input.sizes(),
              "Loom RMSNorm+FP8 output shape must match input");
  const int64_t rows = input.numel() / input.size(-1);
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == 1,
              "Loom RMSNorm+FP8 scales must have shape [rows, 1]");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm+FP8 epsilon must be finite and positive");
  TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight) &&
                  !byte_ranges_overlap(output, scales) &&
                  !byte_ranges_overlap(scales, input) &&
                  !byte_ranges_overlap(scales, weight),
              "Loom RMSNorm+FP8 mutable tensor storage must not overlap");
}

void launch_rms_norm_dynamic_fp8(const at::Tensor& input,
                                 const at::Tensor& weight, at::Tensor output,
                                 at::Tensor scales, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm+FP8 shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto epsilon_f32 = static_cast<float>(epsilon);
  auto* output_bytes = reinterpret_cast<uint8_t*>(output.data_ptr());
  auto* scale_values = scales.data_ptr<float>();
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kFloat) {
    status = loom_cuda_rms_norm_dynamic_fp8_f32(
        input.data_ptr<float>(), weight.data_ptr<float>(), output_bytes,
        scale_values, rows, hidden_size, epsilon_f32, stream.stream());
  } else if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_rms_norm_dynamic_fp8_f16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::Half>()),
        output_bytes, scale_values, rows, hidden_size, epsilon_f32,
        stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_rms_norm_dynamic_fp8_bf16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
        output_bytes, scale_values, rows, hidden_size, epsilon_f32,
        stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA RMSNorm+FP8 launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
}

void rms_norm_dynamic_fp8(const at::Tensor& input, const at::Tensor& weight,
                          at::Tensor output, at::Tensor scales,
                          double epsilon) {
  check_dynamic_fp8_contract(input, weight, output, scales, epsilon);
  launch_rms_norm_dynamic_fp8(input, weight, output, scales, epsilon);
}

void check_silu_and_mul_contract(const at::Tensor& input,
                                 const at::Tensor& output) {
  TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul input must be CUDA");
  TORCH_CHECK(output.device() == input.device(),
              "Loom SiLU-and-Mul tensors must be on the same CUDA device");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "Loom SiLU-and-Mul input/output dtypes must match");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom SiLU-and-Mul supports F32, FP16, and BF16");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul input must be non-empty");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul input last dimension must be even");
  TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul output prefix shape must match input");
  }
  TORCH_CHECK(output.size(-1) == input.size(-1) / 2,
              "Loom SiLU-and-Mul output last dimension must be half input");
  TORCH_CHECK(!byte_ranges_overlap(input, output),
              "Loom SiLU-and-Mul input/output storage must not overlap");
}

void launch_silu_and_mul(const at::Tensor& input, at::Tensor output) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kFloat) {
    status = loom_cuda_silu_and_mul_f32(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, width,
        stream.stream());
  } else if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_silu_and_mul_f16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(output.data_ptr<at::Half>()), rows, width,
        stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_silu_and_mul_bf16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>()), rows,
        width, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA SiLU-and-Mul launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
}

void silu_and_mul(const at::Tensor& input, at::Tensor output) {
  check_silu_and_mul_contract(input, output);
  launch_silu_and_mul(input, output);
}

void check_silu_and_mul_dynamic_fp8_contract(const at::Tensor& input,
                                              const at::Tensor& output,
                                              const at::Tensor& scales,
                                              int64_t group_size,
                                              bool scales_transposed = false) {
  TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul+FP8 input must be CUDA");
  TORCH_CHECK(output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom SiLU-and-Mul+FP8 tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom SiLU-and-Mul+FP8 supports FP16 and BF16 input");
  TORCH_CHECK(output.scalar_type() == at::kFloat8_e4m3fn,
              "Loom SiLU-and-Mul+FP8 output must use torch.float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == at::kFloat,
              "Loom SiLU-and-Mul+FP8 scales must use F32");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul+FP8 input/output must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul+FP8 input must be non-empty");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul+FP8 input last dimension must be even");
  TORCH_CHECK(group_size == 64 || group_size == 128,
              "Loom SiLU-and-Mul+FP8 group size must be 64 or 128");
  const int64_t width = input.size(-1) / 2;
  TORCH_CHECK(width % group_size == 0,
              "Loom SiLU-and-Mul+FP8 width must be divisible by group size");
  TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul+FP8 output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul+FP8 output prefix shape must match input");
  }
  TORCH_CHECK(output.size(-1) == width,
              "Loom SiLU-and-Mul+FP8 output last dimension must be half input");
  const int64_t rows = input.numel() / input.size(-1);
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == width / group_size,
              "Loom SiLU-and-Mul+FP8 scales must have shape "
              "[rows, width / group_size]");
  if (scales_transposed) {
    TORCH_CHECK(scales.stride(0) == 1 && scales.stride(1) == rows,
                "Loom transposed FP8 scales must use group-major storage");
  } else {
    TORCH_CHECK(scales.is_contiguous(),
                "Loom row-major FP8 scales must be contiguous");
  }
  TORCH_CHECK(!byte_ranges_overlap(input, output) &&
                  !byte_ranges_overlap(input, scales) &&
                  !byte_ranges_overlap(output, scales),
              "Loom SiLU-and-Mul+FP8 mutable tensor storage must not overlap");
}

void launch_silu_and_mul_dynamic_fp8_layout(
    const at::Tensor& input, at::Tensor output, at::Tensor scales,
    int64_t group_size_i64, const std::optional<at::Tensor>& scale_ub,
    bool scales_transposed) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max() &&
                  group_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul+FP8 shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const auto group_size = static_cast<uint32_t>(group_size_i64);
  auto* output_bytes = reinterpret_cast<uint8_t*>(output.data_ptr());
  auto* scale_values = scales.data_ptr<float>();
  const float* scale_ub_value =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_silu_and_mul_dynamic_fp8_f16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::Half>()),
        output_bytes, scale_values, rows, width, group_size, scale_ub_value,
        scales_transposed ? 1U : 0U, stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_silu_and_mul_dynamic_fp8_bf16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>()),
        output_bytes, scale_values, rows, width, group_size, scale_ub_value,
        scales_transposed ? 1U : 0U, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA SiLU-and-Mul+FP8 launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
}

void launch_silu_and_mul_dynamic_fp8(const at::Tensor& input,
                                      at::Tensor output, at::Tensor scales,
                                      int64_t group_size) {
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         std::nullopt, false);
}

void silu_and_mul_dynamic_fp8(const at::Tensor& input, at::Tensor output,
                              at::Tensor scales, int64_t group_size) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size);
  launch_silu_and_mul_dynamic_fp8(input, output, scales, group_size);
}

void vllm_silu_and_mul_per_block_fp8(
    at::Tensor output, const at::Tensor& input, at::Tensor scales,
    int64_t group_size, const std::optional<at::Tensor>& scale_ub,
    bool scales_transposed) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size,
                                          scales_transposed);
  if (scale_ub.has_value()) {
    TORCH_CHECK(scale_ub->device() == input.device() &&
                    scale_ub->scalar_type() == at::kFloat &&
                    scale_ub->numel() == 1 && scale_ub->is_contiguous(),
                "Loom FP8 scale upper bound must be one same-device F32 value");
  }
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         scale_ub, scales_transposed);
  vllm_silu_and_mul_per_block_fp8_launches.fetch_add(
      1, std::memory_order_relaxed);
}

int64_t vllm_silu_and_mul_per_block_fp8_launch_count() {
  return vllm_silu_and_mul_per_block_fp8_launches.load(
      std::memory_order_relaxed);
}

void reset_vllm_silu_and_mul_per_block_fp8_launch_count() {
  vllm_silu_and_mul_per_block_fp8_launches.store(0,
                                                 std::memory_order_relaxed);
}

}  // namespace

TORCH_LIBRARY(loom_kernels, library) {
  library.def(
      "add_rms_norm_mut(Tensor(a!) input_tensor, Tensor(b!) residual, "
      "Tensor weight, float epsilon) -> ()");
  library.def(
      "add_rms_norm_mut_unchecked(Tensor(a!) input_tensor, Tensor(b!) "
      "residual, Tensor weight, float epsilon) -> ()");
  library.def(
      "rms_norm_dynamic_fp8(Tensor input_tensor, Tensor weight, "
      "Tensor(a!) output, Tensor(b!) scales, float epsilon) -> ()");
  library.def(
      "rms_norm_dynamic_fp8_unchecked(Tensor input_tensor, Tensor weight, "
      "Tensor(a!) output, Tensor(b!) scales, float epsilon) -> ()");
  library.def(
      "silu_and_mul(Tensor input_tensor, Tensor(a!) output) -> ()");
  library.def(
      "silu_and_mul_unchecked(Tensor input_tensor, Tensor(a!) output) -> ()");
  library.def(
      "silu_and_mul_dynamic_fp8(Tensor input_tensor, Tensor(a!) output, "
      "Tensor(b!) scales, int group_size) -> ()");
  library.def(
      "silu_and_mul_dynamic_fp8_unchecked(Tensor input_tensor, "
      "Tensor(a!) output, Tensor(b!) scales, int group_size) -> ()");
  library.def(
      "silu_and_mul_per_block_fp8(Tensor(a!) out, Tensor input, "
      "Tensor(b!) scales, int group_size, Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");
  library.def("vllm_silu_and_mul_per_block_fp8_launch_count() -> int",
              &vllm_silu_and_mul_per_block_fp8_launch_count);
  library.def("reset_vllm_silu_and_mul_per_block_fp8_launch_count() -> ()",
              &reset_vllm_silu_and_mul_per_block_fp8_launch_count);
}

TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl("add_rms_norm_mut", &add_rms_norm_mut);
  library.impl("add_rms_norm_mut_unchecked", &launch_add_rms_norm);
  library.impl("rms_norm_dynamic_fp8", &rms_norm_dynamic_fp8);
  library.impl("rms_norm_dynamic_fp8_unchecked",
               &launch_rms_norm_dynamic_fp8);
  library.impl("silu_and_mul", &silu_and_mul);
  library.impl("silu_and_mul_unchecked", &launch_silu_and_mul);
  library.impl("silu_and_mul_dynamic_fp8", &silu_and_mul_dynamic_fp8);
  library.impl("silu_and_mul_dynamic_fp8_unchecked",
               &launch_silu_and_mul_dynamic_fp8);
  library.impl("silu_and_mul_per_block_fp8",
               &vllm_silu_and_mul_per_block_fp8);
}
