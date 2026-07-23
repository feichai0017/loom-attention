#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))

#include "loom_cuda_bridge.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

class CudaDeviceGuard final {
 public:
  explicit CudaDeviceGuard(const torch::stable::Device& device)
      : guard_(device.index()) {}

 private:
  torch::stable::accelerator::DeviceGuard guard_;
};

class CurrentCudaStream final {
 public:
  explicit CurrentCudaStream(int32_t device_index) {
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_));
  }

  void* stream() const {
    return stream_;
  }

 private:
  void* stream_ = nullptr;
};

CurrentCudaStream current_cuda_stream(int32_t device_index) {
  return CurrentCudaStream(device_index);
}

uint64_t tensor_nbytes(const Tensor& tensor) {
  STD_TORCH_CHECK(tensor.numel() >= 0, "negative tensor element count");
  const auto elements = static_cast<uint64_t>(tensor.numel());
  const auto element_size = static_cast<uint64_t>(tensor.element_size());
  STD_TORCH_CHECK(
      element_size == 0 ||
          elements <= std::numeric_limits<uint64_t>::max() / element_size,
      "tensor byte size exceeds uint64");
  return elements * element_size;
}

Tensor new_empty(
    const Tensor& reference,
    std::initializer_list<int64_t> sizes,
    ScalarType dtype) {
  return torch::stable::new_empty(reference, sizes, dtype);
}

uint32_t bridge_dtype(const Tensor& tensor) {
  if (tensor.scalar_type() == ScalarType::Float) {
    return LOOM_CUDA_BRIDGE_F32;
  }
  if (tensor.scalar_type() == ScalarType::Half) {
    return LOOM_CUDA_BRIDGE_F16;
  }
  if (tensor.scalar_type() == ScalarType::BFloat16) {
    return LOOM_CUDA_BRIDGE_BF16;
  }
  STD_TORCH_CHECK(false, "unsupported Loom bridge dtype");
}

uint64_t storage_span_elements(const Tensor& tensor) {
  STD_TORCH_CHECK(tensor.numel() > 0,
              "Loom bridge tensors must contain at least one element");
  uint64_t span = 1;
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    STD_TORCH_CHECK(tensor.size(dimension) > 0 && tensor.stride(dimension) > 0,
                "Loom bridge requires positive tensor sizes and strides");
    const auto extent =
        static_cast<uint64_t>(tensor.size(dimension) - 1);
    const auto stride = static_cast<uint64_t>(tensor.stride(dimension));
    STD_TORCH_CHECK(
        extent == 0 ||
            stride <=
                (std::numeric_limits<uint64_t>::max() - span) / extent,
        "Loom tensor storage span exceeds the bridge ABI");
    span += extent * stride;
  }
  return span;
}

void check_bridge_status(int status, const char* operation) {
  STD_TORCH_CHECK(status == LOOM_CUDA_BRIDGE_SUCCESS, "Loom Rust ", operation,
              " bridge failed: ", loom_cuda_bridge_last_error_message(),
              " (status ", status, ")");
}

bool byte_ranges_overlap(const Tensor& left, const Tensor& right) {
  const auto left_begin =
      reinterpret_cast<uintptr_t>(left.const_data_ptr());
  const auto right_begin =
      reinterpret_cast<uintptr_t>(right.const_data_ptr());
  const auto left_bytes = tensor_nbytes(left);
  const auto right_bytes = tensor_nbytes(right);
  STD_TORCH_CHECK(
      left_bytes <= std::numeric_limits<uintptr_t>::max() - left_begin &&
          right_bytes <= std::numeric_limits<uintptr_t>::max() - right_begin,
      "tensor byte range exceeds uintptr_t");
  const auto left_end = left_begin + left_bytes;
  const auto right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

bool has_dense_nhd_inner_strides(const Tensor& tensor) {
  if (tensor.dim() != 4) {
    return false;
  }
  const int64_t block_elements =
      tensor.size(1) * tensor.size(2) * tensor.size(3);
  return tensor.stride(3) == 1 &&
         tensor.stride(2) == tensor.size(3) &&
         tensor.stride(1) == tensor.size(2) * tensor.size(3) &&
         tensor.stride(0) >= block_elements;
}

void check_rms_norm_contract(const Tensor& input,
                             const Tensor& weight,
                             const Tensor& output, double epsilon) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom RMSNorm input must be CUDA");
  STD_TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device(),
              "Loom RMSNorm tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
                  output.scalar_type() == input.scalar_type(),
              "Loom RMSNorm tensors must have matching dtypes");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom RMSNorm supports F32, FP16, and BF16");
  STD_TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "Loom RMSNorm tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm input must be non-empty");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm weight must match the hidden dimension");
  STD_TORCH_CHECK(output.sizes().equals(input.sizes()),
              "Loom RMSNorm output shape must match input");
  STD_TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm epsilon must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight),
              "Loom RMSNorm output storage must not overlap inputs");
}

void rms_norm(const Tensor& input, const Tensor& weight,
              Tensor output, double epsilon) {
  check_rms_norm_contract(input, weight, output, epsilon);
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm shape exceeds the bridge ABI");
  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const int status = loom_cuda_bridge_rms_norm(
      bridge_dtype(input), input.const_data_ptr(),
      static_cast<uint64_t>(input.numel()), weight.const_data_ptr(),
      static_cast<uint64_t>(weight.numel()), output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()),
      static_cast<uint32_t>(rows_i64),
      static_cast<uint32_t>(hidden_size_i64), static_cast<float>(epsilon),
      stream.stream());
  check_bridge_status(status, "RMSNorm");
}

void check_contract(const Tensor& input, const Tensor& residual,
                    const Tensor& weight, double epsilon) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom Add+RMSNorm input must be CUDA");
  STD_TORCH_CHECK(residual.device() == input.device() &&
                  weight.device() == input.device(),
              "Loom Add+RMSNorm tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(),
              "Loom Add+RMSNorm tensors must have matching dtypes");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom Add+RMSNorm supports F32, FP16, and BF16");
  STD_TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous(),
              "Loom Add+RMSNorm tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom Add+RMSNorm input must be non-empty");
  STD_TORCH_CHECK(input.sizes().equals(residual.sizes()),
              "Loom Add+RMSNorm input/residual shapes must match");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom Add+RMSNorm weight must match the hidden dimension");
  STD_TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom Add+RMSNorm epsilon must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(input, residual) &&
                  !byte_ranges_overlap(input, weight) &&
                  !byte_ranges_overlap(residual, weight),
              "Loom Add+RMSNorm tensor storage ranges must not overlap");
}

void launch_add_rms_norm(Tensor input, Tensor residual,
                         const Tensor& weight, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom Add+RMSNorm shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto input_elements = static_cast<uint64_t>(input.numel());
  const auto residual_elements = static_cast<uint64_t>(residual.numel());
  const auto weight_elements = static_cast<uint64_t>(weight.numel());
  const auto epsilon_f32 = static_cast<float>(epsilon);
  const int status = loom_cuda_bridge_add_rms_norm(
      bridge_dtype(input), input.mutable_data_ptr(), input_elements,
      residual.mutable_data_ptr(), residual_elements, weight.const_data_ptr(),
      weight_elements, rows, hidden_size, epsilon_f32, stream.stream());
  check_bridge_status(status, "Add+RMSNorm");
}

void add_rms_norm_mut(Tensor input, Tensor residual,
                      const Tensor& weight, double epsilon) {
  check_contract(input, residual, weight, epsilon);
  launch_add_rms_norm(input, residual, weight, epsilon);
}

void check_dynamic_fp8_contract(const Tensor& input,
                                const Tensor& weight,
                                const Tensor& output,
                                const Tensor& scales, double epsilon) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom RMSNorm+FP8 input must be CUDA");
  STD_TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom RMSNorm+FP8 tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "Loom RMSNorm+FP8 input and weight dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom RMSNorm+FP8 supports F32, FP16, and BF16 inputs");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Float8_e4m3fn,
              "Loom RMSNorm+FP8 output must use torch.float8_e4m3fn");
  STD_TORCH_CHECK(scales.scalar_type() == ScalarType::Float,
              "Loom RMSNorm+FP8 scales must use F32");
  STD_TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous() && scales.is_contiguous(),
              "Loom RMSNorm+FP8 tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm+FP8 input must be non-empty");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm+FP8 weight must match the hidden dimension");
  STD_TORCH_CHECK(output.sizes().equals(input.sizes()),
              "Loom RMSNorm+FP8 output shape must match input");
  const int64_t rows = input.numel() / input.size(-1);
  STD_TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == 1,
              "Loom RMSNorm+FP8 scales must have shape [rows, 1]");
  STD_TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm+FP8 epsilon must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight) &&
                  !byte_ranges_overlap(output, scales) &&
                  !byte_ranges_overlap(scales, input) &&
                  !byte_ranges_overlap(scales, weight),
              "Loom RMSNorm+FP8 mutable tensor storage must not overlap");
}

void launch_rms_norm_dynamic_fp8(const Tensor& input,
                                 const Tensor& weight, Tensor output,
                                 Tensor scales, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm+FP8 shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto input_elements = static_cast<uint64_t>(input.numel());
  const auto weight_elements = static_cast<uint64_t>(weight.numel());
  const auto output_elements = static_cast<uint64_t>(output.numel());
  const auto scale_elements = static_cast<uint64_t>(scales.numel());
  const auto epsilon_f32 = static_cast<float>(epsilon);
  auto* output_bytes =
      reinterpret_cast<uint8_t*>(output.mutable_data_ptr());
  auto* scale_values = scales.mutable_data_ptr<float>();
  const int status = loom_cuda_bridge_rms_norm_dynamic_fp8(
      bridge_dtype(input), input.const_data_ptr(), input_elements,
      weight.const_data_ptr(), weight_elements, output_bytes, output_elements,
      scale_values, scale_elements, rows, hidden_size, epsilon_f32,
      stream.stream());
  check_bridge_status(status, "RMSNorm+FP8");
}

void rms_norm_dynamic_fp8(const Tensor& input, const Tensor& weight,
                          Tensor output, Tensor scales,
                          double epsilon) {
  check_dynamic_fp8_contract(input, weight, output, scales, epsilon);
  launch_rms_norm_dynamic_fp8(input, weight, output, scales, epsilon);
}

void check_silu_and_mul_contract(const Tensor& input,
                                 const Tensor& output) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul input must be CUDA");
  STD_TORCH_CHECK(output.device() == input.device(),
              "Loom SiLU-and-Mul tensors must be on the same CUDA device");
  STD_TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "Loom SiLU-and-Mul input/output dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Float ||
                  input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom SiLU-and-Mul supports F32, FP16, and BF16");
  STD_TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul input must be non-empty");
  STD_TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul input last dimension must be even");
  STD_TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    STD_TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul output prefix shape must match input");
  }
  STD_TORCH_CHECK(output.size(-1) == input.size(-1) / 2,
              "Loom SiLU-and-Mul output last dimension must be half input");
  STD_TORCH_CHECK(!byte_ranges_overlap(input, output),
              "Loom SiLU-and-Mul input/output storage must not overlap");
}

void launch_silu_and_mul(const Tensor& input, Tensor output) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const int status = loom_cuda_bridge_silu_and_mul(
      bridge_dtype(input), input.const_data_ptr(),
      static_cast<uint64_t>(input.numel()), output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()), rows, width, stream.stream());
  check_bridge_status(status, "SiLU-and-Mul");
}

void silu_and_mul(const Tensor& input, Tensor output) {
  check_silu_and_mul_contract(input, output);
  launch_silu_and_mul(input, output);
}

void check_silu_and_mul_dynamic_fp8_contract(const Tensor& input,
                                              const Tensor& output,
                                              const Tensor& scales,
                                              int64_t group_size,
                                              bool scales_transposed = false) {
  STD_TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul+FP8 input must be CUDA");
  STD_TORCH_CHECK(output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom SiLU-and-Mul+FP8 tensors must be on the same CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Half ||
                  input.scalar_type() == ScalarType::BFloat16,
              "Loom SiLU-and-Mul+FP8 supports FP16 and BF16 input");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Float8_e4m3fn,
              "Loom SiLU-and-Mul+FP8 output must use torch.float8_e4m3fn");
  STD_TORCH_CHECK(scales.scalar_type() == ScalarType::Float,
              "Loom SiLU-and-Mul+FP8 scales must use F32");
  STD_TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul+FP8 input/output must be contiguous");
  STD_TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul+FP8 input must be non-empty");
  STD_TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul+FP8 input last dimension must be even");
  STD_TORCH_CHECK(group_size == 64 || group_size == 128,
              "Loom SiLU-and-Mul+FP8 group size must be 64 or 128");
  const int64_t width = input.size(-1) / 2;
  STD_TORCH_CHECK(width % group_size == 0,
              "Loom SiLU-and-Mul+FP8 width must be divisible by group size");
  STD_TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul+FP8 output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    STD_TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul+FP8 output prefix shape must match input");
  }
  STD_TORCH_CHECK(output.size(-1) == width,
              "Loom SiLU-and-Mul+FP8 output last dimension must be half input");
  const int64_t rows = input.numel() / input.size(-1);
  STD_TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == width / group_size,
              "Loom SiLU-and-Mul+FP8 scales must have shape "
              "[rows, width / group_size]");
  if (scales_transposed) {
    STD_TORCH_CHECK(scales.stride(0) == 1 && scales.stride(1) == rows,
                "Loom transposed FP8 scales must use group-major storage");
  } else {
    STD_TORCH_CHECK(scales.is_contiguous(),
                "Loom row-major FP8 scales must be contiguous");
  }
  STD_TORCH_CHECK(!byte_ranges_overlap(input, output) &&
                  !byte_ranges_overlap(input, scales) &&
                  !byte_ranges_overlap(output, scales),
              "Loom SiLU-and-Mul+FP8 mutable tensor storage must not overlap");
}

void launch_silu_and_mul_dynamic_fp8_layout(
    const Tensor& input, Tensor output, Tensor scales,
    int64_t group_size_i64, const std::optional<Tensor>& scale_ub,
    bool scales_transposed) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  STD_TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max() &&
                  group_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul+FP8 shape exceeds the CUDA ABI");

  const CudaDeviceGuard device_guard(input.device());
  const auto stream = current_cuda_stream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const auto group_size = static_cast<uint32_t>(group_size_i64);
  auto* output_bytes =
      reinterpret_cast<uint8_t*>(output.mutable_data_ptr());
  auto* scale_values = scales.mutable_data_ptr<float>();
  const float* scale_ub_value =
      scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;
  const int status = loom_cuda_bridge_silu_and_mul_dynamic_fp8(
      bridge_dtype(input), input.const_data_ptr(),
      static_cast<uint64_t>(input.numel()), output_bytes,
      static_cast<uint64_t>(output.numel()), scale_values,
      static_cast<uint64_t>(scales.numel()), scale_ub_value,
      scale_ub.has_value() ? static_cast<uint64_t>(scale_ub->numel()) : 0U,
      rows, width, group_size, scales_transposed ? 1U : 0U, stream.stream());
  check_bridge_status(status, "SiLU-and-Mul+FP8");
}

void launch_silu_and_mul_dynamic_fp8(const Tensor& input,
                                      Tensor output, Tensor scales,
                                      int64_t group_size) {
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         std::nullopt, false);
}

void silu_and_mul_dynamic_fp8(const Tensor& input, Tensor output,
                              Tensor scales, int64_t group_size) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size);
  launch_silu_and_mul_dynamic_fp8(input, output, scales, group_size);
}

void vllm_silu_and_mul_per_block_fp8(
    Tensor output, const Tensor& input, Tensor scales,
    int64_t group_size, const std::optional<Tensor>& scale_ub,
    bool scales_transposed) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size,
                                          scales_transposed);
  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(scale_ub->device() == input.device() &&
                    scale_ub->scalar_type() == ScalarType::Float &&
                    scale_ub->numel() == 1 && scale_ub->is_contiguous(),
                "Loom FP8 scale upper bound must be one same-device F32 value");
  }
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         scale_ub, scales_transposed);
}

void check_greedy_sample_logprobs_shape(const Tensor& logits) {
  STD_TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom greedy sampling logits must be non-empty rank-2");
  STD_TORCH_CHECK(logits.size(0) <= std::numeric_limits<uint32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<int32_t>::max(),
              "Loom greedy sampling shape exceeds the CUDA ABI");
}

void check_greedy_sample_logprobs_contract(const Tensor& logits) {
  check_greedy_sample_logprobs_shape(logits);
  STD_TORCH_CHECK(logits.is_cuda(), "Loom greedy sampling logits must be CUDA");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::Float ||
                  logits.scalar_type() == ScalarType::Half ||
                  logits.scalar_type() == ScalarType::BFloat16,
              "Loom greedy sampling supports F32, FP16, and BF16 logits");
  STD_TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
              "Loom greedy sampling logits require unit vocabulary stride "
              "and non-overlapping positive row stride");
}

std::tuple<Tensor, Tensor, Tensor>
launch_greedy_sample_logprobs(const Tensor& logits) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const auto logits_elements = storage_span_elements(logits);
  const auto output_elements = static_cast<uint64_t>(logits.size(0));
  Tensor token_ids = new_empty(logits, {logits.size(0)}, ScalarType::Int);
  Tensor logprobs = new_empty(logits, {logits.size(0)}, ScalarType::Float);
  Tensor ranks = new_empty(logits, {logits.size(0)}, ScalarType::Long);

  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_greedy_sample_logprobs(
      bridge_dtype(logits), logits.const_data_ptr(), logits_elements,
      token_ids.mutable_data_ptr<int32_t>(), output_elements,
      logprobs.mutable_data_ptr<float>(), output_elements,
      ranks.mutable_data_ptr<int64_t>(),
      output_elements, rows, vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "greedy-sampling");
  return {token_ids, logprobs, ranks};
}

std::tuple<Tensor, Tensor, Tensor> greedy_sample_logprobs(
    const Tensor& logits) {
  check_greedy_sample_logprobs_contract(logits);
  return launch_greedy_sample_logprobs(logits);
}

std::tuple<Tensor, Tensor, Tensor> greedy_sample_logprobs_meta(
    const Tensor& logits) {
  check_greedy_sample_logprobs_shape(logits);
  return {
      new_empty(logits, {logits.size(0)}, ScalarType::Int),
      new_empty(logits, {logits.size(0)}, ScalarType::Float),
      new_empty(logits, {logits.size(0)}, ScalarType::Long),
  };
}

void check_selected_token_logprobs_shape(const Tensor& logits,
                                         const Tensor& token_ids) {
  check_greedy_sample_logprobs_shape(logits);
  STD_TORCH_CHECK(token_ids.dim() == 1 && token_ids.size(0) == logits.size(0),
              "Loom selected token IDs must contain one value per logits row");
}

void check_selected_token_logprobs_contract(const Tensor& logits,
                                            const Tensor& token_ids) {
  check_greedy_sample_logprobs_contract(logits);
  check_selected_token_logprobs_shape(logits, token_ids);
  STD_TORCH_CHECK(token_ids.device() == logits.device(),
              "Loom selected token IDs and logits must share a CUDA device");
  STD_TORCH_CHECK(token_ids.scalar_type() == ScalarType::Long,
              "Loom selected token IDs must be int64");
  STD_TORCH_CHECK(token_ids.is_contiguous(),
              "Loom selected token IDs must be contiguous");
}

std::tuple<Tensor, Tensor> launch_selected_token_logprobs(
    const Tensor& logits, const Tensor& token_ids) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  Tensor logprobs = new_empty(logits, {logits.size(0)}, ScalarType::Float);
  Tensor ranks = new_empty(logits, {logits.size(0)}, ScalarType::Long);

  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const auto output_elements = static_cast<uint64_t>(logits.size(0));
  const int status = loom_cuda_bridge_selected_token_logprobs(
      bridge_dtype(logits), logits.const_data_ptr(),
      storage_span_elements(logits), token_ids.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(token_ids.numel()),
      logprobs.mutable_data_ptr<float>(), output_elements,
      ranks.mutable_data_ptr<int64_t>(),
      output_elements, rows, vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "selected-token logprob");
  return {logprobs, ranks};
}

std::tuple<Tensor, Tensor> selected_token_logprobs(
    const Tensor& logits, const Tensor& token_ids) {
  check_selected_token_logprobs_contract(logits, token_ids);
  return launch_selected_token_logprobs(logits, token_ids);
}

std::tuple<Tensor, Tensor> selected_token_logprobs_meta(
    const Tensor& logits, const Tensor& token_ids) {
  check_selected_token_logprobs_shape(logits, token_ids);
  return {
      new_empty(logits, {logits.size(0)}, ScalarType::Float),
      new_empty(logits, {logits.size(0)}, ScalarType::Long),
  };
}

void check_greedy_speculative_verify_shape(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    int64_t max_draft_tokens) {
  STD_TORCH_CHECK(
      draft_token_ids.dim() == 1 && draft_token_ids.numel() > 0,
      "Loom greedy speculative draft token IDs must be non-empty rank-1");
  STD_TORCH_CHECK(
      target_token_ids.dim() == 1 &&
          target_token_ids.numel() == draft_token_ids.numel(),
      "Loom greedy speculative target token IDs must match the flattened "
      "draft shape");
  STD_TORCH_CHECK(
      cumulative_draft_lengths.dim() == 1 &&
          cumulative_draft_lengths.numel() > 0,
      "Loom cumulative draft lengths must be non-empty rank-1");
  const int64_t requests = cumulative_draft_lengths.numel();
  STD_TORCH_CHECK(
      bonus_token_ids.dim() == 2 && bonus_token_ids.size(0) == requests &&
          bonus_token_ids.size(1) == 1,
      "Loom speculative bonus token IDs must have shape [requests, 1]");
  STD_TORCH_CHECK(
      max_draft_tokens > 0 &&
          max_draft_tokens <
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Loom maximum draft length must fit the CUDA ABI");
  STD_TORCH_CHECK(
      requests <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
          draft_token_ids.numel() <=
              static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Loom greedy speculative batch exceeds the CUDA ABI");
  const auto capacity =
      static_cast<uint64_t>(requests) *
      static_cast<uint64_t>(max_draft_tokens);
  STD_TORCH_CHECK(
      static_cast<uint64_t>(draft_token_ids.numel()) <= capacity,
      "Loom flattened draft token count exceeds the ragged batch capacity");
}

void check_greedy_speculative_verify_output_shape(
    const Tensor& output_token_ids, const Tensor& accepted_lengths,
    const Tensor& emitted_lengths, int64_t requests,
    int64_t max_draft_tokens) {
  STD_TORCH_CHECK(
      output_token_ids.dim() == 2 &&
          output_token_ids.size(0) == requests &&
          output_token_ids.size(1) == max_draft_tokens + 1,
      "Loom speculative output token IDs must have shape "
      "[requests, max_draft_tokens + 1]");
  STD_TORCH_CHECK(
      accepted_lengths.dim() == 1 &&
          accepted_lengths.numel() == requests &&
          emitted_lengths.dim() == 1 &&
          emitted_lengths.numel() == requests,
      "Loom speculative accepted and emitted lengths must have shape "
      "[requests]");
}

void check_greedy_speculative_verify_contract(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    const Tensor& output_token_ids, const Tensor& accepted_lengths,
    const Tensor& emitted_lengths, int64_t max_draft_tokens) {
  check_greedy_speculative_verify_shape(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, max_draft_tokens);
  const int64_t requests = cumulative_draft_lengths.numel();
  check_greedy_speculative_verify_output_shape(
      output_token_ids, accepted_lengths, emitted_lengths, requests,
      max_draft_tokens);
  STD_TORCH_CHECK(draft_token_ids.is_cuda(),
                  "Loom greedy speculative inputs must be CUDA");
  STD_TORCH_CHECK(
      target_token_ids.device() == draft_token_ids.device() &&
          bonus_token_ids.device() == draft_token_ids.device() &&
          cumulative_draft_lengths.device() == draft_token_ids.device() &&
          output_token_ids.device() == draft_token_ids.device() &&
          accepted_lengths.device() == draft_token_ids.device() &&
          emitted_lengths.device() == draft_token_ids.device(),
      "Loom greedy speculative tensors must share one CUDA device");
  STD_TORCH_CHECK(
      draft_token_ids.scalar_type() == ScalarType::Int &&
          bonus_token_ids.scalar_type() == ScalarType::Int &&
          cumulative_draft_lengths.scalar_type() == ScalarType::Int &&
          output_token_ids.scalar_type() == ScalarType::Int &&
          accepted_lengths.scalar_type() == ScalarType::Int &&
          emitted_lengths.scalar_type() == ScalarType::Int,
      "Loom draft, bonus, cumulative, and output tensors must use int32");
  STD_TORCH_CHECK(target_token_ids.scalar_type() == ScalarType::Long,
                  "Loom target token IDs must use int64");
  STD_TORCH_CHECK(
      draft_token_ids.is_contiguous() && target_token_ids.is_contiguous() &&
          bonus_token_ids.is_contiguous() &&
          cumulative_draft_lengths.is_contiguous() &&
          output_token_ids.is_contiguous() &&
          accepted_lengths.is_contiguous() &&
          emitted_lengths.is_contiguous(),
      "Loom greedy speculative tensors must be contiguous");
}

void launch_greedy_speculative_verify(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    Tensor output_token_ids, Tensor accepted_lengths, Tensor emitted_lengths,
    int64_t max_draft_tokens) {
  const int64_t requests = cumulative_draft_lengths.numel();
  const CudaDeviceGuard device_guard(draft_token_ids.device());
  const auto stream =
      current_cuda_stream(draft_token_ids.device().index());
  const int status = loom_cuda_bridge_greedy_speculative_verify(
      draft_token_ids.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(draft_token_ids.numel()),
      target_token_ids.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(target_token_ids.numel()),
      bonus_token_ids.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(bonus_token_ids.numel()),
      cumulative_draft_lengths.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(cumulative_draft_lengths.numel()),
      output_token_ids.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(output_token_ids.numel()),
      accepted_lengths.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(accepted_lengths.numel()),
      emitted_lengths.mutable_data_ptr<int32_t>(),
      static_cast<uint64_t>(emitted_lengths.numel()),
      static_cast<uint32_t>(requests),
      static_cast<uint32_t>(draft_token_ids.numel()),
      static_cast<uint32_t>(max_draft_tokens), stream.stream());
  check_bridge_status(status, "greedy speculative verification");
}

void greedy_speculative_verify(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    Tensor output_token_ids, Tensor accepted_lengths, Tensor emitted_lengths,
    int64_t max_draft_tokens) {
  check_greedy_speculative_verify_contract(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, output_token_ids, accepted_lengths,
      emitted_lengths, max_draft_tokens);
  launch_greedy_speculative_verify(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, output_token_ids, accepted_lengths,
      emitted_lengths, max_draft_tokens);
}

void greedy_speculative_verify_meta(
    const Tensor& draft_token_ids, const Tensor& target_token_ids,
    const Tensor& bonus_token_ids, const Tensor& cumulative_draft_lengths,
    Tensor output_token_ids, Tensor accepted_lengths, Tensor emitted_lengths,
    int64_t max_draft_tokens) {
  check_greedy_speculative_verify_shape(
      draft_token_ids, target_token_ids, bonus_token_ids,
      cumulative_draft_lengths, max_draft_tokens);
  check_greedy_speculative_verify_output_shape(
      output_token_ids, accepted_lengths, emitted_lengths,
      cumulative_draft_lengths.numel(), max_draft_tokens);
}

void check_min_p_filter_shape(const Tensor& logits,
                              const Tensor& min_p) {
  STD_TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom min-p logits must be non-empty rank-2");
  STD_TORCH_CHECK(logits.size(0) <= std::numeric_limits<uint32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<uint32_t>::max(),
              "Loom min-p shape exceeds the CUDA ABI");
  STD_TORCH_CHECK((min_p.dim() == 1 && min_p.size(0) == logits.size(0)) ||
                  (min_p.dim() == 2 && min_p.size(0) == logits.size(0) &&
                   min_p.size(1) == 1),
              "Loom min-p probabilities must have shape [rows] or [rows, 1]");
}

void check_min_p_filter_contract(const Tensor& logits,
                                 const Tensor& min_p) {
  check_min_p_filter_shape(logits, min_p);
  STD_TORCH_CHECK(logits.is_cuda(), "Loom min-p logits must be CUDA");
  STD_TORCH_CHECK(min_p.device() == logits.device(),
              "Loom min-p probabilities and logits must share a CUDA device");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::Float ||
                  logits.scalar_type() == ScalarType::Half ||
                  logits.scalar_type() == ScalarType::BFloat16,
              "Loom min-p supports F32, FP16, and BF16 logits");
  STD_TORCH_CHECK(min_p.scalar_type() == ScalarType::Float,
              "Loom min-p probabilities must use F32");
  STD_TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
              "Loom min-p logits require unit vocabulary stride and "
              "non-overlapping positive row stride");
  STD_TORCH_CHECK(min_p.is_contiguous(),
              "Loom min-p probabilities must be contiguous");
  STD_TORCH_CHECK(!byte_ranges_overlap(logits, min_p),
              "Loom min-p logits and probabilities must not overlap");
}

void launch_min_p_filter(Tensor logits, const Tensor& min_p) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const CudaDeviceGuard device_guard(logits.device());
  const auto stream = current_cuda_stream(logits.device().index());
  const int status = loom_cuda_bridge_min_p_filter(
      bridge_dtype(logits), logits.mutable_data_ptr(),
      storage_span_elements(logits), min_p.const_data_ptr<float>(),
      static_cast<uint64_t>(min_p.numel()), rows,
      vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "min-p");
}

void min_p_filter(Tensor logits, const Tensor& min_p) {
  check_min_p_filter_contract(logits, min_p);
  launch_min_p_filter(logits, min_p);
}

void check_rope_paged_kv_write_contract(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    const Tensor& key_cache, const Tensor& value_cache,
    const Tensor& slot_mapping) {
  STD_TORCH_CHECK(query.is_cuda(), "Loom RoPE+paged-KV query must be CUDA");
  STD_TORCH_CHECK(key.device() == query.device() &&
                  value.device() == query.device() &&
                  positions.device() == query.device() &&
                  cos_sin_cache.device() == query.device() &&
                  key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  slot_mapping.device() == query.device(),
              "Loom RoPE+paged-KV tensors must be on one CUDA device");
  STD_TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == cos_sin_cache.scalar_type() &&
                  query.scalar_type() == key_cache.scalar_type() &&
                  query.scalar_type() == value_cache.scalar_type(),
              "Loom RoPE+paged-KV data and native caches must share a dtype");
  STD_TORCH_CHECK(query.scalar_type() == ScalarType::Float ||
                  query.scalar_type() == ScalarType::Half ||
                  query.scalar_type() == ScalarType::BFloat16,
              "Loom RoPE+paged-KV supports F32, FP16, and BF16 native caches");
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Long &&
                  slot_mapping.scalar_type() == ScalarType::Long,
              "Loom RoPE+paged-KV positions and slot mapping must be int64");
  STD_TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "Loom RoPE+paged-KV Q/K/V must have rank 3");
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
  STD_TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "Loom paged K/V cache views must have rank 4");
  STD_TORCH_CHECK(key_cache.size(0) > 0 && key_cache.size(1) > 0 &&
                  key_cache.size(2) == key.size(1) &&
                  key_cache.size(3) == key.size(2),
              "Loom key cache must have logical shape "
              "[blocks, block_size, kv_heads, head_size]");
  STD_TORCH_CHECK(value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == key_cache.size(1) &&
                  value_cache.size(2) == value.size(1) &&
                  value_cache.size(3) == value.size(2),
              "Loom value cache must have logical shape "
              "[blocks, block_size, kv_heads, value_head_size]");
  STD_TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_cache.stride(0) > 0 && key_cache.stride(1) > 0 &&
                  key_cache.stride(2) > 0 && value_cache.stride(0) > 0 &&
                  value_cache.stride(1) > 0 && value_cache.stride(2) > 0,
              "Loom paged K/V caches require unit dim stride and positive "
              "block/page/head strides");
}

void launch_rope_paged_kv_write(
    Tensor query, Tensor key, const Tensor& value,
    const Tensor& positions, const Tensor& cos_sin_cache,
    Tensor key_cache, Tensor value_cache,
    const Tensor& slot_mapping, bool is_neox) {
  const int64_t limits[] = {
      query.size(0),       query.size(1),       key.size(1),
      query.size(2),       value.size(2),       cos_sin_cache.size(1),
      cos_sin_cache.size(0), key_cache.size(0), key_cache.size(1),
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
  const auto num_blocks = static_cast<uint32_t>(key_cache.size(0));
  const auto block_size = static_cast<uint32_t>(key_cache.size(1));
  const auto query_token_stride = static_cast<uint64_t>(query.stride(0));
  const auto query_head_stride = static_cast<uint64_t>(query.stride(1));
  const auto key_token_stride = static_cast<uint64_t>(key.stride(0));
  const auto key_head_stride = static_cast<uint64_t>(key.stride(1));
  const auto value_token_stride = static_cast<uint64_t>(value.stride(0));
  const auto value_head_stride = static_cast<uint64_t>(value.stride(1));
  const auto key_block_stride = static_cast<uint64_t>(key_cache.stride(0));
  const auto key_page_stride = static_cast<uint64_t>(key_cache.stride(1));
  const auto key_cache_head_stride =
      static_cast<uint64_t>(key_cache.stride(2));
  const auto value_block_stride =
      static_cast<uint64_t>(value_cache.stride(0));
  const auto value_page_stride =
      static_cast<uint64_t>(value_cache.stride(1));
  const auto value_cache_head_stride =
      static_cast<uint64_t>(value_cache.stride(2));

  const int status = loom_cuda_bridge_rope_paged_kv_write(
      bridge_dtype(query), query.mutable_data_ptr(),
      storage_span_elements(query), key.mutable_data_ptr(),
      storage_span_elements(key), value.const_data_ptr(),
      storage_span_elements(value), positions.const_data_ptr<int64_t>(),
      static_cast<uint64_t>(positions.numel()),
      cos_sin_cache.const_data_ptr(),
      static_cast<uint64_t>(cos_sin_cache.numel()),
      key_cache.mutable_data_ptr(), storage_span_elements(key_cache),
      value_cache.mutable_data_ptr(), storage_span_elements(value_cache),
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
    Tensor key_cache, Tensor value_cache,
    const Tensor& slot_mapping, bool is_neox) {
  check_rope_paged_kv_write_contract(query, key, value, positions,
                                     cos_sin_cache, key_cache, value_cache,
                                     slot_mapping);
  launch_rope_paged_kv_write(query, key, value, positions, cos_sin_cache,
                             key_cache, value_cache, slot_mapping, is_neox);
}

void check_paged_decode_attention_contract(
    const Tensor& query, const Tensor& key_cache,
    const Tensor& value_cache, const Tensor& block_tables,
    const Tensor& sequence_lengths, const Tensor& output,
    int64_t max_sequence_length, double scale) {
  STD_TORCH_CHECK(query.is_cuda(), "Loom paged decode query must be CUDA");
  STD_TORCH_CHECK(key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  block_tables.device() == query.device() &&
                  sequence_lengths.device() == query.device() &&
                  output.device() == query.device(),
              "Loom paged decode tensors must be on one CUDA device");
  STD_TORCH_CHECK(query.scalar_type() == ScalarType::Float ||
                  query.scalar_type() == ScalarType::Half ||
                  query.scalar_type() == ScalarType::BFloat16,
              "Loom paged decode supports F32, FP16, and BF16 native caches");
  STD_TORCH_CHECK(key_cache.scalar_type() == query.scalar_type() &&
                  value_cache.scalar_type() == query.scalar_type() &&
                  output.scalar_type() == query.scalar_type(),
              "Loom paged decode data tensors must share a dtype");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int &&
                  sequence_lengths.scalar_type() == ScalarType::Int,
              "Loom paged decode metadata must use int32");
  STD_TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 4 &&
                  value_cache.dim() == 4 && block_tables.dim() == 2 &&
                  sequence_lengths.dim() == 1 && output.dim() == 3,
              "Loom paged decode requires rank-3 query/output, rank-4 K/V "
              "caches, rank-2 block tables, and rank-1 sequence lengths");
  STD_TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
                  key_cache.size(0) > 0 && key_cache.size(1) > 0 &&
                  key_cache.size(2) > 0 && value_cache.size(3) > 0,
              "Loom paged decode dimensions must be positive");
  STD_TORCH_CHECK(key_cache.size(3) == query.size(2),
              "Loom paged decode Q/K head sizes must match");
  STD_TORCH_CHECK(value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == key_cache.size(1) &&
                  value_cache.size(2) == key_cache.size(2),
              "Loom paged decode K/V cache prefixes must match");
  STD_TORCH_CHECK(query.size(1) % key_cache.size(2) == 0,
              "Loom paged decode query heads must be divisible by KV heads");
  STD_TORCH_CHECK(block_tables.size(0) == query.size(0) &&
                  block_tables.size(1) > 0 &&
                  sequence_lengths.size(0) == query.size(0),
              "Loom paged decode metadata batch dimensions must match query");
  STD_TORCH_CHECK(output.size(0) == query.size(0) &&
                  output.size(1) == query.size(1) &&
                  output.size(2) == value_cache.size(3),
              "Loom paged decode output must have shape [B, Hq, Dv]");
  STD_TORCH_CHECK(query.is_contiguous() &&
                  has_dense_nhd_inner_strides(key_cache) &&
                  has_dense_nhd_inner_strides(value_cache) &&
                  block_tables.is_contiguous() &&
                  sequence_lengths.is_contiguous() && output.is_contiguous(),
              "Loom paged decode requires contiguous query/output/metadata "
              "and dense-inner NHD caches with an optional block stride");
  STD_TORCH_CHECK(max_sequence_length > 0 && max_sequence_length <= 1024 &&
                  max_sequence_length <=
                      block_tables.size(1) * key_cache.size(1),
              "Loom paged decode max_sequence_length must be within table "
              "capacity and the first-kernel limit 1024");
  STD_TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "Loom paged decode scale must be finite and positive");
  STD_TORCH_CHECK(!byte_ranges_overlap(output, query) &&
                  !byte_ranges_overlap(output, key_cache) &&
                  !byte_ranges_overlap(output, value_cache) &&
                  !byte_ranges_overlap(output, block_tables) &&
                  !byte_ranges_overlap(output, sequence_lengths),
              "Loom paged decode output storage must not overlap inputs");

  const int64_t limits[] = {
      query.size(0),       query.size(1),      key_cache.size(2),
      query.size(2),       value_cache.size(3), key_cache.size(0),
      key_cache.size(1),   block_tables.size(1), max_sequence_length,
  };
  for (const int64_t value_to_check : limits) {
    STD_TORCH_CHECK(value_to_check > 0 &&
                    value_to_check <= std::numeric_limits<uint32_t>::max(),
                "Loom paged decode shape exceeds the CUDA ABI");
  }
  STD_TORCH_CHECK(query.size(0) <=
                  std::numeric_limits<int32_t>::max() / query.size(1),
              "Loom paged decode grid exceeds the CUDA ABI");
}

void launch_paged_decode_attention(
    const Tensor& query, const Tensor& key_cache,
    const Tensor& value_cache, const Tensor& block_tables,
    const Tensor& sequence_lengths, Tensor output,
    int64_t max_sequence_length, double scale) {
  const auto sequences = static_cast<uint32_t>(query.size(0));
  const auto query_heads = static_cast<uint32_t>(query.size(1));
  const auto kv_heads = static_cast<uint32_t>(key_cache.size(2));
  const auto head_size = static_cast<uint32_t>(query.size(2));
  const auto value_head_size = static_cast<uint32_t>(value_cache.size(3));
  const auto num_blocks = static_cast<uint32_t>(key_cache.size(0));
  const auto block_size = static_cast<uint32_t>(key_cache.size(1));
  const auto key_block_stride =
      static_cast<uint64_t>(key_cache.stride(0));
  const auto value_block_stride =
      static_cast<uint64_t>(value_cache.stride(0));
  const auto max_blocks_per_sequence =
      static_cast<uint32_t>(block_tables.size(1));
  const auto max_context = static_cast<uint32_t>(max_sequence_length);
  const auto scale_f32 = static_cast<float>(scale);
  const CudaDeviceGuard device_guard(query.device());
  const auto stream = current_cuda_stream(query.device().index());
  uint64_t split_k_workspace_elements = 0;
  int status = loom_cuda_bridge_paged_decode_workspace_elements(
      bridge_dtype(query), sequences, query_heads, kv_heads, head_size,
      value_head_size, num_blocks, block_size, max_blocks_per_sequence,
      max_context, scale_f32, &split_k_workspace_elements);
  check_bridge_status(status, "paged decode workspace query");
  STD_TORCH_CHECK(split_k_workspace_elements <=
                  static_cast<uint64_t>(
                      std::numeric_limits<int64_t>::max()),
              "Loom paged decode split-K workspace exceeds PyTorch limits");
  Tensor split_k_workspace;
  if (split_k_workspace_elements != 0U) {
    split_k_workspace = new_empty(
        query,
        {static_cast<int64_t>(split_k_workspace_elements)},
        ScalarType::Float);
  }
  float* split_k_workspace_pointer =
      split_k_workspace.defined()
          ? split_k_workspace.mutable_data_ptr<float>()
                                  : nullptr;

  status = loom_cuda_bridge_paged_decode_attention(
      bridge_dtype(query), query.const_data_ptr(),
      static_cast<uint64_t>(query.numel()), key_cache.const_data_ptr(),
      storage_span_elements(key_cache), value_cache.const_data_ptr(),
      storage_span_elements(value_cache),
      block_tables.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(block_tables.numel()),
      sequence_lengths.const_data_ptr<int32_t>(),
      static_cast<uint64_t>(sequence_lengths.numel()),
      output.mutable_data_ptr(),
      static_cast<uint64_t>(output.numel()), split_k_workspace_pointer,
      split_k_workspace_elements, sequences, query_heads, kv_heads, head_size,
      value_head_size, num_blocks, block_size, key_block_stride,
      value_block_stride, max_blocks_per_sequence, max_context, scale_f32,
      stream.stream());
  check_bridge_status(status, "paged decode attention");
}

void paged_decode_attention(
    const Tensor& query, const Tensor& key_cache,
    const Tensor& value_cache, const Tensor& block_tables,
    const Tensor& sequence_lengths, Tensor output,
    int64_t max_sequence_length, double scale) {
  check_paged_decode_attention_contract(
      query, key_cache, value_cache, block_tables, sequence_lengths, output,
      max_sequence_length, scale);
  launch_paged_decode_attention(query, key_cache, value_cache, block_tables,
                                sequence_lengths, output,
                                max_sequence_length, scale);
}

int64_t bridge_abi_version() {
  return static_cast<int64_t>(loom_cuda_bridge_abi_version());
}

int64_t bridge_launch_count(int64_t operation) {
  STD_TORCH_CHECK(operation >= 0 &&
                  operation <= LOOM_CUDA_BRIDGE_GREEDY_SPECULATIVE_VERIFY,
              "Loom bridge operator id is out of range");
  uint64_t count = 0;
  const int status = loom_cuda_bridge_launch_count(
      static_cast<uint32_t>(operation), &count);
  check_bridge_status(status, "telemetry query");
  STD_TORCH_CHECK(
      count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "Loom bridge launch count exceeds int64");
  return static_cast<int64_t>(count);
}

void reset_bridge_launch_count(int64_t operation) {
  STD_TORCH_CHECK(operation >= 0 &&
                  operation <= LOOM_CUDA_BRIDGE_GREEDY_SPECULATIVE_VERIFY,
              "Loom bridge operator id is out of range");
  const int status =
      loom_cuda_bridge_reset_launch_count(static_cast<uint32_t>(operation));
  check_bridge_status(status, "telemetry reset");
}

}  // namespace

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
      "Tensor positions, Tensor cos_sin_cache, Tensor(c!) key_cache, "
      "Tensor(d!) value_cache, Tensor slot_mapping, bool is_neox) -> ()");
  library.def("bridge_abi_version() -> int");
  library.def("bridge_launch_count(int operation) -> int");
  library.def("reset_bridge_launch_count(int operation) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl("rms_norm", TORCH_BOX(&rms_norm));
  library.impl("add_rms_norm_mut", TORCH_BOX(&add_rms_norm_mut));
  library.impl("rms_norm_dynamic_fp8", TORCH_BOX(&rms_norm_dynamic_fp8));
  library.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  library.impl(
      "silu_and_mul_dynamic_fp8",
      TORCH_BOX(&silu_and_mul_dynamic_fp8));
  library.impl("silu_and_mul_per_block_fp8",
               TORCH_BOX(&vllm_silu_and_mul_per_block_fp8));
  library.impl(
      "greedy_sample_logprobs", TORCH_BOX(&greedy_sample_logprobs));
  library.impl(
      "selected_token_logprobs", TORCH_BOX(&selected_token_logprobs));
  library.impl(
      "greedy_speculative_verify",
      TORCH_BOX(&greedy_speculative_verify));
  library.impl("min_p_filter_", TORCH_BOX(&min_p_filter));
  library.impl(
      "paged_decode_attention", TORCH_BOX(&paged_decode_attention));
  library.impl("rope_paged_kv_write_", TORCH_BOX(&rope_paged_kv_write));
}

STABLE_TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl(
      "greedy_sample_logprobs", TORCH_BOX(&greedy_sample_logprobs_meta));
  library.impl(
      "selected_token_logprobs",
      TORCH_BOX(&selected_token_logprobs_meta));
  library.impl(
      "greedy_speculative_verify",
      TORCH_BOX(&greedy_speculative_verify_meta));
}

STABLE_TORCH_LIBRARY_IMPL(
    loom_kernels, CompositeExplicitAutograd, library) {
  library.impl("bridge_abi_version", TORCH_BOX(&bridge_abi_version));
  library.impl("bridge_launch_count", TORCH_BOX(&bridge_launch_count));
  library.impl(
      "reset_bridge_launch_count", TORCH_BOX(&reset_bridge_launch_count));
}
