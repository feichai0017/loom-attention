# vLLM IR Provider

Loom Kernels can replace vLLM 0.24's fused residual Add+RMSNorm implementation
through the vLLM IR provider registry. The integration is inference-only,
mutates both tensors in place, launches on PyTorch's current CUDA stream, and
survives vLLM compilation and CUDA Graph capture.

The same package also provides an opt-in out-of-tree replacement for vLLM's
standard `SiluAndMul` layer. It is not enabled merely by installing the plugin:
the current H20 result establishes exact compatibility and graph parity, not a
performance win.

A second opt-in replaces vLLM 0.24's fused SiLU-and-Mul plus dynamic symmetric
per-block FP8 implementations for group sizes 64 and 128. This boundary is
bitwise compatible with vLLM's fused operator and has an operator-level H20
advantage. It has also completed a pinned Qwen2.5 online-FP8 engine gate with
direct compiler-match and launch evidence; that small-model end-to-end result
is at parity rather than a demonstrated speedup.

The registered contract is:

```text
residual = input + residual
input = RMSNorm(residual, weight, epsilon)
```

## Build

Use an isolated Python environment with a CUDA-enabled PyTorch and vLLM:

```bash
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -e 'python[vllm,test]'

CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  .venv-vllm/bin/python python/build_native.py

CUDA_HOME=/usr/local/cuda \
  .venv-vllm/bin/python python/build_torch_extension.py
```

The first command builds `build/libloom_kernels_cuda.so` from the same CUDA
sources used by the Rust backend. The second builds a small C++ dispatcher shim
at `build/libloom_kernels_torch.so`; this avoids Python/ctypes overhead on the
vLLM hot path. Repository checkouts discover both files automatically. A
packaged deployment can instead set `LOOM_KERNELS_CUDA_LIBRARY` and
`LOOM_KERNELS_TORCH_LIBRARY` to absolute library paths.

## Direct PyTorch Use

```python
from loom_kernels.torch_ops import add_rms_norm_

output, updated_residual = add_rms_norm_(
    input_tensor,
    residual,
    weight,
    1.0e-5,
)

from loom_kernels import (
    silu_and_mul,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_fp8_out,
    silu_and_mul_out,
)

output = silu_and_mul(gate_and_up)
silu_and_mul_out(gate_and_up, reusable_output)

fp8_output, block_scales = silu_and_mul_dynamic_fp8(
    gate_and_up_bf16,
    group_size=128,
)
silu_and_mul_dynamic_fp8_out(
    gate_and_up_bf16,
    reusable_fp8_output,
    reusable_block_scales,
    group_size=128,
)
```

Add+RMSNorm and standalone SiLU-and-Mul tensors must be contiguous CUDA tensors
using their documented matching F32, FP16, or BF16 dtype. The dynamic-block-FP8
path accepts FP16/BF16 input, group size 64 or 128, and a width divisible by the
group. `weight` must be one-dimensional and match the final normalization
dimension. Checked public operators reject gradients and aliasing.

## vLLM Use

Installing the Python package exposes a `vllm.general_plugins` entry point.
Select Loom for only the supported IR operation:

```python
from vllm import LLM

engine = LLM(
    model="/path/to/model",
    ir_op_priority={"fused_add_rms_norm": ["loom_cuda"]},
)
```

vLLM appends its native fallback to the priority list. Loom declines tensors
outside its contiguous same-dtype contract, weighted RMSNorm calls without a
normal variance size, and unsupported devices.

To replace vLLM's standard SwiGLU layer as well, opt in before the engine
process starts:

```bash
LOOM_KERNELS_ENABLE_SILU_AND_MUL=1 python your_vllm_service.py
```

Python embedding code can instead call
`loom_kernels.vllm.register_vllm_silu_and_mul()` explicitly before constructing
the model. The replacement supports contiguous CUDA F32/FP16/BF16 input with
an even final dimension and preserves vLLM's output dtype and rounding.

To replace the activation-quant fusion table entries for dynamic symmetric FP8
groups 64 and 128, enable the separate opt-in before vLLM imports its model:

```bash
LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8=1 python your_vllm_service.py
```

Embedding code can call
`loom_kernels.vllm.register_vllm_silu_and_mul_dynamic_fp8()` explicitly. The
replacement uses vLLM's mutable custom-op schema, including an optional F32
scale upper bound and row-major or transposed scale storage. Registration is
intentionally version-specific to vLLM 0.24's activation-quant compiler pass;
unsupported versions should leave the opt-in unset.

The provider can only replace a graph-visible activation-quant boundary. On
the tested H20 stack, vLLM's automatic `fp8_per_block` selection uses a
FlashInfer/DeepGEMM linear kernel that accepts BF16 and performs activation
quantization inside GEMM. That path contains no separate node for Loom to
replace. The engine A/B therefore fixes `linear_backend="cutlass"`, enables
the `quant_fp8` custom op, and enables `fuse_act_quant` for both providers. The
GEMM is identical on both sides; only the fused activation-quant operator
changes.

To verify selection without starting an engine:

```bash
.venv-vllm/bin/python - <<'PY'
from loom_kernels.vllm import provider_metadata, register_vllm_ir

register_vllm_ir()
print(provider_metadata())
PY
```

## Validation

```bash
.venv-vllm/bin/pytest -q python/tests

.venv-vllm/bin/python benchmarks/vllm_ir_add_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_engine_add_rms_norm.py \
  --model build/synthetic-qwen2-h4096-l4 \
  --provider loom_cuda --batch-size 8 \
  --input-len 128 --output-len 128

.venv-vllm/bin/python benchmarks/vllm_silu_and_mul.py \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_silu_and_mul_dynamic_fp8.py \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 2000 --samples 15 \
  --provider-order forward

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/qwen25-fp8-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --provider-order loom-first \
  --result-json /tmp/qwen25-fp8-loom-first.json
```

The microbenchmark compares `loom_cuda` and `vllm_c` through the same vLLM IR
eager dispatcher and CUDA Graph replay. It warms the GPU before each provider
to avoid clock-state order bias. The engine benchmark uses the normal Qwen2
model runner, compilation, scheduler, KV cache, and graph capture, but the
repository's generated checkpoint contains random weights and is not a
pretrained model.

On H20 with vLLM 0.24.0, Loom and `vllm_c` were bitwise identical for the
tested BF16 shapes. Both differ slightly from vLLM IR's FP32-add formal
reference because the CUDA path materializes the BF16 residual before its RMS
reduction. See the
[H20 integration report](../results/h20-vllm-ir-add-rms-norm-20260721.json).

For SiLU-and-Mul, F32/FP16/BF16 and odd-width fallback tests were bitwise equal
to vLLM. Order-reversed CUDA Graph medians were within 0.1%, while eager
dispatch was sensitive to run order. The synthetic Qwen2 engine completed
compilation, graph capture, and generation with the opt-in replacement. See
the [H20 SiLU-and-Mul report](../results/h20-silu-and-mul-20260721.json).

For SiLU-and-Mul+block-FP8, Loom was bitwise identical to vLLM's fused
operator for both supported input dtypes and group sizes. On BF16 `8x11008`,
order-reversed runs showed `1.216-1.231x` eager speedup ratios
(`17.7-18.8%` lower latency) and `1.037-1.082x` CUDA Graph ratios
(`3.6-7.5%` lower latency). The composed vLLM SiLU-then-quantize path is slower
but rounds an intermediate BF16 tensor, so it is not the exact correctness
baseline. See the
[H20 fused activation-quant report](../results/h20-silu-and-mul-dynamic-fp8-20260721.json).

The real-model gate pins Qwen2.5-0.5B-Instruct, online-quantizes it with
vLLM's `fp8_per_block` mode, and runs each provider in a fresh process with an
isolated compile cache. Both provider orders matched every generated token,
each compiler graph recorded two activation-quant matches, and the launch
probe recorded 1584 Loom submissions only in the Loom process. Across the
three cases, batch-latency ratios ranged from `0.9991x` to `1.0043x`, so this
is integration and correctness evidence rather than a model-level performance
claim. See the
[H20 Qwen2.5 engine report](../results/h20-vllm-qwen25-05b-fp8-engine-20260722.json).

The compatible arithmetic and schema follow vLLM 0.24's
[fused CUDA implementation](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu)
and its documented
[fusion mechanism](https://docs.vllm.ai/en/v0.23.0/design/fusions/).

The provider API follows vLLM's
[IR design](https://docs.vllm.ai/en/v0.22.1/design/vllm_ir/) and the mutable
dispatcher bridge follows PyTorch's
[custom-operator contract](https://docs.pytorch.org/docs/stable/library.html).

## Current Limits

- Linux and CUDA only;
- source/editable deployment; an automated binary-wheel build is not provided;
- inference-only mutation, with no autograd implementation;
- one selectable IR provider (`fused_add_rms_norm`), one opt-in out-of-tree
  layer replacement (`SiluAndMul`), and one vLLM-version-specific
  activation-quant fusion-table replacement;
- the activation-quant provider requires a graph-visible quantization boundary;
  it does not intercept vLLM's fused BF16-input FlashInfer/DeepGEMM path;
- the isolated operator is faster on H20 and real-model invocation is proven,
  but no model-level speedup has been established.
