# Loom Kernels

[Website](https://feichai0017.github.io/loom-kernels/) ·
[Operator catalog](https://feichai0017.github.io/loom-kernels/docs/operators/) ·
[H20 evidence](https://feichai0017.github.io/loom-kernels/benchmarks/)

Loom Kernels is a Rust-first high-performance operator library for LLM
inference. It provides backend-independent contracts and CPU references,
checked accelerator dispatch, handwritten CUDA kernels, and reproducible
correctness/performance gates.

The project is intentionally not an inference engine or tensor framework. It
targets the small set of decode-critical operators where fusion, layout-aware
execution, or lower dispatch overhead can create measurable engine value.

## Current Status

- `loom-kernels`: dtype, tensor, capability, normalization, quantization, and
  split-half SiLU-and-Mul contracts plus CPU oracles;
- `loom-cuda`: safe CUDA stream/buffer/event ownership and checked dispatch;
- `loom-cuda-sys`: dependency-light raw C ABI;
- handwritten F32 plus vectorized FP16/BF16 RMSNorm, fused Add+RMSNorm,
  dynamic per-token FP8 E4M3FN output quantization, SiLU-and-Mul, and fused
  SiLU-and-Mul+dynamic per-block FP8 validated on NVIDIA H20;
- PyTorch current-stream custom operators and a vLLM 0.24 Add+RMSNorm IR
  provider validated through compilation, CUDA Graph capture, and a real Qwen2
  engine generate loop;
- an opt-in vLLM `SiluAndMul` layer replacement is bitwise compatible and
  engine-valid, but graph latency is at parity, so no speedup is claimed;
- an opt-in vLLM 0.24 activation-quant fusion replacement covers dynamic
  symmetric FP8 groups 64/128 and is bitwise compatible with vLLM's fused
  kernel; pinned Qwen2.5-0.5B online-FP8 runs now prove compiler path hits,
  CUDA Graph execution, and exact generation parity, while end-to-end latency
  remains at parity;
- RMSNorm+FP8 is bitwise compatible with vLLM's named CUDA baseline; routing it
  through a real engine graph is the remaining integration gate.

## Workspace

| Path | Responsibility |
| --- | --- |
| `crates/loom-kernels` | public contracts, capability queries, and CPU references |
| `crates/loom-cuda` | safe Rust CUDA backend and benchmarks |
| `crates/loom-cuda-sys` | raw CUDA C ABI and build plumbing |
| `cuda` | handwritten CUDA kernels |
| `python` | PyTorch dispatcher bridge and vLLM IR provider |
| `benchmarks` | named external baselines |
| `docs/results` | hardware-qualified validation artifacts |
| `website` | Astro documentation and project site |

## Operator Priorities

| Priority | Operator family | Why it matters |
| --- | --- | --- |
| P0 | RMSNorm, Add+RMSNorm, Norm+Quant | memory-bound decode kernels with useful fusion boundaries |
| P0 | RoPE+KV write, KV append/layout/quantization | removes extra HBM passes around KV-cache updates |
| P0 | SwiGLU/GELU fused epilogues | combines activation, multiply, bias, and quantization |
| P0 | sampling and selected-token logprob | reduces decode-tail launches and temporary tensors |
| P1 | paged decode attention | important, but only after the common backend is stable |
| P1 | MoE top-k, permutation, grouped dispatch | routing and movement often dominate small expert batches |
| P1 | quantized GEMM epilogues | wrap vendor GEMM and own the fusion, not another basic GEMM |
| P2 | communication-aware fusions | RMSNorm/all-reduce and TP epilogues after single-GPU evidence |

See the [complete operator catalog](docs/operator-catalog.md),
[operator library design](docs/design/operator-library.md), and
[roadmap](docs/roadmap.md).

## Build And Test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo check --workspace --release

cd website
npm ci
npm run build
```

On a CUDA host:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench rms_norm -- \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 20 --iterations 100 --samples 7
```

The inference-engine-style fused path uses the explicit double in-place
contract `residual = input + residual`, followed by
`input = RMSNorm(residual, weight)`:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench add_rms_norm -- \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 50 --iterations 1000 --samples 9
```

RMSNorm plus dynamic per-token FP8 uses caller-owned output buffers and emits
one F32 dequantization scale per row:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench rms_norm_dynamic_fp8 -- \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15
```

Split-half SiLU-and-Mul accepts `[... , 2 * width]` and produces
`silu(gate) * up` with shape `[... , width]`:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench silu_and_mul -- \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15
```

SiLU-and-Mul plus dynamic per-block FP8 removes the low-precision activation
intermediate and emits one F32 scale per 64- or 128-element output group:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench silu_and_mul_dynamic_fp8 -- \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 1000 --samples 9
```

These programs live under `crates/loom-cuda/benches`, not `src/bin`: they are
validation tools rather than installable product executables. `harness = false`
keeps their JSON CLI behavior while preserving the correct Cargo target
boundary.

The benchmark checks the GPU result against the CPU oracle before reporting
CUDA-event latency. The named PyTorch baselines are:

```bash
python3 benchmarks/pytorch_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 20 --iterations 100 --samples 7

python3 benchmarks/pytorch_add_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 20 --iterations 100 --samples 7

PYTHONPATH=python/src python3 benchmarks/vllm_rms_norm_dynamic_fp8.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15

PYTHONPATH=python/src python3 benchmarks/vllm_silu_and_mul.py \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15

PYTHONPATH=python/src python3 benchmarks/vllm_silu_and_mul_dynamic_fp8.py \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --provider-order baseline-first --result-json /tmp/loom-fp8-ab.json
```

The H20 reports cover
[F32 bring-up](docs/results/h20-rms-norm-f32-smoke-20260721.json) and
[FP16/BF16 vectorization](docs/results/h20-rms-norm-low-precision-20260721.json),
the [fused Add+RMSNorm gate](docs/results/h20-add-rms-norm-20260721.json), and
the [vLLM IR integration](docs/results/h20-vllm-ir-add-rms-norm-20260721.json),
plus the
[RMSNorm+dynamic-FP8 gate](docs/results/h20-rms-norm-dynamic-fp8-20260721.json)
and the
[SiLU-and-Mul compatibility gate](docs/results/h20-silu-and-mul-20260721.json),
plus the
[fused SiLU-and-Mul+block-FP8 gate](docs/results/h20-silu-and-mul-dynamic-fp8-20260721.json).
The
[Qwen2.5 FP8 engine gate](docs/results/h20-vllm-qwen25-05b-fp8-engine-20260722.json)
records the pinned real checkpoint, compiler matches, direct Loom launch
evidence, exact generated tokens, and order-reversed end-to-end measurements.
The fused operator is faster than `vllm_c` in the qualified microbenchmark;
the real engine run proves integration but does not show a measurable
end-to-end speedup. Standalone SiLU-and-Mul is graph-parity coverage. Its
activation-plus-FP8 boundary has an order-stable operator-level advantage and
a real-model correctness gate, but still needs a workload with measurable
model-level benefit.

For the Python build and engine configuration, see the
[vLLM IR provider guide](docs/guides/vllm-ir-provider.md).

## License

MIT
