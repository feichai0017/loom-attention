# Implementation Status

## Implemented

- Rust workspace split into public contracts, safe CUDA backend, and raw FFI;
- `DType`, contiguous `TensorSpec`, normalization operator specs,
  `OperatorSpec`, and backend capability query;
- F64-accumulation F32/FP16/BF16 RMSNorm and fused Add+RMSNorm CPU oracles;
- owned CUDA streams, device buffers, and events;
- checked handwritten F32 plus pair-vectorized FP16/BF16 RMSNorm dispatch;
- double in-place Add+RMSNorm with Rust-exclusive mutable-buffer contracts;
- aligned 128-bit pack8 Add+RMSNorm with pair and scalar fallbacks;
- three-pass RMSNorm plus dynamic per-token FP8 E4M3FN quantization, with
  pack4 output conversion and scalar fallback;
- checked safe-Rust, raw C ABI, caller-allocated PyTorch out, and convenience
  allocating entrypoints for RMSNorm+FP8;
- split-half SiLU-and-Mul contracts and CPU oracles for F32/FP16/BF16;
- handwritten SiLU-and-Mul with aligned 16-byte packs and scalar fallback;
- FP16/BF16 SiLU-and-Mul fused directly into dynamic per-block FP8 E4M3FN,
  with groups 64/128, optional scale upper bound, and row/group-major scales;
- a C++ PyTorch dispatcher bridge using the current CUDA stream;
- a source-adapter Python wheel with explicit framework extras, project
  metadata, license/readme payloads, and a CI install/entry-point smoke gate;
- a `loom_cuda` vLLM IR provider with native fallback and an opt-in vLLM
  `SiluAndMul` out-of-tree layer replacement, plus an opt-in activation-quant
  fusion-table replacement for vLLM 0.24;
- per-operator JSON correctness/latency benchmarks and named vLLM baselines.

## Validated

- local formatting, clippy, tests, and release build;
- the Python adapter wheel built, installed into an isolated environment,
  passed dependency checks, imported, and exposed its vLLM entry point;
- CUDA-feature clippy on `forge-gas1`;
- NVIDIA H20 correctness for six shapes from `1x1` through `16x8192` and a
  `64x4096` batch;
- maximum absolute error `4.77e-7` against the CPU oracle;
- `8x4096` post-cleanup seven-sample isolated median `7.27 us`.
- FP16 and BF16 pair paths plus odd-size scalar fallbacks validated on H20;
- `8x4096` seven-sample medians: FP16 `4.75 us`, BF16 `4.71 us`;
- maximum FP16 absolute error `0.001953125`; BF16 cases were exact against the
  quantized-output oracle for the tested deterministic inputs.
- fused Add+RMSNorm CUDA-feature Clippy, release build, and all tests passed;
- fused `8x4096` H20 medians after pack8 optimization: FP16 `3.061 us` and
  BF16 `2.914 us` over 15 samples and 2000 launches per sample; the F32 path
  remains `9.062 us` in its earlier gate;
- fused residual outputs were exact against the oracle in every tested dtype
  and shape; maximum normalized-output error was `4.77e-7` for F32,
  `4.88e-4` for FP16, and zero for the tested BF16 cases;
- fused FP16/BF16 scalar fallbacks passed at `3x127`, and low-precision
  large-shape execution passed at `16x8192`.
- PyTorch external-stream, mutation-schema/FakeTensor, `torch.compile`, and
  CUDA Graph tests passed with the C++ dispatcher bridge;
- Loom and vLLM's CUDA provider were bitwise identical for BF16 at `1x4096`,
  `8x4096`, `128x4096`, and `8x8192`;
- through the same vLLM IR dispatch, order-reversed H20 runs measured Loom
  `8.260-8.261 us` versus `vllm_c` `9.133-9.202 us` in eager execution, and
  Loom `2.769-2.779 us` versus `vllm_c` `2.866-2.877 us` in CUDA Graph replay;
- the provider completed vLLM 0.24 compilation, graph capture, and repeated
  generation with the normal Qwen2 model runner and a synthetic random model.
- RMSNorm+FP8 F32/FP16/BF16 outputs and F32 row scales were bitwise identical
  to vLLM 0.24 at `8x4096`, `3x127`, and `2x128` on H20;
- the RMSNorm+FP8 PyTorch suite passed 20 tests covering external streams,
  dispatcher mutation, out-buffer reuse, fullgraph compilation, and CUDA Graph
  capture/replay;
- order-reversed BF16 `8x4096` named-baseline runs measured Loom
  `5.311-5.430 us` versus vLLM `7.994-8.052 us` through eager C++ dispatch,
  and Loom `3.947-4.044 us` versus vLLM `4.246-4.274 us` under graph replay.
- SiLU-and-Mul F32, FP16, and BF16 kernels passed CPU-oracle checks at
  `8x11008`, and the BF16 scalar fallback passed at `3x127`; observed raw
  medians were `3.975 us`, `4.279 us`, `4.392 us`, and `2.932 us`
  respectively;
- SiLU-and-Mul matched vLLM bitwise for F32/FP16/BF16 over representative,
  odd-width, and higher-rank inputs; the full Python suite passed 43 tests,
  including external streams, schema/FakeTensor checks, fullgraph compilation,
  CUDA Graph capture/replay, and vLLM layer dispatch;
- order-reversed BF16 `8x11008` named-baseline runs put Loom and vLLM within
  `0.1%` under CUDA Graph replay (`3.932-4.060 us` versus
  `3.935-4.061 us`). Eager dispatch changed materially with run order, so no
  standalone SiLU speedup is claimed;
- with the opt-in vLLM layer override, a synthetic Qwen2 engine completed
  model compilation, CUDA Graph capture, and generation on H20;
- SiLU-and-Mul+block-FP8 matched vLLM's fused operator exactly for FP16/BF16,
  groups 64/128, representative and higher-rank inputs, row/group-major scale
  layouts, and optional scale upper bounds;
- the complete H20 Python suite passed 62 tests; the 27 focused activation and
  vLLM tests also covered external streams, mutation schema/FakeTensor,
  `torch.compile`, auto-functionalization, CUDA Graph replay, and fusion-table
  registration; vLLM's official activation-quant pattern matcher rewrote the
  composed graph to Loom with one match;
- raw `8x11008` medians were `3.268 us`/`3.060 us` for FP16 group 64/128 and
  `3.262 us`/`2.952 us` for BF16 group 64/128, with zero byte or scale errors;
- order-reversed BF16 group-128 named-baseline runs measured
  `1.216-1.231x` eager speedup ratios (`17.7-18.8%` lower latency) and
  `1.037-1.082x` CUDA Graph ratios (`3.6-7.5%` lower latency) against vLLM's
  semantically identical fused operator.
- a pinned Qwen2.5-0.5B-Instruct checkpoint completed vLLM 0.24 online
  `fp8_per_block` loading, compilation, activation-quant rewriting, CUDA Graph
  capture, and repeated generation through Loom on H20;
- baseline-first and Loom-first runs matched every generated token ID across
  `1x128x128`, `8x128x128`, and `32x128x64` request shapes; both providers
  recorded two compiler-pattern matches, while the process-local path probe
  recorded zero Loom launches for the baseline and 1584 for Loom;
- order-reversed end-to-end batch-latency ratios ranged from `0.9991x` to
  `1.0043x`. This closes real-model invocation and correctness, but is parity
  rather than evidence of model-level TTFT, TPOT, or throughput improvement.

See the [F32 report](results/h20-rms-norm-f32-smoke-20260721.json) and
[low-precision report](results/h20-rms-norm-low-precision-20260721.json), plus
the [fused Add+RMSNorm report](results/h20-add-rms-norm-20260721.json) and
[vLLM IR integration report](results/h20-vllm-ir-add-rms-norm-20260721.json).
The [RMSNorm+dynamic-FP8 report](results/h20-rms-norm-dynamic-fp8-20260721.json)
contains its exact contract, raw CUDA results, and order-reversed vLLM
comparison. The [SiLU-and-Mul report](results/h20-silu-and-mul-20260721.json)
records exact vLLM compatibility, graph parity, eager instability, and the
engine smoke gate. The
[SiLU-and-Mul+dynamic-block-FP8 report](results/h20-silu-and-mul-dynamic-fp8-20260721.json)
records the fused contract, exact vLLM comparison, raw CUDA results, and
order-reversed named baseline. The
[Qwen2.5 FP8 engine report](results/h20-vllm-qwen25-05b-fp8-engine-20260722.json)
records the pinned checkpoint, path-hit evidence, exact-token gates, and
order-reversed end-to-end parity result.

## Not Yet Proven

- fused Add+RMSNorm or RMSNorm+FP8 model-level benefit;
- SiLU-and-Mul+FP8 model-level benefit on a workload where the exposed
  activation-quant boundary is material;
- integration into SGLang or a Rust-native engine path;
- larger production-model and serving-workload validation;
- automated binary-wheel packaging;
- end-to-end TTFT, TPOT, throughput, or memory improvement.
