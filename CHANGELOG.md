# Changelog

Loom Kernels follows Semantic Versioning. The Rust crates use Cargo's SemVer
spelling; Python source-adapter metadata uses the equivalent PEP 440 spelling.

## Unreleased

### Added

- non-owning `CudaStreamRef` execution for framework-controlled CUDA streams;
- typed `DeviceSlice` and `DeviceSliceMut` views over framework-owned device
  allocations;
- sealed read/write memory traits so every safe Rust operator accepts owned or
  borrowed storage through the same contract validation;
- an H20 smoke path covering borrowed stream and allocation execution without
  ownership transfer.

## 1.0.0-alpha.1 — 2026-07-22

First public alpha of Loom Kernels as a Rust-first CUDA operator backend for
LLM inference.

GitHub tag: `loom-kernels-v1.0.0-alpha.1`. The namespaced tag preserves the
repository's unrelated historical `v1.0.0-alpha.1` QuillCache release.

### Included

- backend-independent Rust contracts, capability queries, and deterministic
  CPU oracles;
- safe Rust CUDA streams, buffers, events, checked dispatch, and a raw C ABI;
- handwritten CUDA for normalization/quantization, SwiGLU, RoPE plus paged-KV
  writes, decode-tail sampling and logprobs, Min-P, and paged decode attention;
- opt-in PyTorch and vLLM 0.24 adapters with explicit shape and policy gates;
- H20 correctness, framework, engine, and named-baseline evidence kept as
  machine-readable artifacts;
- self-contained Cargo source archives and a pure Rust CUDA smoke example.

### Alpha boundaries

- APIs and admitted shape envelopes may change before 1.0 stable;
- CUDA is opt-in and requires a local NVIDIA toolkit at build time;
- Python packaging is source-adapter metadata, not a portable CUDA/LibTorch
  binary wheel;
- unsupported engine contracts intentionally fall back to the native backend.
