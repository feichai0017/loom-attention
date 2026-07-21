# Loom Kernels Python adapters

This package exposes PyTorch current-stream operators and narrow vLLM 0.24
integration points for [Loom Kernels](https://github.com/feichai0017/loom-kernels).

The source wheel contains the Python adapter layer. Build the CUDA C ABI and
LibTorch dispatcher bridge from a repository checkout before GPU use:

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  python python/build_native.py

CUDA_HOME=/usr/local/cuda \
  python python/build_torch_extension.py
```

Repository checkouts discover both libraries under `build/`. A packaged or
externally managed deployment can set `LOOM_KERNELS_CUDA_LIBRARY` and
`LOOM_KERNELS_TORCH_LIBRARY` to their absolute paths. Automated binary wheels
are not published yet.

Install the framework dependencies needed by the consumer:

```bash
pip install -e 'python[torch]'
pip install -e 'python[vllm,test]'
```

See the full
[vLLM provider guide](https://github.com/feichai0017/loom-kernels/blob/main/docs/guides/vllm-ir-provider.md)
for supported contracts, opt-ins, and validation commands.
