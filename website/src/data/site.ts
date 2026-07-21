export const repositoryUrl = "https://github.com/feichai0017/loom-kernels";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Integration", href: "/docs/integration/" },
  { label: "Evidence", href: "/benchmarks/" },
];

export const supportedOperators = [
  {
    name: "RMSNorm",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Standalone normalization",
    status: "supported",
  },
  {
    name: "Add + RMSNorm",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Residual update + normalization",
    status: "supported",
  },
  {
    name: "RMSNorm + dynamic FP8",
    dtypes: "F32 · FP16 · BF16 → E4M3FN",
    boundary: "Normalization + GEMM input quantization",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Split-half SwiGLU activation",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul + block FP8",
    dtypes: "FP16 · BF16 → E4M3FN",
    boundary: "Activation + group-64/128 quantization",
    status: "supported",
  },
];

export const nextOperators = [
  {
    name: "RoPE + paged-KV write",
    reason: "Remove an extra K pass at the cache boundary.",
  },
  {
    name: "Decode-tail sampling",
    reason: "Fuse penalties, filtering, selection, and logprob work.",
  },
  {
    name: "MoE routing + movement",
    reason: "Own the launch-heavy path around vendor grouped GEMM.",
  },
  {
    name: "Paged decode attention",
    reason: "Integrate only against an engine-owned KV contract.",
  },
];

export const evidence = [
  {
    operator: "Add + RMSNorm",
    shape: "BF16 · 8 × 4096",
    result: "2.914 µs",
    detail: "Raw H20 kernel median",
  },
  {
    operator: "RMSNorm + FP8",
    shape: "BF16 · 8 × 4096",
    result: "1.057–1.076×",
    detail: "CUDA Graph ratio vs vLLM",
  },
  {
    operator: "SiLU + Mul + FP8",
    shape: "BF16 · 8 × 11008 · G128",
    result: "1.037–1.082×",
    detail: "CUDA Graph ratio vs vLLM fused",
  },
  {
    operator: "Qwen2.5 FP8 engine",
    shape: "0.5B · batches 1 / 8 / 32",
    result: "0.999–1.004×",
    detail: "Exact-token path hit; end-to-end parity",
  },
];
