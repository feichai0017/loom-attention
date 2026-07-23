"""Shared PyTorch tensor metadata."""

import torch


_DTYPE_NAMES = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def _require_inference_tensors(*tensors: torch.Tensor) -> None:
    if any(tensor.requires_grad for tensor in tensors):
        raise ValueError("Loom Kernels operators are inference-only")
