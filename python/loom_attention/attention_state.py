"""Attention-state executors shared by Loom data-path experiments."""

from __future__ import annotations

from typing import Any, Sequence


ATTENTION_BACKENDS = ("flashinfer", "reference")
AttentionState = tuple[Any, Any]

__all__ = [
    "ATTENTION_BACKENDS",
    "AttentionState",
    "compute_attention_state",
    "merge_attention_states",
]


def _load_flashinfer() -> Any:
    try:
        import flashinfer
    except ImportError as error:
        raise RuntimeError(
            "FlashInfer backend requested; install './python[flashinfer]'"
        ) from error
    return flashinfer


def _flashinfer_attention_state(
    torch: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    scale: float,
) -> AttentionState:
    flashinfer = _load_flashinfer()
    outputs = []
    logsumexp = []
    for row in query:
        output, row_lse = flashinfer.single_decode_with_kv_cache(
            row.contiguous(),
            key,
            value,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=scale,
            return_lse=True,
        )
        outputs.append(output)
        logsumexp.append(row_lse)
    return (
        torch.stack(outputs).contiguous(),
        torch.stack(logsumexp).float().contiguous(),
    )


def compute_attention_state(
    torch: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    kv_heads: int,
    scale: float,
    backend: str,
) -> AttentionState:
    """Compute an output-plus-LSE state for one contiguous KV segment."""
    if backend == "flashinfer":
        return _flashinfer_attention_state(
            torch,
            query,
            key,
            value,
            scale=scale,
        )
    if backend != "reference":
        raise ValueError(f"unsupported attention backend: {backend}")

    rows, query_heads, head_dim = query.shape
    if key.shape[0] == 0:
        raise ValueError("attention state requires at least one KV token")
    groups = query_heads // kv_heads
    grouped_query = query.float().reshape(rows, kv_heads, groups, head_dim)
    scores = torch.einsum("rhgd,thd->rhgt", grouped_query, key.float()) * scale
    max_logits = scores.amax(dim=-1)
    weights = torch.exp(scores - max_logits.unsqueeze(-1))
    exp_sums = weights.sum(dim=-1)
    weighted_values = torch.einsum("rhgt,thd->rhgd", weights, value.float())
    output = weighted_values / exp_sums.unsqueeze(-1)
    logsumexp = max_logits + torch.log(exp_sums)
    return (
        output.reshape(rows, query_heads, head_dim)
        .to(dtype=query.dtype)
        .contiguous(),
        logsumexp.reshape(rows, query_heads).contiguous(),
    )


def merge_attention_states(
    torch: Any,
    states: Sequence[AttentionState],
    *,
    backend: str,
) -> AttentionState:
    """Merge disjoint KV-segment states using their log-sum-exp values."""
    if not states:
        raise ValueError("at least one attention state is required")
    if backend == "flashinfer":
        flashinfer = _load_flashinfer()
        output, logsumexp = flashinfer.merge_states(
            torch.stack([state[0] for state in states], dim=1),
            torch.stack([state[1] for state in states], dim=1).float(),
        )
        return output.contiguous(), logsumexp.contiguous()
    if backend != "reference":
        raise ValueError(f"unsupported attention backend: {backend}")

    merged_lse = torch.logsumexp(
        torch.stack([state[1] for state in states]), dim=0
    )
    merged_output = torch.zeros_like(states[0][0], dtype=torch.float32)
    for output, logsumexp in states:
        correction = torch.exp(logsumexp - merged_lse)
        merged_output.add_(correction.unsqueeze(-1) * output.float())
    return merged_output.to(dtype=states[0][0].dtype), merged_lse
