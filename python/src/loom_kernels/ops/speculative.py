"""Speculative-decoding predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _greedy_speculative_verify


def supports_greedy_speculative_verify(
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    cumulative_draft_lengths: torch.Tensor,
    max_draft_tokens: int,
) -> bool:
    """Return whether tensors match Loom's flattened ragged verifier."""
    if (
        not isinstance(max_draft_tokens, int)
        or isinstance(max_draft_tokens, bool)
        or max_draft_tokens <= 0
        or max_draft_tokens >= 0xFFFF_FFFF
    ):
        return False
    if cumulative_draft_lengths.dim() != 1:
        return False
    requests = cumulative_draft_lengths.numel()
    return bool(
        draft_token_ids.device.type == "cuda"
        and draft_token_ids.dtype == torch.int32
        and draft_token_ids.dim() == 1
        and 0 < draft_token_ids.numel() <= 0xFFFF_FFFF
        and target_token_ids.device == draft_token_ids.device
        and target_token_ids.dtype == torch.int64
        and target_token_ids.shape == draft_token_ids.shape
        and bonus_token_ids.device == draft_token_ids.device
        and bonus_token_ids.dtype == torch.int32
        and bonus_token_ids.dim() == 2
        and bonus_token_ids.shape == (requests, 1)
        and cumulative_draft_lengths.device == draft_token_ids.device
        and cumulative_draft_lengths.dtype == torch.int32
        and 0 < requests <= 0xFFFF_FFFF
        and draft_token_ids.numel() <= requests * max_draft_tokens
        and draft_token_ids.is_contiguous()
        and target_token_ids.is_contiguous()
        and bonus_token_ids.is_contiguous()
        and cumulative_draft_lengths.is_contiguous()
    )


def _validate_greedy_speculative_verify(
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    cumulative_draft_lengths: torch.Tensor,
    max_draft_tokens: int,
) -> None:
    if not supports_greedy_speculative_verify(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        cumulative_draft_lengths,
        max_draft_tokens,
    ):
        raise ValueError(
            "Loom greedy speculative verification requires non-empty "
            "contiguous CUDA tensors: flattened int32 draft IDs, matching "
            "int64 target IDs, int32 bonus IDs shaped [requests, 1], and "
            "inclusive int32 cumulative draft lengths shaped [requests]; "
            "max_draft_tokens must be positive and contain every ragged row"
        )


def greedy_speculative_verify(
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    cumulative_draft_lengths: torch.Tensor,
    max_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify greedy drafts and return tokens, accepted lengths, and lengths.

    ``cumulative_draft_lengths`` is the inclusive cumulative sum of each
    request's flattened draft-token count. The returned token tensor is int32
    ``[requests, max_draft_tokens + 1]`` with unused entries set to ``-1``.
    """
    _validate_greedy_speculative_verify(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        cumulative_draft_lengths,
        max_draft_tokens,
    )
    requests = cumulative_draft_lengths.numel()
    output_elements = requests * (max_draft_tokens + 1)
    storage = torch.empty(
        output_elements + 2 * requests,
        dtype=torch.int32,
        device=draft_token_ids.device,
    )
    output_token_ids = storage[:output_elements].view(
        requests, max_draft_tokens + 1
    )
    accepted_lengths = storage[output_elements : output_elements + requests]
    emitted_lengths = storage[output_elements + requests :]
    _greedy_speculative_verify(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        cumulative_draft_lengths,
        output_token_ids,
        accepted_lengths,
        emitted_lengths,
        max_draft_tokens,
    )
    return output_token_ids, accepted_lengths, emitted_lengths
