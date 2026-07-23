from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    greedy_speculative_verify,
    launch_count,
    reset_launch_count,
)


def reference(
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    cumulative_draft_lengths: torch.Tensor,
    max_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cumulative = cumulative_draft_lengths.cpu().tolist()
    draft = draft_token_ids.cpu().tolist()
    target = target_token_ids.cpu().tolist()
    bonus = bonus_token_ids.flatten().cpu().tolist()
    output = torch.full(
        (len(cumulative), max_draft_tokens + 1),
        -1,
        dtype=torch.int32,
    )
    accepted = torch.empty(len(cumulative), dtype=torch.int32)
    emitted = torch.empty(len(cumulative), dtype=torch.int32)
    start = 0
    for request, end in enumerate(cumulative):
        length = end - start
        accepted_count = length
        for position in range(length):
            if draft[start + position] != target[start + position]:
                accepted_count = position
                break
        if accepted_count:
            output[request, :accepted_count] = torch.tensor(
                draft[start : start + accepted_count],
                dtype=torch.int32,
            )
        output[request, accepted_count] = (
            target[start + accepted_count]
            if accepted_count < length
            else bonus[request]
        )
        accepted[request] = accepted_count
        emitted[request] = accepted_count + 1
        start = end
    return output.cuda(), accepted.cuda(), emitted.cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_speculative_verify_matches_ragged_reference():
    draft = torch.tensor(
        [10, 11, 12, 20, 21, 22, 23],
        dtype=torch.int32,
        device="cuda",
    )
    target = torch.tensor(
        [10, 99, 12, 20, 21, 22, 23],
        dtype=torch.int64,
        device="cuda",
    )
    bonus = torch.tensor([[100], [200], [300]], dtype=torch.int32, device="cuda")
    cumulative = torch.tensor([3, 3, 7], dtype=torch.int32, device="cuda")

    reset_launch_count(Operator.GREEDY_SPECULATIVE_VERIFY)
    actual = greedy_speculative_verify(draft, target, bonus, cumulative, 4)
    expected = reference(draft, target, bonus, cumulative, 4)
    torch.cuda.synchronize()

    assert launch_count(Operator.GREEDY_SPECULATIVE_VERIFY) == 1
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("lengths", "max_draft_tokens"),
    [([1], 1), ([0, 4, 2, 1], 4), ([8, 3, 0, 7, 2], 8)],
)
def test_greedy_speculative_verify_randomized_shapes(lengths, max_draft_tokens):
    torch.manual_seed(20260723 + sum(lengths))
    total = sum(lengths)
    draft = torch.randint(
        0, 32000, (total,), dtype=torch.int32, device="cuda"
    )
    target = draft.to(torch.int64)
    for request, start in enumerate(
        [0, *torch.tensor(lengths).cumsum(0).tolist()[:-1]]
    ):
        if lengths[request] > 1 and request % 2:
            target[start + 1] += 7
    bonus = torch.randint(
        0, 32000, (len(lengths), 1), dtype=torch.int32, device="cuda"
    )
    cumulative = torch.tensor(
        torch.tensor(lengths).cumsum(0).tolist(),
        dtype=torch.int32,
        device="cuda",
    )

    actual = greedy_speculative_verify(
        draft, target, bonus, cumulative, max_draft_tokens
    )
    expected = reference(
        draft, target, bonus, cumulative, max_draft_tokens
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_speculative_verify_uses_current_external_stream():
    draft = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    target = torch.tensor([1, 2, 9, 4], dtype=torch.int64, device="cuda")
    bonus = torch.tensor([[5], [6]], dtype=torch.int32, device="cuda")
    cumulative = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = greedy_speculative_verify(
            draft, target, bonus, cumulative, 2
        )
    stream.synchronize()
    expected = reference(draft, target, bonus, cumulative, 2)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_speculative_verify_dispatcher_contract_and_fake_tensor():
    draft = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    target = draft.long()
    bonus = torch.tensor([[4], [5]], dtype=torch.int32, device="cuda")
    cumulative = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    torch.library.opcheck(
        torch.ops.loom_kernels.greedy_speculative_verify.default,
        (
            draft,
            target,
            bonus,
            cumulative,
            torch.empty((2, 3), dtype=torch.int32, device="cuda"),
            torch.empty(2, dtype=torch.int32, device="cuda"),
            torch.empty(2, dtype=torch.int32, device="cuda"),
            2,
        ),
        test_utils=(
            "test_schema",
            "test_faketensor",
            "test_autograd_registration",
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_speculative_verify_survives_torch_compile():
    @torch.compile(fullgraph=True)
    def compiled(draft, target, bonus, cumulative):
        return greedy_speculative_verify(draft, target, bonus, cumulative, 3)

    draft = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
    target = torch.tensor([1, 9, 3, 4, 5], dtype=torch.int64, device="cuda")
    bonus = torch.tensor([[6], [7]], dtype=torch.int32, device="cuda")
    cumulative = torch.tensor([2, 5], dtype=torch.int32, device="cuda")
    actual = compiled(draft, target, bonus, cumulative)
    expected = reference(draft, target, bonus, cumulative, 3)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_speculative_verify_cuda_graph_replay():
    draft = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    target = draft.long()
    bonus = torch.tensor([[5], [6]], dtype=torch.int32, device="cuda")
    cumulative = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
    for _ in range(3):
        outputs = greedy_speculative_verify(
            draft, target, bonus, cumulative, 2
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = greedy_speculative_verify(
            draft, target, bonus, cumulative, 2
        )
    target[0] = 9
    graph.replay()
    torch.cuda.synchronize()

    expected = reference(draft, target, bonus, cumulative, 2)
    for actual_tensor, expected_tensor in zip(outputs, expected):
        assert torch.equal(actual_tensor, expected_tensor)


def test_greedy_speculative_verify_rejects_invalid_inputs():
    draft = torch.tensor([1, 2], dtype=torch.int32)
    target = draft.long()
    bonus = torch.tensor([[3]], dtype=torch.int32)
    cumulative = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(ValueError, match="contiguous CUDA tensors"):
        greedy_speculative_verify(draft, target, bonus, cumulative, 2)
