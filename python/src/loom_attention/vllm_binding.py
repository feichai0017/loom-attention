"""Translate vLLM scheduler output into Loom's engine-neutral binding step."""

from __future__ import annotations

from typing import Any

from .block_binding import BindingStep, RequestBlockUpdate


def binding_step_from_scheduler_output(
    engine_id: str, scheduler_output: Any
) -> BindingStep:
    updates: list[RequestBlockUpdate] = []
    for request in scheduler_output.scheduled_new_reqs:
        updates.append(
            RequestBlockUpdate(
                request_id=str(request.req_id),
                mode="replace",
                block_ids=_freeze_groups(request.block_ids),
            )
        )

    cached = scheduler_output.scheduled_cached_reqs
    if len(cached.req_ids) != len(cached.new_block_ids):
        raise ValueError("vLLM cached request and block update counts differ")
    for request_id, block_ids in zip(cached.req_ids, cached.new_block_ids):
        if block_ids is None:
            continue
        updates.append(
            RequestBlockUpdate(
                request_id=str(request_id),
                mode=("replace" if request_id in cached.resumed_req_ids else "append"),
                block_ids=_freeze_groups(block_ids),
            )
        )

    invalidated = scheduler_output.new_block_ids_to_zero or ()
    return BindingStep(
        engine_id=engine_id,
        scheduled_request_ids=tuple(
            str(request_id) for request_id in scheduler_output.num_scheduled_tokens
        ),
        updates=tuple(updates),
        finished_request_ids=tuple(
            sorted(str(request_id) for request_id in scheduler_output.finished_req_ids)
        ),
        invalidated_block_ids=tuple(int(block_id) for block_id in invalidated),
    )


def _freeze_groups(groups: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(block_id) for block_id in group) for group in groups)
