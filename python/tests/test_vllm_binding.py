from __future__ import annotations

from types import SimpleNamespace
import unittest

from loom_attention.vllm_binding import binding_step_from_scheduler_output


class VllmBindingAdapterTest(unittest.TestCase):
    def test_translates_new_append_resume_finish_and_reuse_events(self) -> None:
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="new", block_ids=([1, 2],)),
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["running", "resumed", "unchanged"],
                resumed_req_ids={"resumed"},
                new_block_ids=[([3],), ([7, 8],), None],
            ),
            num_scheduled_tokens={
                "running": 1,
                "new": 16,
                "resumed": 1,
                "unchanged": 1,
            },
            finished_req_ids={"finished"},
            new_block_ids_to_zero=[1, 7],
        )

        result = binding_step_from_scheduler_output("engine", scheduler_output)

        self.assertEqual(
            result.scheduled_request_ids,
            ("running", "new", "resumed", "unchanged"),
        )
        self.assertEqual(
            [
                (update.request_id, update.mode, update.block_ids)
                for update in result.updates
            ],
            [
                ("new", "replace", ((1, 2),)),
                ("running", "append", ((3,),)),
                ("resumed", "replace", ((7, 8),)),
            ],
        )
        self.assertEqual(result.finished_request_ids, ("finished",))
        self.assertEqual(result.invalidated_block_ids, (1, 7))

    def test_rejects_misaligned_cached_request_updates(self) -> None:
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["request"],
                resumed_req_ids=set(),
                new_block_ids=[],
            ),
            num_scheduled_tokens={},
            finished_req_ids=set(),
            new_block_ids_to_zero=None,
        )

        with self.assertRaisesRegex(ValueError, "counts differ"):
            binding_step_from_scheduler_output("engine", scheduler_output)


if __name__ == "__main__":
    unittest.main()
