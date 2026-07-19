from __future__ import annotations

import unittest

from loom_attention.block_binding import (
    BindingStep,
    BlockBindingContractError,
    BlockBindingRegistry,
    PoolObjectRef,
    ReadLeaseProof,
    RequestBlockUpdate,
    _reset_registries_for_testing,
    binding_telemetry_snapshot,
    registry_for_engine,
)
from test_local_delegate import FakeTensor


def step(
    *,
    scheduled=("request-a",),
    updates=(),
    finished=(),
    invalidated=(),
):
    return BindingStep(
        engine_id="engine-a",
        scheduled_request_ids=tuple(scheduled),
        updates=tuple(updates),
        finished_request_ids=tuple(finished),
        invalidated_block_ids=tuple(invalidated),
    )


class BlockBindingRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_registries_for_testing()
        self.registry = BlockBindingRegistry("engine-a")

    def test_mirrors_replace_append_and_resume_updates(self) -> None:
        first = self.registry.apply_step(
            step(updates=(RequestBlockUpdate("request-a", "replace", ((2, 3),)),))
        )
        self.assertEqual(first.scheduled_requests[0].block_ids, ((2, 3),))

        second = self.registry.apply_step(
            step(updates=(RequestBlockUpdate("request-a", "append", ((4,),)),))
        )
        self.assertEqual(second.scheduled_requests[0].block_ids, ((2, 3, 4),))

        resumed = self.registry.apply_step(
            step(
                updates=(RequestBlockUpdate("request-a", "replace", ((8, 9),)),),
                invalidated=(8, 9),
            )
        )
        self.assertEqual(resumed.scheduled_requests[0].block_ids, ((8, 9),))
        self.assertEqual(resumed.generation, 3)

    def test_rejects_incremental_update_without_full_request_state(self) -> None:
        with self.assertRaisesRegex(BlockBindingContractError, "unknown request"):
            self.registry.apply_step(
                step(updates=(RequestBlockUpdate("request-a", "append", ((4,),)),))
            )

    def test_registers_live_cache_without_reading_device_values(self) -> None:
        cache = FakeTensor((32, 2, 16, 2, 64), address=0xCAFE)
        self.registry.register_kv_caches({"model.layers.0.self_attn": cache})

        snapshot = self.registry.snapshot()

        self.assertEqual(len(snapshot.cache_tensors), 1)
        self.assertEqual(snapshot.cache_tensors[0].data_ptr, 0xCAFE)
        self.assertEqual(
            self.registry.kv_cache_tensor("model.layers.0.self_attn"), cache
        )
        self.assertEqual(cache.tolist_calls, 0)

    def test_external_binding_requires_exact_layout_generation_and_lease(self) -> None:
        self.registry.apply_step(
            step(updates=(RequestBlockUpdate("request-a", "replace", ((8,),)),))
        )
        object_ref = PoolObjectRef("pool", "object", 7, "layout")
        lease = ReadLeaseProof("pool", "lease", 200, (object_ref,))

        binding = self.registry.bind_external_block(
            group_index=0,
            physical_block_id=8,
            object_ref=object_ref,
            lease=lease,
            local_layout_digest="layout",
            now_ms=100,
        )

        self.assertEqual(binding.object_ref.generation, 7)
        self.assertEqual(
            self.registry.resolve_external_block(0, 8, now_ms=199), binding
        )
        with self.assertRaisesRegex(BlockBindingContractError, "expired"):
            self.registry.resolve_external_block(0, 8, now_ms=200)
        with self.assertRaisesRegex(BlockBindingContractError, "layout"):
            self.registry.bind_external_block(
                group_index=0,
                physical_block_id=8,
                object_ref=object_ref,
                lease=lease,
                local_layout_digest="other-layout",
                now_ms=100,
            )

    def test_physical_block_reuse_invalidates_external_binding(self) -> None:
        self.registry.apply_step(
            step(updates=(RequestBlockUpdate("request-a", "replace", ((8,),)),))
        )
        object_ref = PoolObjectRef("pool", "object", 7, "layout")
        lease = ReadLeaseProof("pool", "lease", 200, (object_ref,))
        self.registry.bind_external_block(
            group_index=0,
            physical_block_id=8,
            object_ref=object_ref,
            lease=lease,
            local_layout_digest="layout",
            now_ms=100,
        )

        snapshot = self.registry.apply_step(
            step(
                scheduled=("request-b",),
                updates=(RequestBlockUpdate("request-b", "replace", ((8,),)),),
                finished=("request-a",),
            )
        )

        self.assertEqual(snapshot.external_bindings, ())
        self.assertIsNone(self.registry.resolve_external_block(0, 8, now_ms=100))
        self.assertEqual(self.registry.telemetry()["invalidated_external_bindings"], 1)

    def test_failed_step_rolls_back_finished_requests_and_counters(self) -> None:
        self.registry.apply_step(
            step(updates=(RequestBlockUpdate("request-a", "replace", ((2,),)),))
        )
        before = self.registry.snapshot()
        before_telemetry = self.registry.telemetry()

        with self.assertRaisesRegex(BlockBindingContractError, "unknown request"):
            self.registry.apply_step(
                step(
                    scheduled=("missing",),
                    updates=(RequestBlockUpdate("missing", "append", ((3,),)),),
                    finished=("request-a",),
                )
            )

        self.assertEqual(self.registry.snapshot(), before)
        self.assertEqual(self.registry.telemetry(), before_telemetry)

    def test_global_telemetry_omits_raw_device_addresses(self) -> None:
        registry = registry_for_engine("engine-global")
        registry.register_kv_caches(
            {"model.layers.0.self_attn": FakeTensor((8, 2, 16, 2, 64))}
        )

        telemetry = binding_telemetry_snapshot()

        self.assertEqual(telemetry["registry_count"], 1)
        tensor = telemetry["registries"][0]["cache_tensors"][0]
        self.assertNotIn("data_ptr", tensor)


if __name__ == "__main__":
    unittest.main()
