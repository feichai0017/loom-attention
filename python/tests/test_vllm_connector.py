from __future__ import annotations

from enum import Enum
import importlib
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from loom_attention.block_binding import (
    _reset_registries_for_testing,
    binding_telemetry_snapshot,
    validate_active_binding_step,
)
from test_local_delegate import FakeTensor


class FakeRole(Enum):
    SCHEDULER = 0
    WORKER = 1


class FakeMetadata:
    pass


class FakeConnectorBase:
    def __init__(self, vllm_config, role, kv_cache_config) -> None:
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._role = role
        self._connector_metadata = None

    @property
    def role(self):
        return self._role

    def bind_connector_metadata(self, metadata) -> None:
        self._connector_metadata = metadata

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None


def fake_vllm_modules() -> dict[str, ModuleType]:
    names = (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    )
    modules = {name: ModuleType(name) for name in names}
    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorBase_V1 = FakeConnectorBase
    base.KVConnectorMetadata = FakeMetadata
    base.KVConnectorRole = FakeRole
    modules[base.__name__] = base
    return modules


def scheduler_output():
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="request", block_ids=([4, 5],)),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set(), new_block_ids=[]
        ),
        num_scheduled_tokens={"request": 16},
        finished_req_ids=set(),
        new_block_ids_to_zero=None,
    )


class VllmConnectorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_registries_for_testing()
        sys.modules.pop("loom_attention.vllm_connector", None)

    def test_scheduler_metadata_drives_worker_registry(self) -> None:
        with patch.dict(sys.modules, fake_vllm_modules(), clear=False):
            connector_module = importlib.import_module("loom_attention.vllm_connector")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(engine_id="engine")
            )
            scheduler = connector_module.LoomMetadataConnector(
                config, FakeRole.SCHEDULER, SimpleNamespace()
            )
            worker = connector_module.LoomMetadataConnector(
                config, FakeRole.WORKER, SimpleNamespace()
            )
            cache = FakeTensor((16, 2, 16, 2, 64))
            worker.register_kv_caches({"model.layers.0.self_attn": cache})

            metadata = scheduler.build_connector_meta(scheduler_output())
            worker.bind_connector_metadata(metadata)
            active = validate_active_binding_step(request_count=1)
            self.assertIsNotNone(active)
            worker.clear_connector_metadata()
            self.assertIsNone(validate_active_binding_step(request_count=1))

        telemetry = binding_telemetry_snapshot()
        self.assertEqual(telemetry["registry_count"], 1)
        self.assertEqual(telemetry["metadata_steps"], 1)
        self.assertEqual(telemetry["request_updates"], 1)
        self.assertEqual(telemetry["registered_cache_tensor_count"], 1)
        self.assertEqual(telemetry["unique_block_ids_seen"], 2)
        self.assertEqual(telemetry["validated_attention_forwards"], 1)
        self.assertEqual(cache.tolist_calls, 0)


if __name__ == "__main__":
    unittest.main()
