"""vLLM 0.25 KVConnector adapter for Loom's physical-block directory.

This connector observes allocation metadata and registers live paged-KV
tensors. It intentionally performs no external load or save yet; Mooncake and
other pool adapters build on the resulting destination-block contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from .block_binding import (
    BindingStep,
    activate_binding_step,
    clear_active_binding_step,
    registry_for_engine,
)
from .vllm_binding import binding_step_from_scheduler_output

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class LoomConnectorMetadata(KVConnectorMetadata):
    binding_step: BindingStep


class LoomMetadataConnector(KVConnectorBase_V1):
    """Metadata-only bridge; it does not claim external KV cache hits."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.engine_id = str(self._kv_transfer_config.engine_id)
        self.registry = registry_for_engine(self.engine_id)

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self.role == KVConnectorRole.WORKER:
            self.registry.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if not isinstance(connector_metadata, LoomConnectorMetadata):
            raise TypeError("Loom received incompatible KV connector metadata")
        super().bind_connector_metadata(connector_metadata)
        if self.role == KVConnectorRole.WORKER:
            snapshot = self.registry.apply_step(connector_metadata.binding_step)
            activate_binding_step(self.engine_id, snapshot.generation)

    def clear_connector_metadata(self) -> None:
        if self.role == KVConnectorRole.WORKER:
            clear_active_binding_step(self.engine_id)
        super().clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens != 0:
            raise RuntimeError(
                "metadata-only Loom connector cannot allocate external tokens"
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return LoomConnectorMetadata(
            binding_step_from_scheduler_output(self.engine_id, scheduler_output)
        )
