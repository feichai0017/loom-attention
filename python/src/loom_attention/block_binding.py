"""Node-local binding between engine block slots and external KV objects.

The inference engine owns its physical block table. Loom observes CPU-side
allocation updates and keeps an equivalent request-to-block directory without
reading device page-table values back to the host. External pool objects may be
bound to those slots only with an exact generation and lease proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import local, Lock, RLock
from typing import Any, Literal


class BlockBindingContractError(RuntimeError):
    """Raised when an engine or pool update would create an unsafe binding."""


@dataclass(frozen=True)
class PoolObjectRef:
    pool_id: str
    object_key: str
    generation: int
    layout_digest: str
    checksum: str | None = None


@dataclass(frozen=True)
class ReadLeaseProof:
    pool_id: str
    lease_id: str
    expires_at_ms: int
    objects: tuple[PoolObjectRef, ...]


@dataclass(frozen=True)
class CacheTensorDescriptor:
    layer_name: str
    shape: tuple[int, ...]
    device: str
    dtype: str
    data_ptr: int
    bytes: int


@dataclass(frozen=True)
class RequestBlockUpdate:
    request_id: str
    mode: Literal["replace", "append"]
    block_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BindingStep:
    engine_id: str
    scheduled_request_ids: tuple[str, ...]
    updates: tuple[RequestBlockUpdate, ...]
    finished_request_ids: tuple[str, ...]
    invalidated_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class RequestBlockState:
    request_id: str
    block_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ExternalBlockBinding:
    group_index: int
    physical_block_id: int
    object_ref: PoolObjectRef
    lease: ReadLeaseProof
    local_layout_digest: str
    installed_generation: int


@dataclass(frozen=True)
class BlockBindingSnapshot:
    engine_id: str
    generation: int
    scheduled_requests: tuple[RequestBlockState, ...]
    active_request_count: int
    cache_tensors: tuple[CacheTensorDescriptor, ...]
    external_bindings: tuple[ExternalBlockBinding, ...]


class BlockBindingRegistry:
    """Generation-checked mirror built only from engine CPU allocation events."""

    def __init__(self, engine_id: str) -> None:
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        self.engine_id = engine_id
        self._lock = RLock()
        self._generation = 0
        self._requests: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._scheduled_request_ids: tuple[str, ...] = ()
        self._cache_tensors: dict[str, Any] = {}
        self._cache_descriptors: dict[str, CacheTensorDescriptor] = {}
        self._external: dict[tuple[int, int], ExternalBlockBinding] = {}
        self._metadata_steps = 0
        self._request_updates = 0
        self._invalidated_bindings = 0
        self._seen_block_ids: set[int] = set()
        self._validated_attention_forwards = 0

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if not kv_caches:
            raise BlockBindingContractError("vLLM registered no KV cache tensors")
        descriptors: dict[str, CacheTensorDescriptor] = {}
        for layer_name, tensor in kv_caches.items():
            descriptors[layer_name] = _describe_cache_tensor(layer_name, tensor)
        with self._lock:
            if self._cache_descriptors and self._cache_descriptors != descriptors:
                raise BlockBindingContractError(
                    "KV cache tensors changed without creating a new engine registry"
                )
            self._cache_tensors = dict(kv_caches)
            self._cache_descriptors = descriptors

    def apply_step(self, step: BindingStep) -> BlockBindingSnapshot:
        if step.engine_id != self.engine_id:
            raise BlockBindingContractError(
                f"step engine {step.engine_id!r} does not match {self.engine_id!r}"
            )
        with self._lock:
            rollback = (
                dict(self._requests),
                self._scheduled_request_ids,
                dict(self._external),
                self._metadata_steps,
                self._request_updates,
                self._invalidated_bindings,
                set(self._seen_block_ids),
            )
            try:
                for request_id in step.finished_request_ids:
                    self._requests.pop(request_id, None)

                # Every block sent in an allocation update may represent a
                # reused physical slot. Clearing a shared prefix binding is
                # conservative but safe; this step can install it again.
                invalidated = set(step.invalidated_block_ids)
                invalidated.update(
                    block_id
                    for update in step.updates
                    for group in update.block_ids
                    for block_id in group
                )
                if any(block_id < 0 for block_id in invalidated):
                    raise BlockBindingContractError(
                        "invalidated physical block IDs must be non-negative"
                    )
                for key in tuple(self._external):
                    if key[1] in invalidated:
                        del self._external[key]
                        self._invalidated_bindings += 1

                for update in step.updates:
                    self._apply_update(update)

                missing = [
                    request_id
                    for request_id in step.scheduled_request_ids
                    if request_id not in self._requests
                ]
                if missing:
                    raise BlockBindingContractError(
                        "scheduled requests have no physical block state: "
                        + ", ".join(missing)
                    )
                self._scheduled_request_ids = step.scheduled_request_ids
                self._metadata_steps += 1
                self._generation += 1
                return self._snapshot_locked()
            except BaseException:
                (
                    self._requests,
                    self._scheduled_request_ids,
                    self._external,
                    self._metadata_steps,
                    self._request_updates,
                    self._invalidated_bindings,
                    self._seen_block_ids,
                ) = rollback
                raise

    def bind_external_block(
        self,
        *,
        group_index: int,
        physical_block_id: int,
        object_ref: PoolObjectRef,
        lease: ReadLeaseProof,
        local_layout_digest: str,
        now_ms: int,
    ) -> ExternalBlockBinding:
        _validate_external_binding(
            group_index=group_index,
            physical_block_id=physical_block_id,
            object_ref=object_ref,
            lease=lease,
            local_layout_digest=local_layout_digest,
            now_ms=now_ms,
        )
        with self._lock:
            if not self._is_live_physical_block(group_index, physical_block_id):
                raise BlockBindingContractError(
                    "cannot bind an external object to an unobserved physical block"
                )
            self._generation += 1
            binding = ExternalBlockBinding(
                group_index=group_index,
                physical_block_id=physical_block_id,
                object_ref=object_ref,
                lease=lease,
                local_layout_digest=local_layout_digest,
                installed_generation=self._generation,
            )
            self._external[(group_index, physical_block_id)] = binding
            return binding

    def resolve_external_block(
        self,
        group_index: int,
        physical_block_id: int,
        *,
        now_ms: int,
    ) -> ExternalBlockBinding | None:
        with self._lock:
            binding = self._external.get((group_index, physical_block_id))
            if binding is None:
                return None
            if binding.lease.expires_at_ms <= now_ms:
                raise BlockBindingContractError(
                    f"read lease {binding.lease.lease_id!r} has expired"
                )
            return binding

    def kv_cache_tensor(self, layer_name: str) -> Any:
        with self._lock:
            try:
                return self._cache_tensors[layer_name]
            except KeyError as error:
                raise BlockBindingContractError(
                    f"KV cache tensor for layer {layer_name!r} is not registered"
                ) from error

    def snapshot(self) -> BlockBindingSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            block_ids = sorted(self._seen_block_ids)
            return {
                "engine_id": self.engine_id,
                "generation": self._generation,
                "metadata_steps": self._metadata_steps,
                "request_updates": self._request_updates,
                "active_request_count": len(self._requests),
                "scheduled_request_count": len(self._scheduled_request_ids),
                "registered_cache_tensor_count": len(self._cache_descriptors),
                "cache_tensors": [
                    {
                        key: value
                        for key, value in asdict(descriptor).items()
                        if key != "data_ptr"
                    }
                    for descriptor in sorted(
                        self._cache_descriptors.values(),
                        key=lambda item: item.layer_name,
                    )
                ],
                "unique_block_ids_seen": len(block_ids),
                "min_block_id_seen": block_ids[0] if block_ids else None,
                "max_block_id_seen": block_ids[-1] if block_ids else None,
                "external_binding_count": len(self._external),
                "invalidated_external_bindings": self._invalidated_bindings,
                "validated_attention_forwards": self._validated_attention_forwards,
            }

    def validate_attention_forward(
        self, *, generation: int, request_count: int
    ) -> BlockBindingSnapshot:
        with self._lock:
            if generation != self._generation:
                raise BlockBindingContractError(
                    "attention forward observed a stale block-binding generation"
                )
            if not self._cache_descriptors:
                raise BlockBindingContractError(
                    "attention forward has no registered KV cache tensors"
                )
            scheduled_count = len(self._scheduled_request_ids)
            if scheduled_count <= 0 or scheduled_count > request_count:
                raise BlockBindingContractError(
                    "attention metadata request count does not cover the "
                    "scheduled block bindings"
                )
            self._validated_attention_forwards += 1
            return self._snapshot_locked()

    def _apply_update(self, update: RequestBlockUpdate) -> None:
        if not update.request_id:
            raise BlockBindingContractError("request_id must be non-empty")
        groups = _validate_block_groups(update.block_ids)
        if update.mode == "replace":
            self._requests[update.request_id] = groups
        elif update.mode == "append":
            current = self._requests.get(update.request_id)
            if current is None:
                raise BlockBindingContractError(
                    f"cannot append blocks for unknown request {update.request_id!r}"
                )
            if len(current) != len(groups):
                raise BlockBindingContractError(
                    "KV cache group count changed during an append"
                )
            self._requests[update.request_id] = tuple(
                existing + added for existing, added in zip(current, groups)
            )
        else:
            raise BlockBindingContractError(
                f"unsupported block update mode: {update.mode!r}"
            )
        self._request_updates += 1
        for group in groups:
            self._seen_block_ids.update(group)

    def _is_live_physical_block(self, group_index: int, block_id: int) -> bool:
        return any(
            group_index < len(groups) and block_id in groups[group_index]
            for groups in self._requests.values()
        )

    def _snapshot_locked(self) -> BlockBindingSnapshot:
        scheduled = tuple(
            RequestBlockState(request_id, self._requests[request_id])
            for request_id in self._scheduled_request_ids
        )
        external = tuple(self._external[key] for key in sorted(self._external))
        descriptors = tuple(
            sorted(
                self._cache_descriptors.values(),
                key=lambda item: item.layer_name,
            )
        )
        return BlockBindingSnapshot(
            engine_id=self.engine_id,
            generation=self._generation,
            scheduled_requests=scheduled,
            active_request_count=len(self._requests),
            cache_tensors=descriptors,
            external_bindings=external,
        )


_REGISTRY_LOCK = Lock()
_REGISTRIES: dict[str, BlockBindingRegistry] = {}
_ACTIVE_BINDING = local()


def registry_for_engine(engine_id: str) -> BlockBindingRegistry:
    with _REGISTRY_LOCK:
        registry = _REGISTRIES.get(engine_id)
        if registry is None:
            registry = BlockBindingRegistry(engine_id)
            _REGISTRIES[engine_id] = registry
        return registry


def binding_telemetry_snapshot() -> dict[str, Any]:
    with _REGISTRY_LOCK:
        registries = list(_REGISTRIES.values())
    reports = [registry.telemetry() for registry in registries]
    return {
        "registry_count": len(reports),
        "metadata_steps": sum(item["metadata_steps"] for item in reports),
        "request_updates": sum(item["request_updates"] for item in reports),
        "registered_cache_tensor_count": sum(
            item["registered_cache_tensor_count"] for item in reports
        ),
        "unique_block_ids_seen": sum(item["unique_block_ids_seen"] for item in reports),
        "external_binding_count": sum(
            item["external_binding_count"] for item in reports
        ),
        "validated_attention_forwards": sum(
            item["validated_attention_forwards"] for item in reports
        ),
        "registries": reports,
    }


def activate_binding_step(engine_id: str, generation: int) -> None:
    _ACTIVE_BINDING.value = (engine_id, generation)


def clear_active_binding_step(engine_id: str) -> None:
    active = getattr(_ACTIVE_BINDING, "value", None)
    if active is not None and active[0] == engine_id:
        del _ACTIVE_BINDING.value


def has_active_binding_step() -> bool:
    return getattr(_ACTIVE_BINDING, "value", None) is not None


def validate_active_binding_step(*, request_count: int) -> BlockBindingSnapshot | None:
    active = getattr(_ACTIVE_BINDING, "value", None)
    if active is None:
        return None
    engine_id, generation = active
    return registry_for_engine(engine_id).validate_attention_forward(
        generation=generation,
        request_count=request_count,
    )


def _reset_registries_for_testing() -> None:
    with _REGISTRY_LOCK:
        _REGISTRIES.clear()
    if hasattr(_ACTIVE_BINDING, "value"):
        del _ACTIVE_BINDING.value


def _validate_block_groups(
    groups: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if not groups:
        raise BlockBindingContractError(
            "a request block update must contain at least one KV cache group"
        )
    if any(block_id < 0 for group in groups for block_id in group):
        raise BlockBindingContractError("physical block IDs must be non-negative")
    return tuple(tuple(int(block_id) for block_id in group) for group in groups)


def _describe_cache_tensor(layer_name: str, tensor: Any) -> CacheTensorDescriptor:
    if not layer_name:
        raise BlockBindingContractError("KV cache layer names must be non-empty")
    missing = [
        name
        for name in ("shape", "device", "dtype", "data_ptr")
        if not hasattr(tensor, name)
    ]
    if missing:
        raise BlockBindingContractError(
            f"KV cache {layer_name!r} is missing tensor attributes: "
            + ", ".join(missing)
        )
    shape = tuple(int(dimension) for dimension in tensor.shape)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise BlockBindingContractError(
            f"KV cache {layer_name!r} must have a non-empty positive shape"
        )
    numel = _integer_tensor_method(tensor, "numel", _product(shape))
    element_size = _integer_tensor_method(tensor, "element_size", 0)
    if numel <= 0 or element_size <= 0:
        raise BlockBindingContractError(
            f"KV cache {layer_name!r} has invalid storage size"
        )
    data_ptr = int(tensor.data_ptr())
    if data_ptr <= 0:
        raise BlockBindingContractError(
            f"KV cache {layer_name!r} has no live storage address"
        )
    return CacheTensorDescriptor(
        layer_name=layer_name,
        shape=shape,
        device=str(tensor.device),
        dtype=str(tensor.dtype),
        data_ptr=data_ptr,
        bytes=numel * element_size,
    )


def _integer_tensor_method(tensor: Any, name: str, default: int) -> int:
    method = getattr(tensor, name, None)
    return int(method()) if callable(method) else int(default)


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _validate_external_binding(
    *,
    group_index: int,
    physical_block_id: int,
    object_ref: PoolObjectRef,
    lease: ReadLeaseProof,
    local_layout_digest: str,
    now_ms: int,
) -> None:
    if group_index < 0 or physical_block_id < 0:
        raise BlockBindingContractError(
            "group index and physical block ID must be non-negative"
        )
    if not object_ref.pool_id or not object_ref.object_key:
        raise BlockBindingContractError("pool object identity must be non-empty")
    if object_ref.generation <= 0:
        raise BlockBindingContractError("pool object generation must be positive")
    if not local_layout_digest or object_ref.layout_digest != local_layout_digest:
        raise BlockBindingContractError(
            "pool object layout does not match the local KV cache layout"
        )
    if lease.pool_id != object_ref.pool_id or not lease.lease_id:
        raise BlockBindingContractError("read lease does not cover the object pool")
    if lease.expires_at_ms <= now_ms:
        raise BlockBindingContractError("read lease has already expired")
    if object_ref not in lease.objects:
        raise BlockBindingContractError(
            "read lease does not cover the exact object generation"
        )
