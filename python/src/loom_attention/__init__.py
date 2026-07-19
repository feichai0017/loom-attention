"""Engine integration and distributed-attention utilities for Loom."""

from .local_delegate import LocalForwardObserver, TensorContractError
from .paged_executor import (
    FlashInferPagedExecutor,
    PagedKvContractError,
    PagedKvView,
)
from .step_metadata import (
    StepMetadataContractError,
    StepMetadataObserver,
    StepMetadataSnapshot,
    TensorDescriptor,
)

__all__ = [
    "LocalForwardObserver",
    "FlashInferPagedExecutor",
    "PagedKvContractError",
    "PagedKvView",
    "StepMetadataContractError",
    "StepMetadataObserver",
    "StepMetadataSnapshot",
    "TensorContractError",
    "TensorDescriptor",
]
