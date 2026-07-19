"""Engine adapters for QuillCache."""

from .local_delegate import LocalForwardObserver, TensorContractError

__all__ = ["LocalForwardObserver", "TensorContractError"]
