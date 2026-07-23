"""Discovery, loading, and ABI validation for the PyTorch extension."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ._native_build import native_build_info


BRIDGE_ABI_VERSION = 1
_LOCK = threading.Lock()
_LOADED_PATH: Path | None = None


def _version_pair(value: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise RuntimeError(f"cannot parse version pair from {value!r}")
    return int(match.group(1)), int(match.group(2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_native_runtime(
    manifest: Mapping[str, Any],
    extension_path: Path,
) -> None:
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported Loom native manifest schema")
    build = manifest.get("build")
    runtime = manifest.get("runtime")
    if not isinstance(build, Mapping) or not isinstance(runtime, Mapping):
        raise RuntimeError("Loom native manifest is missing build/runtime sections")
    if build.get("bridge_abi_version") != BRIDGE_ABI_VERSION:
        raise RuntimeError(
            "Loom native manifest and Python bridge ABI versions do not match"
        )

    current_torch = _version_pair(torch.__version__)
    torch_runtime = runtime.get("torch")
    if not isinstance(torch_runtime, Mapping):
        raise RuntimeError("Loom native manifest has no PyTorch runtime range")
    if build.get("torch_stable_abi_target") != torch_runtime.get("minimum"):
        raise RuntimeError(
            "Loom native manifest's Stable ABI target and runtime floor differ"
        )
    minimum_torch = _version_pair(str(torch_runtime.get("minimum")))
    maximum_torch = _version_pair(str(torch_runtime.get("maximum_exclusive")))
    if not minimum_torch <= current_torch < maximum_torch:
        raise RuntimeError(
            "Loom's installed native wheel requires PyTorch "
            f">={minimum_torch[0]}.{minimum_torch[1]},"
            f"<{maximum_torch[0]}.{maximum_torch[1]}; "
            f"found {torch.__version__}"
        )

    libraries = manifest.get("libraries")
    if not isinstance(libraries, Mapping):
        raise RuntimeError("Loom native manifest has no library audit")
    expected_hashes = libraries.get("sha256")
    if not isinstance(expected_hashes, Mapping):
        raise RuntimeError("Loom native manifest has no library hashes")
    for library in (
        extension_path.parent / "libloom_cuda_bridge.so",
        extension_path,
    ):
        expected = expected_hashes.get(library.name)
        if (
            not library.is_file()
            or not isinstance(expected, str)
            or _sha256(library) != expected
        ):
            raise RuntimeError(
                f"Loom native library hash mismatch for {library.name}"
            )


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (
            parent / "crates" / "loom-cuda-bridge"
        ).is_dir():
            return parent
    return None


def _candidates() -> tuple[Path, ...]:
    installed = (
        Path(__file__).resolve().parent / "lib" / "libloom_kernels_torch.so"
    )
    if native_build_info() is not None:
        return (installed,)
    repository = _repository_root()
    if repository is not None:
        return (repository / "build" / "libloom_kernels_torch.so",)
    return (installed,)


def torch_extension_path() -> Path | None:
    """Return the first installed Loom PyTorch extension path."""
    for candidate in _candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def torch_extension_available() -> bool:
    """Return whether the single supported framework extension is installed."""
    return torch_extension_path() is not None


def load_torch_extension() -> Path:
    """Load the extension once and reject an incompatible Rust bridge ABI."""
    global _LOADED_PATH
    if _LOADED_PATH is not None:
        return _LOADED_PATH

    path = torch_extension_path()
    if path is None:
        searched = "\n".join(f"  - {candidate}" for candidate in _candidates())
        raise RuntimeError(
            "Loom Kernels requires its compiled PyTorch extension; no Python "
            "or ctypes fallback exists. Run `python python/build_native.py` "
            "and `python python/build_torch_extension.py` from a source "
            f"checkout, or reinstall the native wheel. Searched:\n{searched}"
        )

    manifest = native_build_info()
    if manifest is not None:
        _validate_native_runtime(manifest, path)

    with _LOCK:
        if _LOADED_PATH is None:
            torch.ops.load_library(str(path))
            actual_abi = int(torch.ops.loom_kernels.bridge_abi_version())
            if actual_abi != BRIDGE_ABI_VERSION:
                raise RuntimeError(
                    "Loom Kernels bridge ABI mismatch: Python expects "
                    f"{BRIDGE_ABI_VERSION}, extension reports {actual_abi}. "
                    "Rebuild the Rust bridge and PyTorch extension together."
                )
            _LOADED_PATH = path
    return _LOADED_PATH


__all__ = [
    "BRIDGE_ABI_VERSION",
    "load_torch_extension",
    "torch_extension_available",
    "torch_extension_path",
]
