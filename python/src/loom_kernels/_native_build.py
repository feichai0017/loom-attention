"""Read-only access to the installed native-wheel matrix manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def native_build_info() -> dict[str, Any] | None:
    """Return the native-wheel manifest, or None for a source-only checkout."""
    path = Path(__file__).resolve().parent / "lib" / "native.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Loom native manifest at {path} is not an object")
    return payload


__all__ = ["native_build_info"]
