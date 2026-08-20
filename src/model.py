"""Shared model types: the task family enum and the per-model discovery record."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .manifest import Manifest, Task

__all__ = ["ModelSpec", "Task"]


@dataclass(frozen=True)
class ModelSpec:
    """A discovered model: where it lives, where its assets come from, and its manifest."""

    folder_name: str
    path: Path
    source: str  # "local" (ONNX present) or "hf" (resolved via manifest repo)
    manifest: Manifest

    @property
    def api_id(self) -> str:
        return self.manifest.name
