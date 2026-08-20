"""Model discovery: scan the model root for valid, loadable models.

A directory is a valid model when it has a ``manifest.yaml`` **and** either at least
one ONNX file on disk (``local``) or an ``hf_url``/``hf_repo`` the service can pull
from (``hf``). Invalid directories are logged and skipped, never fatal.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .manifest import Manifest
from .model import ModelSpec

logger = logging.getLogger("camaron.discovery")

_ONNX_SUFFIX = ".onnx"


def _has_onnx(model_dir: Path) -> bool:
    return any(p.suffix == _ONNX_SUFFIX and p.is_file() for p in model_dir.rglob("*"))


def discover(model_path: Path) -> list[ModelSpec]:
    if not model_path.is_dir():
        logger.warning("model path %r does not exist; no models available", model_path)
        return []

    specs: list[ModelSpec] = []
    for entry in sorted(model_path.iterdir()):
        if not entry.is_dir():
            continue
        manifest_file = entry / "manifest.yaml"
        if not manifest_file.exists():
            continue  # not a model directory; silently ignored

        try:
            manifest = Manifest.load(manifest_file, entry.name)
        except Exception as exc:  # bad manifest must not kill discovery of siblings
            logger.warning("skipping %r: invalid manifest (%s)", entry.name, exc)
            continue

        if _has_onnx(entry):
            specs.append(ModelSpec(entry.name, entry, "local", manifest))
        elif manifest.hf_repo:
            specs.append(ModelSpec(entry.name, entry, "hf", manifest))
        else:
            logger.warning("skipping %r: no local ONNX files and no hf_url", entry.name)

    return specs
