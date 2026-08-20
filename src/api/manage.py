"""Manage endpoints: model registry browsing, installed model listing, model fetch."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query, Request

from ..manifest import Manifest
from ..model import ModelSpec
from . import run_in_pool

logger = logging.getLogger("camaron.manage")
router = APIRouter(prefix="/manage", tags=["manage"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REGISTRY_FILE = _PROJECT_ROOT / "hf_registry.yaml"


# ── helpers -----------------------------------------------------------------

def _load_registry() -> list[dict[str, Any]]:
    if not _REGISTRY_FILE.is_file():
        logger.warning("hf_registry.yaml not found at %s", _REGISTRY_FILE)
        return []
    with open(_REGISTRY_FILE) as fh:
        return yaml.safe_load(fh) or []


def _is_installed(model_root: Path, name: str) -> bool:
    d = model_root / name
    if not (d / "manifest.yaml").is_file():
        return False
    return any(p.suffix == ".onnx" and p.is_file() for p in d.rglob("*"))


def _manifest_data(entry: dict[str, Any]) -> dict[str, Any]:
    """Build manifest.yaml content from a registry entry (excluding download-only fields)."""
    skip = {"download_files", "description", "voice_repo"}
    data = {k: v for k, v in entry.items() if k not in skip and v is not None}
    if "hf_repo" in data:
        data["hf_url"] = f"https://huggingface.co/{data.pop('hf_repo')}"
    return data


def _download_blocking(entry: dict[str, Any], model_dir: Path) -> None:
    """Blocking file downloads (run in the thread pool via ``run_in_pool``)."""
    from huggingface_hub import hf_hub_download

    for f in entry.get("download_files", []):
        target = model_dir / f
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            hf_hub_download(entry["hf_repo"], f, local_dir=str(model_dir))

    voice_repo = entry.get("voice_repo")
    voices = entry.get("voices") or []
    if not voice_repo or not voices:
        return

    voices_dir = model_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        npy = voices_dir / f"{voice}.npy"
        if npy.exists():
            continue
        pt_path = hf_hub_download(voice_repo, f"voices/{voice}.pt")
        import numpy as np
        import torch

        table = torch.load(pt_path, map_location="cpu", weights_only=True)
        np.save(npy, table.numpy().astype("float32"))


def _serialize_manifest(m: Manifest) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": m.name,
        "task": m.task.value,
        "languages": m.languages,
        "voices": m.voices,
        "default_voice": m.default_voice,
        "sample_rate": m.sample_rate,
    }
    if m.params:
        out["params"] = {
            k: {"default": p.default, "min": p.min, "max": p.max}
            for k, p in m.params.items()
        }
    if m.voice_map:
        out["voice_map"] = m.voice_map
    if m.metadata:
        out["metadata"] = m.metadata
    return out


# ── endpoints ---------------------------------------------------------------


@router.get("/hf/models")
def hf_models(request: Request, type: str | None = Query(None, pattern="^(asr|tts)$")):
    """List models available in the HF registry, each with its install status."""
    model_root: Path = request.app.state.config.model_path
    out: list[dict[str, Any]] = []
    for entry in _load_registry():
        name = entry.get("name", "?")
        task = entry.get("task", "?")
        if type and task != type:
            continue
        out.append({
            "name": name,
            "task": task,
            "description": entry.get("description", ""),
            "status": "installed" if _is_installed(model_root, name) else "not_installed",
        })
    return out


@router.get("/models")
def installed_models(request: Request, type: str | None = Query(None, pattern="^(asr|tts)$")):
    """List manifests of models currently loaded in the active registry."""
    registry = request.app.state.registry
    out: list[dict[str, Any]] = []
    for name in registry.list():
        model = registry.get(name)
        if model is None:
            continue
        if type and model.manifest.task.value != type:
            continue
        out.append(_serialize_manifest(model.manifest))
    return out


@router.post("/hf/fetch/{name}")
async def fetch_model(name: str, request: Request):
    """Download, prepare, and hot-load a model from Hugging Face."""
    config = request.app.state.config
    registry = request.app.state.registry

    entry = next((e for e in _load_registry() if e.get("name") == name), None)
    if entry is None:
        raise HTTPException(404, f"model '{name}' not found in the HF registry")

    model_dir = config.model_path / name

    if _is_installed(config.model_path, name):
        if registry.get(name) is None:
            manifest = Manifest.load(str(model_dir / "manifest.yaml"), name)
            registry.add_model(ModelSpec(name, model_dir, "local", manifest))
        return {"status": "installed", "model": name}

    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        await run_in_pool(request, _download_blocking, entry, model_dir)
    except Exception as exc:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(500, f"download failed: {exc}") from exc

    manifest_data = _manifest_data(entry)
    with open(model_dir / "manifest.yaml", "w") as fh:
        yaml.safe_dump(manifest_data, fh, sort_keys=False)

    try:
        manifest = Manifest.parse(manifest_data, name)
        registry.add_model(ModelSpec(name, model_dir, "local", manifest))
    except Exception as exc:
        logger.exception("downloaded but failed to hot-load %s", name)
        raise HTTPException(500, f"model downloaded but failed to load: {exc}") from exc

    return {"status": "installed", "model": name}
