"""Per-model ``manifest.yaml`` parsing and validation.

The manifest carries everything the service needs that is *not* part of the ONNX
graph: model identity, task family, voice mapping (TTS), inference parameter bounds,
and pointers (HF repo id) to assets that live outside the ONNX file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import yaml


class Task(StrEnum):
    """Model family. Determines which inference handler is instantiated."""

    ASR = "asr"
    TTS = "tts"


_HUGGINGFACE_PREFIXES = ("https://huggingface.co/", "https://hf.co/")


def repo_id_from_url(url: str) -> str:
    """Reduce a Hugging Face URL (or bare repo id) to ``org/name``."""
    url = url.strip()
    for prefix in _HUGGINGFACE_PREFIXES:
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    url = url.rstrip("/")
    for sep in ("/tree/", "/resolve/", "/blob/"):
        if sep in url:
            url = url.split(sep)[0]
    if not url or url.count("/") < 1:
        raise ValueError(f"not a Hugging Face repo id or URL: {url!r}")
    return url


@dataclass
class ParamBounds:
    """Bounds for an inference knob, exposed to the API for input validation."""

    name: str
    default: float
    min: float
    max: float

    def clamp(self, value: float | None) -> float:
        if value is None:
            return self.default
        return max(self.min, min(self.max, float(value)))


@dataclass
class Manifest:
    """Validated, typed view of a single model's ``manifest.yaml``."""

    name: str
    task: Task
    hf_repo: str | None = None       # ONNX + tokenizer assets (model + tokenizer.json)
    voice_repo: str | None = None    # TTS: where voices/*.npy|.pt|.bin live (defaults to hf_repo)
    languages: list[str] = field(default_factory=list)
    voices: list[str] = field(default_factory=list)
    voice_map: dict[str, str] = field(default_factory=dict)
    voice_language: dict[str, str] = field(default_factory=dict)  # voice id -> phonemizer language
    default_voice: str | None = None
    sample_rate: int | None = None   # audio output sample rate (TTS) or input (ASR)
    params: dict[str, ParamBounds] = field(default_factory=dict)
    onnx_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str, folder_name: str) -> Manifest:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.parse(dict(raw), folder_name)

    @classmethod
    def parse(cls, data: dict[str, Any], folder_name: str) -> Manifest:
        if "task" not in data or data["task"] not in ("asr", "tts"):
            raise ValueError("manifest: 'task' must be 'asr' or 'tts'")
        task = Task(data["task"])

        hf_repo = None
        href = data.get("hf_url") or data.get("hf_repo")
        if href:
            hf_repo = repo_id_from_url(str(href))

        pmap: dict[str, ParamBounds] = {}
        for pname, p in (data.get("params") or {}).items():
            p = p or {}
            pmap[pname] = ParamBounds(
                name=str(pname),
                default=float(p.get("default", 0.0)),
                min=float(p.get("min", 0.0)),
                max=float(p.get("max", 1.0)),
            )

        onnx_files = data.get("onnx_files") or []
        if isinstance(onnx_files, (list, tuple)):
            onnx_names = [f.get("name") if isinstance(f, dict) else str(f) for f in onnx_files]
            onnx_names = [n for n in onnx_names if n]
        else:
            onnx_names = [str(onnx_files)]

        return cls(
            name=str(data.get("name") or folder_name),
            task=task,
            hf_repo=hf_repo,
            voice_repo=data.get("voice_repo"),
            languages=list(data.get("languages") or []),
            voices=list(data.get("voices") or []),
            voice_map={str(k): str(v) for k, v in (data.get("voice_map") or {}).items()},
            voice_language={
                str(k): str(v).lower() for k, v in (data.get("voice_language") or {}).items()
            },
            default_voice=data.get("default_voice"),
            sample_rate=int(data["sample_rate"]) if data.get("sample_rate") else None,
            params=pmap,
            onnx_files=onnx_names,
            metadata=dict(data.get("metadata") or {}),
        )
