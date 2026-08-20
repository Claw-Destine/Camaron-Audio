"""Prepare the local ``models/`` directory for the two test models.

Downloads from Hugging Face (cached) and writes, under ``models/``:
  whisper-tiny.en/   -- Whisper ONNX (encoder + decoder, tokenizer, mel config)
  kokoro-82m-v1.0/   -- Kokoro ONNX (model + phoneme tokenizer + voice style tables)

Voice style tables are converted from the upstream ``.pt`` (torch) into ``.npy``
so the *running* service never needs torch -- it loads them with plain numpy.
The end-user equivalent of one step is: ``python tools/convert_voices.py``.

Idempotent: existing files are left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"

WHISPER_REPO = "onnx-community/whisper-tiny.en"
KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
VOICE_REPO = "hexgrad/Kokoro-82M"

WHISPER_FILES = [
    "onnx/encoder_model.onnx",
    "onnx/decoder_model.onnx",
    "tokenizer.json",
    "preprocessor_config.json",
    "config.json",
    "added_tokens.json",
    "special_tokens_map.json",
]
KOKORO_FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]

KOKORO_VOICE_IDS = ["af_heart", "af_sarah", "af_nicole", "am_michael", "bf_emma", "bm_george"]

WHISPER_MANIFEST = {
    "name": "whisper-tiny.en",
    "task": "asr",
    "languages": ["en"],
    "onnx": "onnx/decoder_model.onnx",
    "onnx_files": ["onnx/encoder_model.onnx", "onnx/decoder_model.onnx"],
    "hf_url": f"https://huggingface.co/{WHISPER_REPO}",
    "params": {
        "temperature": {"default": 0.0, "min": 0.0, "max": 2.0},
        "top_p": {"default": 1.0, "min": 0.0, "max": 1.0},
    },
}

KOKORO_MANIFEST = {
    "name": "kokoro-82m-v1.0",
    "task": "tts",
    "sample_rate": 24000,
    "voices": KOKORO_VOICE_IDS,
    "voice_map": {
        "alloy": "af_heart",
        "nova": "af_nicole",
        "fable": "af_sarah",
        "echo": "am_michael",
        "onyx": "bf_emma",
        "shimo": "bm_george",
    },
    "default_voice": "af_heart",
    "hf_url": f"https://huggingface.co/{KOKORO_REPO}",
    "params": {
        "temperature": {"default": 0.0, "min": 0.0, "max": 1.0},
        "top_p": {"default": 1.0, "min": 0.0, "max": 1.0},
    },
    "metadata": {"license": "Apache-2.0"},
}


def _repo_files(repo: str, dest: Path, files: list[str]) -> None:
    (dest / "onnx").mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest / f
        if target.exists():
            continue
        hf_hub_download(repo, f, local_dir=str(dest))  # mirrored under dest/<f>
        print(f"  {dest.name}/{f}")


def _write_manifest(dest: Path, manifest: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def _convert_voices(dest_voices: Path, voices: list[str]) -> None:
    import torch

    dest_voices.mkdir(parents=True, exist_ok=True)
    for v in voices:
        dst = dest_voices / f"{v}.npy"
        if dst.exists():
            continue
        src = hf_hub_download(VOICE_REPO, f"voices/{v}.pt")
        table = torch.load(src, map_location="cpu", weights_only=True).numpy().astype(np.float32)
        np.save(dst, table)
        print(f"  kokoro voices/{v}.npy  shape={table.shape}")


def main() -> None:
    print(f"Preparing models under {MODELS}")
    whisper_dir = MODELS / "whisper-tiny.en"
    _repo_files(WHISPER_REPO, whisper_dir, WHISPER_FILES)
    _write_manifest(whisper_dir, WHISPER_MANIFEST)

    kokoro_dir = MODELS / "kokoro-82m-v1.0"
    _repo_files(KOKORO_REPO, kokoro_dir, KOKORO_FILES)
    _write_manifest(kokoro_dir, KOKORO_MANIFEST)
    _convert_voices(kokoro_dir / "voices", KOKORO_VOICE_IDS)
    print("done")


if __name__ == "__main__":
    main()
