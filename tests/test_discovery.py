"""Unit: model discovery rules (no ONNX weight loading required)."""
import textwrap

import pytest

from src.discovery import discover

ASR_MANIFEST = textwrap.dedent("""
    name: dummy-asr
    task: asr
""")

TTS_MANIFEST = textwrap.dedent("""
    name: dummy-tts
    task: tts
    voices: [v1, v2]
    default_voice: v1
""")

HF_MANIFEST = textwrap.dedent("""
    name: hf-model
    task: asr
    hf_url: https://huggingface.co/org/repo
""")


@pytest.fixture
def models_root(tmp_path):
    root = tmp_path / "models"
    # valid local asr
    d = root / "asr-local"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"\x00\x01")
    (d / "manifest.yaml").write_text(ASR_MANIFEST)
    # valid local tts
    d = root / "tts-local"
    d.mkdir(parents=True)
    (d / "onnx").mkdir()
    (d / "onnx" / "model.onnx").write_bytes(b"\x00\x02")
    (d / "manifest.yaml").write_text(TTS_MANIFEST)
    # valid hf (no local onnx)
    d = root / "hf-model"
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(HF_MANIFEST)
    # invalid: no manifest
    d = root / "no-manifest"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"\x00\x03")
    # invalid: bad manifest
    d = root / "bad-manifest"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"\x00\x04")
    (d / "manifest.yaml").write_text("name: x\n")  # no task
    # invalid: manifest present but no local onnx and no hf_url
    d = root / "orphan"
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text("name: orphan\ntask: asr\n")
    # a stray file at the top
    (root / "README.txt").write_text("ignore me")
    return root


def test_discovers_valid_models(models_root):
    specs = discover(models_root)
    names = {s.api_id for s in specs}
    assert names == {"dummy-asr", "dummy-tts", "hf-model"}


def test_invalid_directories_skipped(models_root):
    specs = discover(models_root)
    names = {s.folder_name for s in specs}
    assert "no-manifest" not in names
    assert "bad-manifest" not in names
    assert "orphan" not in names


def test_missing_path_returns_empty(models_root):
    assert discover(models_root / "does-not-exist") == []
