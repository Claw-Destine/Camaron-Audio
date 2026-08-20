"""Unit: manifest parsing, HF-URL reduction, and parameter clamping (no model loading)."""
import pytest

from src.manifest import Manifest, ParamBounds, repo_id_from_url
from src.model import Task


def test_parse_asr_defaults_to_folder_name():
    m = Manifest.parse({"task": "asr"}, folder_name="my-folder")
    assert m.task == Task.ASR
    assert m.name == "my-folder"


def test_missing_task_rejected():
    with pytest.raises(ValueError):
        Manifest.parse({"name": "x"}, folder_name="x")


def test_bad_task_rejected():
    with pytest.raises(ValueError):
        Manifest.parse({"task": "embeddings"}, folder_name="x")


def test_tts_voice_fields():
    m = Manifest.parse(
        {
            "task": "tts",
            "voices": ["v1", "v2"],
            "voice_map": {"alloy": "v1", "nova": "v2"},
            "default_voice": "v1",
            "params": {"temperature": {"default": 0.5, "min": 0.0, "max": 1.0}},
        },
        folder_name="kokoro",
    )
    assert m.voices == ["v1", "v2"]
    assert m.voice_map == {"alloy": "v1", "nova": "v2"}
    assert m.default_voice == "v1"
    assert isinstance(m.params["temperature"], ParamBounds)


def test_hf_url_repo_reductions():
    m = Manifest.parse({"task": "asr", "hf_url": "https://huggingface.co/org/repo/tree/main"},
                       folder_name="x")
    assert m.hf_repo == "org/repo"
    assert repo_id_from_url("https://hf.co/org/name") == "org/name"
    assert repo_id_from_url("org/name") == "org/name"


def test_bad_hf_url_rejected():
    with pytest.raises(ValueError):
        Manifest.parse({"task": "asr", "hf_url": "org"}, folder_name="x")


def test_param_clamp():
    p = ParamBounds(name="t", default=0.5, min=0.0, max=1.0)
    assert p.clamp(None) == 0.5
    assert p.clamp(2.0) == 1.0
    assert p.clamp(-1.0) == 0.0


def test_sample_rate_parsed():
    m = Manifest.parse({"task": "tts", "sample_rate": 24000}, folder_name="x")
    assert m.sample_rate == 24000
