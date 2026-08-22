"""Unit tests for the /manage endpoints (HF registry, installed models, fetch)."""
from __future__ import annotations

import pytest
from conftest import requires_models

pytestmark = pytest.mark.e2e


class TestHfRegistry:
    """GET /manage/hf/models — the HF model catalogue."""

    def test_list_all(self, http):
        r = http.get("/manage/hf/models")
        assert r.status_code == 200
        models = r.json()
        assert isinstance(models, list) and len(models) >= 1
        for m in models:
            assert m["task"] in ("asr", "tts")
            assert m["status"] in ("installed", "not_installed")

    def test_filter_asr(self, http):
        r = http.get("/manage/hf/models", params={"type": "asr"})
        assert r.status_code == 200
        assert all(m["task"] == "asr" for m in r.json())
        assert len(r.json()) >= 1

    def test_filter_tts(self, http):
        r = http.get("/manage/hf/models", params={"type": "tts"})
        assert r.status_code == 200
        assert all(m["task"] == "tts" for m in r.json())

    def test_invalid_type_filter(self, http):
        r = http.get("/manage/hf/models", params={"type": "invalid"})
        assert r.status_code in (422, 400)


@requires_models
class TestInstalledModels:
    """GET /manage/models — manifests of models loaded in the active registry."""

    def test_list(self, http):
        r = http.get("/manage/models")
        assert r.status_code == 200
        models = r.json()
        names = {m["name"] for m in models}
        assert "whisper-tiny.en" in names
        assert "kokoro-82m-v1.0" in names

    def test_filter_asr(self, http):
        r = http.get("/manage/models", params={"type": "asr"})
        assert r.status_code == 200
        models = r.json()
        assert all(m["task"] == "asr" for m in models)
        # the EN-only model must be present; multilingual variants may be prepared too
        assert {m["name"] for m in models} >= {"whisper-tiny.en"}

    def test_filter_tts(self, http):
        r = http.get("/manage/models", params={"type": "tts"})
        assert r.status_code == 200
        models = r.json()
        assert all(m["task"] == "tts" for m in models)

    def test_tts_manifest_structure(self, http):
        r = http.get("/manage/models", params={"type": "tts"})
        model = r.json()[0]
        assert model["name"] == "kokoro-82m-v1.0"
        assert model["task"] == "tts"
        assert "af_heart" in model.get("voices", [])
        assert model.get("default_voice") == "af_heart"
        assert model.get("sample_rate") == 24000

    def test_asr_manifest_structure(self, http):
        r = http.get("/manage/models", params={"type": "asr"})
        model = next(m for m in r.json() if m["name"] == "whisper-tiny.en")
        assert model["task"] == "asr"
        assert model.get("languages") == ["en"]


class TestFetch:
    """POST /manage/hf/fetch/{name} — download and prepare a model."""

    def test_unknown_model_404(self, http):
        r = http.post("/manage/hf/fetch/this-does-not-exist")
        assert r.status_code == 404

    @requires_models
    def test_already_installed(self, http):
        r = http.post("/manage/hf/fetch/whisper-tiny.en")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "installed"
        assert body["model"] == "whisper-tiny.en"
