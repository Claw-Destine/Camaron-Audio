""">Fixtures for the test suite.

E2E tests drive the real ASGI app *through the official OpenAI client*, exactly how a
user would, to prove wire-compatibility. The app is exercised in-process (no server
process / port management needed); the OpenAI client's HTTP transport is swapped for
an ASGI transport bound to ``app``.
"""
from __future__ import annotations

import os
import pathlib

import openai
import pytest
from starlette.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = os.environ.get("CAMARON_MODEL_PATH", str(ROOT / "models"))


def _model_ready(*parts: str) -> bool:
    return pathlib.Path(MODELS, *parts).exists()


def _models_ready() -> bool:
    return (
        _model_ready("whisper-tiny.en", "manifest.yaml")
        and _model_ready("kokoro-82m-v1.0", "manifest.yaml")
    )


requires_models = pytest.mark.skipif(not _models_ready(),
                                     reason="run `python scripts/prepare_models.py` first")

# Optional assets: multilingual whisper / non-English voice tables (all prepared by
# scripts/prepare_models.py). Tests stay green on a minimal install by skipping only
# when a specific asset is missing.
requires_ml_whisper = pytest.mark.skipif(
    not _model_ready("whisper-tiny", "manifest.yaml"),
    reason="multilingual whisper-tiny not prepared (run `python scripts/prepare_models.py`)")
requires_es_voice = pytest.mark.skipif(
    not _model_ready("kokoro-82m-v1.0", "voices", "ef_dora.npy"),
    reason="es voice table not prepared (run `python scripts/prepare_models.py`)")
requires_zh_voice = pytest.mark.skipif(
    not _model_ready("kokoro-82m-v1.0", "voices", "zf_xiaobei.npy"),
    reason="zh voice table not prepared (run `python scripts/prepare_models.py`)")


@pytest.fixture(scope="session")
def app():
    import onnxruntime  # noqa: F401

    from src.__main__ import build_app

    config, registry, app = build_app(["--model-path", MODELS])
    yield app
    registry.shutdown()


@pytest.fixture(scope="session")
def http(app):
    # starlette TestClient is an httpx.Client that speaks ASGI in-process.
    return TestClient(app)


@pytest.fixture(scope="session")
def openai_client(app):
    importorskip_openapi()
    return openai.OpenAI(
        base_url="http://testserver/v1", api_key="test", http_client=TestClient(app)
    )


def importorskip_openapi() -> None:
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai client not installed")


@pytest.fixture(scope="session")
def tts_wav(http, tmp_path_factory):
    """A generated WAV used as the input for the TTS->STT closed-loop tests."""
    r = http.post("/v1/audio/speech", json={
        "model": "kokoro-82m-v1.0",
        "input": "The quick brown fox jumps over the lazy dog.",
        "voice": "af_heart",
        "response_format": "wav",
    })
    r.raise_for_status()
    path = tmp_path_factory.mktemp("audio", numbered=True) / "sample.wav"
    path.write_bytes(r.content)
    return path  # pathlib.Path: accepted by the OpenAI client and by open()
