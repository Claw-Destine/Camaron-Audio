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


def _models_ready() -> bool:
    return (
        pathlib.Path(MODELS, "whisper-tiny.en", "manifest.yaml").exists()
        and pathlib.Path(MODELS, "kokoro-82m-v1.0", "manifest.yaml").exists()
    )


requires_models = pytest.mark.skipif(not _models_ready(),
                                     reason="run `python scripts/prepare_models.py` first")


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
