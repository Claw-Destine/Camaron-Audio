"""E2E: POST /v1/audio/speech (TTS) through the OpenAI client."""
import io
import wave

import pytest
from conftest import requires_models

pytestmark = [pytest.mark.e2e, requires_models]


def test_speech_wav_openai_client(openai_client):
    resp = openai_client.audio.speech.create(
        model="kokoro-82m-v1.0",
        input="The quick brown fox jumps over the lazy dog.",
        voice="af_heart",
        response_format="wav",
    )
    audio = resp.content
    assert len(audio) > 1000
    with wave.open(io.BytesIO(audio)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2        # 16-bit
        assert w.getnframes() > 24000       # >1 s of audio at 24 kHz


def test_speech_pcm_raw(http):
    r = http.post(
        "/v1/audio/speech",
        json={"model": "kokoro-82m-v1.0", "input": "Hello world.",
              "voice": "am_michael", "response_format": "pcm"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/pcm"
    assert len(r.content) % 2 == 0  # 16-bit PCM


def test_speech_unknown_model_404(http):
    r = http.post("/v1/audio/speech", json={"model": "nope", "input": "hi"})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_speech_bad_voice_400(http):
    r = http.post("/v1/audio/speech",
                  json={"model": "kokoro-82m-v1.0", "input": "hi", "voice": "no_such_voice"})
    assert r.status_code == 400


def test_speech_missing_input_400(http):
    r = http.post("/v1/audio/speech", json={"model": "kokoro-82m-v1.0"})
    assert r.status_code == 400


def test_speech_asr_model_rejected(http):
    r = http.post("/v1/audio/speech", json={"model": "whisper-tiny.en", "input": "hi"})
    assert r.status_code == 400
