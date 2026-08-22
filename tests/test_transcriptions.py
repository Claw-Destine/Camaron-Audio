"""E2E: POST /v1/audio/transcriptions + /translations, as a TTS->STT closed loop.

The input audio is produced by the Kokoro TTS endpoint, then sent back to Whisper
-- so a passing run proves the whole service is mutually compatible end to end.
"""
import io

import pytest
from conftest import requires_es_voice, requires_ml_whisper, requires_models

pytestmark = [pytest.mark.e2e, requires_models]

TEXT = "brown fox"


def _upload(path):
    return ("sample.wav", open(path, "rb"), "audio/wav")


def test_transcription_openai_client(openai_client, tts_wav):
    res = openai_client.audio.transcriptions.create(model="whisper-tiny.en", file=tts_wav)
    assert TEXT in res.text.lower()


def test_transcription_text_raw(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny.en", "response_format": "text"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert TEXT in r.text.lower()


def test_transcription_json_raw(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny.en", "response_format": "json"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert TEXT in body["text"].lower()


def test_transcription_srt_raw(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny.en", "response_format": "srt"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 200
    body = r.text
    assert body.lstrip().startswith("1")
    assert "-->" in body
    assert TEXT in body.lower()


def test_transcription_verbose_json(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny.en", "response_format": "verbose_json"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["duration"] > 0


def test_translation_openai_client(openai_client, tts_wav):
    res = openai_client.audio.translations.create(model="whisper-tiny.en", file=tts_wav)
    assert TEXT in res.text.lower()


def test_translation_raw(http, tts_wav):
    r = http.post("/v1/audio/translations",
                  data={"model": "whisper-tiny.en", "response_format": "json"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 200
    assert TEXT in r.json()["text"].lower()


# --- error paths -------------------------------------------------------------
def test_transcription_wrong_model_400(http, tts_wav):
    r = http.post("/v1/audio/transcriptions", data={"model": "kokoro-82m-v1.0"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 400


def test_transcription_unknown_model_404(http, tts_wav):
    r = http.post("/v1/audio/transcriptions", data={"model": "nope"},
                  files={"file": ("sample.wav", open(tts_wav, "rb"), "audio/wav")})
    assert r.status_code == 404


def test_transcription_missing_file_400(http):
    r = http.post("/v1/audio/transcriptions", data={"model": "whisper-tiny.en"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


# --- language selection (multilingual whisper-tiny) --------------------------
@requires_ml_whisper
def test_multilingual_detection(http, tts_wav):
    """No `language` given -> the model detects it; reported language is the
    detected one and the task is "transcribe"."""
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny", "response_format": "verbose_json"},
                  files={"file": _upload(tts_wav)})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["task"] == "transcribe"
    assert TEXT in body["text"].lower()


@requires_ml_whisper
def test_multilingual_forced_language(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny", "language": "en",
                        "response_format": "json"},
                  files={"file": _upload(tts_wav)})
    assert r.status_code == 200
    assert TEXT in r.json()["text"].lower()


@requires_ml_whisper
def test_translations_multilingual(http, tts_wav):
    """The translations endpoint forces language=en + task=translate server-side."""
    r = http.post("/v1/audio/translations",
                  data={"model": "whisper-tiny", "response_format": "verbose_json"},
                  files={"file": _upload(tts_wav)})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["task"] == "translate"


@requires_ml_whisper
def test_invalid_language_400_ml(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny", "language": "gl"},
                  files={"file": _upload(tts_wav)})
    assert r.status_code == 400
    assert "language" in r.json()["error"]["message"].lower()


def test_invalid_language_400_en_only(http, tts_wav):
    r = http.post("/v1/audio/transcriptions",
                  data={"model": "whisper-tiny.en", "language": "fr"},
                  files={"file": _upload(tts_wav)})
    assert r.status_code == 400


# --- cross-language closed loop: TTS(es) -> STT(es) --------------------------
@requires_ml_whisper
@requires_es_voice
def test_spanish_t2t_loop(http):
    """Spanish synthesized by Kokoro, transcribed back: both explicit language and
    auto-detection must land on Spanish."""
    r = http.post("/v1/audio/speech",
                  json={"model": "kokoro-82m-v1.0",
                        "input": "Hola mundo, esta es una prueba de voz.",
                        "voice": "ef_dora", "response_format": "wav"})
    assert r.status_code == 200

    for data in (
        {"model": "whisper-tiny", "response_format": "verbose_json", "language": "es"},
        {"model": "whisper-tiny", "response_format": "verbose_json"},  # auto-detect
    ):
        r2 = http.post("/v1/audio/transcriptions", data=data,
                       files={"file": ("es.wav", io.BytesIO(r.content), "audio/wav")})
        assert r2.status_code == 200
        body = r2.json()
        assert body["language"] == "es"
        assert "mundo" in body["text"].lower()
