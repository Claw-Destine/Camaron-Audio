"""POST /v1/audio/transcriptions — speech to text.

Also hosts the shared logic that /v1/audio/translations reuses (it is identical with
``language`` forced to English). Timing is not recovered in our greedy Whisper decode,
so the ``srt``/``vtt``/``verbose_json`` shapes expose a single cue spanning the input
duration rather than fabricated per-word timestamps.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from ..audio import decode_audio
from ..model import Task
from . import get_model, run_in_pool

router = APIRouter()

_FORMATS = ("text", "json", "verbose_json", "srt", "vtt")


async def handle_transcription(
    request: Request,
    model: str,
    file: UploadFile,
    language: str | None,
    response_format: str = "json",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> PlainTextResponse | JSONResponse:
    m = get_model(request, model)
    if m.manifest.task != Task.ASR:
        from .errors import OpenAIError
        raise OpenAIError(f"The model '{model}' is not an ASR model.", 400)

    if response_format not in _FORMATS:
        response_format = "json"

    data = await file.read()
    audio, sample_rate = decode_audio(data, file.filename)
    duration = len(audio) / sample_rate if sample_rate else 0.0

    text = await run_in_pool(
        request, m.transcribe, audio, sample_rate, 2048, temperature, top_p
    )
    return _format(response_format, text, language, duration)


def _format(fmt: str, text: str, language: str | None, duration: float):
    if fmt == "text":
        return PlainTextResponse(text, media_type="text/plain")
    if fmt == "srt":
        return PlainTextResponse(_srt(text, duration), media_type="text/plain")
    if fmt == "vtt":
        return PlainTextResponse(_vtt(text, duration), media_type="text/plain")
    if fmt == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": language or "en",
            "duration": round(duration, 3),
            "text": text,
        })
    return JSONResponse({"text": text})


def _ts(seconds: float, sep: str) -> str:
    h = int(seconds // 3600)
    m = int(seconds // 60 % 60)
    s = int(seconds % 60)
    frac = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{frac:03d}"


def _srt(text: str, duration: float) -> str:
    return f"1\n00:00:00,000 --> {_ts(max(duration, 0.1), ',')}\n{text.strip()}\n"


def _vtt(text: str, duration: float) -> str:
    return f"WEBVTT\n\n00:00:00.000 --> {_ts(max(duration, 0.1), '.')}\n{text.strip()}\n"


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    model: str = Form(...),
    # B008: FastAPI dependency injection (File/... default)
    file: UploadFile = File(...),  # noqa: B008
    language: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
):
    return await handle_transcription(
        request, model, file, language, response_format, temperature, top_p
    )
