"""POST /v1/audio/speech — text to speech.

Request body (JSON), matching OpenAI::

    {"model": "...", "input": "...", "voice": "...",
     "response_format": "wav|pcm|mp3|flac", "speed": 1.0,
     "temperature": 0.8, "top_p": 0.9}

Kokoro's ONNX graph has no sampling knob, so temperature/top_p are accepted for
OpenAI compatibility but are ignored; only ``speed`` and ``voice`` drive inference.
Returns the full audio in a single response.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..audio import encode_audio
from ..model import Task
from . import get_model, run_in_pool
from .errors import OpenAIError

router = APIRouter()


@router.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    body = await request.json()

    model = get_model(request, str(body.get("model")))
    if model.manifest.task != Task.TTS:
        raise OpenAIError(f"The model '{body.get('model')}' is not a TTS model.", 400)

    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise OpenAIError("'input' must be a non-empty string.", 400)

    voice = body.get("voice")
    fmt = str(body.get("response_format") or "wav")
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError) as exc:
        raise OpenAIError("'speed' must be a number.", 400) from exc
    speed = min(max(speed, 0.25), 4.0)

    waveform = await run_in_pool(request, model.synthesize, text, voice, speed)
    data, mime = encode_audio(waveform, model.sample_rate, fmt)
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": 'attachment; filename="speech"'+_ext(fmt)})


def _ext(fmt: str) -> str:
    return {"wav": ".wav", "pcm": ".pcm", "mp3": ".mp3", "flac": ".flac"}.get(fmt, "")
