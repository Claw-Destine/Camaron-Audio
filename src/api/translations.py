"""POST /v1/audio/translations — speech to *English* text.

Identical to transcriptions with ``language`` forced server-side to "en". Kept as a
separate route so the OpenAI paths behave exactly as expected.
"""
from fastapi import APIRouter, File, Form, Request, UploadFile

from . import transcriptions

router = APIRouter()


@router.post("/v1/audio/translations")
async def translations(
    request: Request,
    model: str = Form(...),
    file: UploadFile = File(...),  # noqa: B008  (FastAPI DI)
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
):
    return await transcriptions.handle_transcription(
        request, model, file, language="en", response_format=response_format,
        temperature=temperature, top_p=top_p,
    )
