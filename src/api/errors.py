"""OpenAI-compatible error envelope, auth, and exception-to-HTTP mapping.

Every error the service returns uses OpenAI's body shape::

    {"error": {"message": "...", "type": "...", "code": null}}

so clients that already parse OpenAI errors keep working.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..audio import CodecError
from ..config import Config
from ..inference.engine import ModelNotFound
from ..inference.tts import TTSBackendError, VoiceError

_INVALID = "invalid_request_error"
_AUTH = "authentication_error"
_SERVER = "server_error"

_STATUS_TYPE = {400: _INVALID, 401: _AUTH, 404: "not_found_error", 422: _INVALID, 500: _SERVER}


def error_response(message: str, http_status: int = 400, type: str | None = None,
                   code: str | None = None) -> JSONResponse:
    err = {
        "message": message,
        "type": type or _STATUS_TYPE.get(http_status, _INVALID),
        "code": code,
    }
    return JSONResponse(status_code=http_status, content={"error": err})


class OpenAIError(Exception):
    """Raised by handlers to produce a custom OpenAI-style error."""

    def __init__(self, message: str, http_status: int = 400, type: str | None = None,
                 code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.type = type
        self.code = code


def require_auth(request: Request) -> None:
    cfg: Config = request.app.state.config
    if not cfg.api_key:
        return  # auth disabled (local dev)
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != cfg.api_key:
        raise OpenAIError("Invalid API key.", http_status=401, type=_AUTH, code="invalid_api_key")


def register_exception_handlers(app) -> None:
    @app.exception_handler(OpenAIError)
    async def _openai_error(_, exc: OpenAIError):
        return error_response(exc.message, exc.http_status, exc.type, exc.code)

    @app.exception_handler(ModelNotFound)
    async def _not_found(_, exc: ModelNotFound):
        return error_response(f"The model '{exc.name}' does not exist.", 404)

    @app.exception_handler(VoiceError)
    async def _voice(_, exc: VoiceError):
        return error_response(str(exc), 400)

    @app.exception_handler(CodecError)
    async def _codec(_, exc: CodecError):
        return error_response(str(exc), 400, code="unsupported_codec")

    @app.exception_handler(RequestValidationError)
    async def _validation(_, exc: RequestValidationError):
        return error_response(f"Invalid request: {exc.errors()}", 400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_, exc: StarletteHTTPException):
        return error_response(str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _server(_, exc: Exception):
        if isinstance(exc, TTSBackendError):
            return error_response(str(exc), 500, type=_SERVER)
        app.state.logger.exception("unhandled error: %s", exc)
        return error_response("An internal error occurred.", 500, type=_SERVER)
