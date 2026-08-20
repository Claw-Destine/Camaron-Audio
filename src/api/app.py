"""FastAPI application factory: binds config + model registry and wires the API."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import Config
from ..inference.engine import Registry
from . import manage as _manage
from . import models as _models
from . import speech as _speech
from . import transcriptions as _transcriptions
from . import translations as _translations
from .errors import register_exception_handlers, require_auth


def create_app(config: Config, registry: Registry) -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.api_key:
        logging.getLogger("camaron").warning(
            "running UNAUTHENTICATED (CAMARON_API_KEY not set) — fine for local dev"
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        registry.shutdown()

    app = FastAPI(
        title="Camaron Audio",
        version=__version__,
        description="OpenAI-compatible ONNX audio service (STT/TTS).",
        dependencies=[Depends(require_auth)],
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.registry = registry
    app.state.logger = logging.getLogger("camaron")

    # Self-hosted: allow the in-browser test app (and any local client) to call it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(_models.router)
    app.include_router(_speech.router)
    app.include_router(_transcriptions.router)
    app.include_router(_translations.router)
    app.include_router(_manage.router)

    register_exception_handlers(app)
    return app
