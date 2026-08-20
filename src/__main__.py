"""Entry point: ``python -m src`` (or the ``camaron-audio`` script).

Loads config, discovers models, builds the FastAPI app. ``build_app`` is split out so
the test suite (and ``uvicorn src.__main__:app``) can reuse it.
"""
from __future__ import annotations

from .api.app import create_app
from .config import load_config
from .discovery import discover
from .inference.engine import Registry


def build_app(argv: list[str] | None = None):
    config = load_config(argv)
    specs = discover(config.model_path)
    registry = Registry(specs, pool_size=config.thread_pool_size)
    return config, registry, create_app(config, registry)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    import uvicorn

    config, registry, app = build_app(argv)
    try:
        uvicorn.run(app, host=config.host, port=config.port,
                    log_level=config.log_level.lower())
    finally:
        registry.shutdown()


if __name__ == "__main__":
    main()
