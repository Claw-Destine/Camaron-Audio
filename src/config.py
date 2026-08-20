"""Configuration: environment variables overridden by CLI flags.

Precedence: ``flag > CAMARON_* env var > built-in default``.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: Path
    host: str
    port: int
    api_key: str | None
    thread_pool_size: int
    log_level: str


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def load_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(prog="camaron-audio", description=__doc__)
    parser.add_argument("--model-path", default=None, help="Model discovery root")
    parser.add_argument("--host", default=None, help="Bind address")
    parser.add_argument("--port", type=int, default=None, help="HTTP port")
    parser.add_argument("--api-key", default=None, help="Bearer token (omit to disable auth)")
    parser.add_argument(
        "--thread-pool-size", type=int, default=None, help="Inference worker threads"
    )
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path or _env("CAMARON_MODEL_PATH") or "./models")
    host = args.host or _env("CAMARON_HOST") or "0.0.0.0"
    port = args.port if args.port is not None else int(_env("CAMARON_PORT") or 8080)
    api_key = args.api_key if args.api_key is not None else _env("CAMARON_API_KEY")
    pool = args.thread_pool_size if args.thread_pool_size is not None \
        else int(_env("CAMARON_THREAD_POOL_SIZE") or 4)
    level = args.log_level or _env("CAMARON_LOG_LEVEL") or "INFO"

    return Config(
        model_path=model_path,
        host=host,
        port=port,
        api_key=api_key,
        thread_pool_size=max(1, pool),
        log_level=level.upper(),
    )
