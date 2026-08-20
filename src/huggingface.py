"""Hugging Face asset resolution.

Everything flows through the standard HF cache (``~/.cache/huggingface`` by default,
honouring the ``HF_HOME`` / ``HF_HUB_CACHE`` env vars). Files are only pulled when a
local model directory lacks them, and only the files actually needed (via
``allow_patterns``) so we do not download every quantization variant.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .manifest import repo_id_from_url


def resolve_repo(repo: str, allow_patterns: list[str] | None = None) -> Path:
    """Return a local directory containing the repo's files (downloading as needed)."""
    repo = repo_id_from_url(repo) if "/" in repo and "://" in repo else repo
    snapshot = snapshot_download(repo, allow_patterns=allow_patterns)
    return Path(snapshot)


def resolve_file(repo: str, filename: str) -> Path:
    """Return a local path for a single file in a repo (downloading as needed)."""
    repo = repo_id_from_url(repo) if "://" in repo else repo
    return Path(hf_hub_download(repo, filename))
