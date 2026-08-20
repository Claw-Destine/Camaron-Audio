""">Model</ and <Registry>: resolve a model's assets, build its inference handler,
and expose the shared session plus a thread pool for concurrency.

Design notes:
- ONNX Runtime sessions are created once per model and shared across request
  threads (ORT sessions are thread-safe). We therefore serialise *CPU* work to the
  pool rather than spinning up a session per request.
- Asset resolution honours the manifest: a local model dir wins; otherwise the
  referenced HF repo is pulled (only the files we need) into the standard cache.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..huggingface import resolve_repo
from ..model import ModelSpec, Task
from .asr import WhisperASR
from .tts import KokoroTTS

logger = logging.getLogger("camaron.engine")

_ASR_PATTERNS = [
    "onnx/encoder_model.onnx", "onnx/decoder_model.onnx",
    "encoder_model.onnx", "decoder_model.onnx",
    "tokenizer.json", "preprocessor_config.json",
    "added_tokens.json", "special_tokens_map.json", "config.json",
]
_TTS_PATTERNS = ["onnx/model.onnx", "model.onnx", "tokenizer.json", "config.json"]


class Model:
    """A single loaded model: resolved asset root + its family handler."""

    def __init__(self, spec: ModelSpec, providers: list[str] | None = None) -> None:
        self.spec = spec
        self.manifest = spec.manifest
        root, voice_dirs = self._resolve_assets(spec)
        if spec.manifest.task == Task.ASR:
            self.handler = WhisperASR(root, providers=providers)
        else:
            self.handler = KokoroTTS(root, self.manifest, voice_dirs, providers=providers)

    def _resolve_assets(self, spec: ModelSpec) -> tuple[Path, list[Path]]:
        if spec.source == "local":
            return spec.path, [spec.path]

        patterns = _ASR_PATTERNS if spec.manifest.task == Task.ASR else _TTS_PATTERNS
        root = resolve_repo(spec.manifest.hf_repo, patterns)
        voice_dirs = [spec.path]
        if (root / "voices").is_dir():
            voice_dirs.append(root)
        return root, voice_dirs

    # -- convenience surface -------------------------------------------------
    @property
    def sample_rate(self) -> int:
        sr = getattr(self.handler, "sample_rate", None)
        return sr or self.manifest.sample_rate or 16000

    def transcribe(self, audio, sample_rate: int, max_new_tokens: int = 2048,
                   temperature: float = 0.0, top_p: float = 1.0) -> str:
        return self.handler.transcribe(audio, sample_rate, max_new_tokens, temperature, top_p)

    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0):
        return self.handler.synthesize(text, voice, speed)


class Registry:
    """Loadable models by API id plus the worker pool used to run inference."""

    def __init__(self, specs: list[ModelSpec], providers: list[str] | None = None,
                 pool_size: int = 4) -> None:
        self.pool = ThreadPoolExecutor(max_workers=max(1, pool_size),
                                       thread_name_prefix="camaron-infer")
        self._models: dict[str, Model] = {}
        for spec in specs:
            try:
                self._models[spec.api_id] = Model(spec, providers=providers)
            except Exception:
                logger.exception("failed to load model %r", spec.api_id)
        logger.info("loaded %d model(s): %s", len(self._models), sorted(self._models))

    def list(self) -> list[str]:
        return sorted(self._models)

    def get(self, name: str) -> Model | None:
        return self._models.get(name)

    def require(self, name: str) -> Model:
        model = self._models.get(name)
        if model is None:
            raise ModelNotFound(name)
        return model

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False)


class ModelNotFound(KeyError):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
