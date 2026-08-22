"""Kokoro family TTS: single full-pipeline ONNX graph (text -> speech).

Calling convention (from the ONNX community export):
  input_ids : int64  [1, n]  -- phoneme token ids, padded 0 at start and end
  style     : float32[1, 256] -- the voice's style row *for this token length*
  speed     : float32[1]
  -> waveform: float32 [1, num_samples]

The voice "style" table ships as ``voices/<voice>.<npy|pt|bin>`` with one 256-dim
row per context length (``table[len(phonemes)]``); that length-indexed selection is
the non-obvious mechanic worth pinning down.

Language selection: the Kokoro graph is multilingual (one model, all languages) --
what changes per language is only the phonemizer and the voice tables. The voice
therefore decides the language (matching OpenAI's API, where /v1/audio/speech has no
language field): ``manifest.voice_language`` maps each raw voice id to a language,
defaulting to "en". Phonemizers are built lazily on first use of a language:
an ASR-only deployment never pulls phonemization deps, and a partial install keeps
serving the languages it has.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..manifest import Manifest

logger = logging.getLogger("camaron.tts")

_KOKORO_SAMPLE_RATE = 24000

# espeak-ng language codes for the non-English/CJK voices; these are phonemized with
# misaki's EspeakG2P backend inside Kokoro itself (e.g. Spanish `es`, French `fr-fr`).
_ESPEAK_CODES = {"es": "es", "fr": "fr-fr", "hi": "hi", "it": "it", "pt": "pt-br"}


class VoiceError(RuntimeError):
    """Client-facing: an unknown voice or voice table that cannot be loaded for a request."""


class TTSBackendError(RuntimeError):
    """Server-facing: the TTS backend is misconfigured (e.g. phonemizer not installed)."""


def _build_phonemizer(language: str):
    if language == "en":
        try:
            from misaki.en import G2P
        except ImportError as exc:
            raise TTSBackendError(
                "TTS phonemization (misaki) is not installed; install the 'tts' extra "
                "(`pip install -e '.[tts]'`) to enable text-to-speech"
            ) from exc
        return G2P("en-us")

    extra = "tts-ja" if language == "ja" else "tts"
    try:
        if language in _ESPEAK_CODES:
            from misaki.espeak import EspeakG2P
            return EspeakG2P(_ESPEAK_CODES[language])
        if language == "zh":
            from misaki.zh import ZHG2P
            return ZHG2P()
        if language == "ja":
            from misaki.ja import JAG2P
            return JAG2P()
    except ImportError as exc:
        raise TTSBackendError(
            f"TTS phonemizer for '{language}' is not installed; "
            f"install the '{extra}' extra (`pip install -e '.[{extra}]'`)"
        ) from exc
    raise TTSBackendError(
        f"no phonemizer for language {language!r}; check the manifest 'voice_language'"
    )


class KokoroTTS:
    def __init__(
        self,
        root: Path,
        manifest: Manifest,
        voice_dirs: list[Path],
        providers: list[str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.root = Path(root)
        self.sample_rate = manifest.sample_rate or _KOKORO_SAMPLE_RATE
        self.voice_dirs = [Path(d) for d in voice_dirs] or [self.root]
        self._voice_cache: dict[str, np.ndarray] = {}
        self._voice_lock = threading.Lock()

        if not manifest.voices and manifest.default_voice is None:
            logger.warning("manifest for %r declares no voices or default_voice", self.root)

        model_path = self._locate("model.onnx")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), options, providers=providers or None)

        vocab_path = self._locate("tokenizer.json")
        self._phoneme2id = json.loads(vocab_path.read_text())["model"]["vocab"]

        self._phonemizers: dict[str, object] = {}  # language -> G2P instance, built lazily

    def _locate(self, name: str) -> Path:
        for base in (self.root, self.root / "onnx"):
            candidate = base / name
            if candidate.exists():
                return candidate
        found = list(self.root.rglob(name))
        if not found:
            raise FileNotFoundError(f"{name} not found under {self.root}")
        return found[0]

    # -- public API ----------------------------------------------------------
    def resolve_voice(self, voice: str | None) -> str:
        if voice is None:
            if self.manifest.default_voice:
                return self.manifest.default_voice
            if self.manifest.voices:
                return self.manifest.voices[0]
            raise VoiceError("no voice specified and the manifest declares no default_voice")
        if voice in self.manifest.voice_map:
            return self.manifest.voice_map[voice]
        if voice in self.manifest.voices:
            return voice
        raise VoiceError(f"voice {voice!r} is not in voice_map/voices and no default is set")

    def synthesize(
        self,
        text: str,
        voice: str | None,
        speed: float = 1.0,
    ) -> np.ndarray:
        voice_id = self.resolve_voice(voice)
        language = self.manifest.voice_language.get(voice_id, "en")
        tokens = self._phonemize(text, language)
        if len(tokens) > 510:  # context window leaves room for the two pad tokens
            raise VoiceError("text too long for a single Kokoro context window (<= 510 phonemes)")
        style = self._style_row(voice_id, len(tokens))
        waveform = self.session.run(
            ["waveform"],
            {
                "input_ids": np.array([[0] + tokens + [0]], dtype=np.int64),
                "style": style,
                "speed": np.array([float(speed)], dtype=np.float32),
            },
        )[0]
        return waveform[0].astype(np.float32)

    # -- internals -----------------------------------------------------------
    def _phonemizer(self, language: str):
        with self._voice_lock:
            if language not in self._phonemizers:
                self._phonemizers[language] = _build_phonemizer(language)
            return self._phonemizers[language]

    def _phonemize(self, text: str, language: str) -> list[int]:
        result = self._phonemizer(language)(text)
        phonemes = result[0] if isinstance(result, (list, tuple)) else result
        ids = [self._phoneme2id.get(c) for c in phonemes]
        unknown = [c for c in phonemes if c not in self._phoneme2id]
        if unknown:
            logger.warning("%d phoneme(s) unmapped and dropped: %r", len(unknown), unknown[:8])
        return [i for i in ids if i is not None]

    def _style_row(self, voice_id: str, n_tokens: int) -> np.ndarray:
        # The voice table has one style row per context length, indexed by the token
        # count for exactly that length. Clamp so a text at the context edge (which
        # still fits input_ids with its two pads) never indexes past the last row.
        table = self._load_voice(voice_id)
        n = max(0, min(n_tokens, table.shape[0] - 1))
        return np.asarray(table[n], dtype=np.float32).reshape(1, 256)

    def _load_voice(self, voice_id: str) -> np.ndarray:
        with self._voice_lock:
            if voice_id in self._voice_cache:
                return self._voice_cache[voice_id]
        table = None
        for base in self.voice_dirs:
            for ext in (".npy", ".pt", ".bin"):
                path = Path(base) / "voices" / f"{voice_id}{ext}"
                if path.exists():
                    table = self._read_voice_file(path, ext)
                    break
        if table is None:
            searched = ", ".join(f"{d}/voices/{voice_id}.(npy|pt|bin)" for d in self.voice_dirs)
            raise VoiceError(f"voice {voice_id!r} not found (searched: {searched})")
        table = table.astype(np.float32)
        if table.shape[-1] != 256 or table.ndim not in (2, 3):
            raise VoiceError(f"voice {voice_id!r} has unexpected shape {table.shape}")
        with self._voice_lock:
            self._voice_cache[voice_id] = table
        return table

    @staticmethod
    def _read_voice_file(path: Path, ext: str) -> np.ndarray:
        if ext == ".npy":
            return np.load(path)
        if ext == ".bin":  # raw float32 blob, shape [-1, 1, 256] per ONNX convention
            raw = np.fromfile(path, dtype=np.float32)
            return raw.reshape(-1, 1, 256)
        if ext == ".pt":
            import torch

            return torch.load(path, map_location="cpu", weights_only=True).numpy()
        raise VoiceError(f"unsupported voice file type: {ext}")
