"""Whisper family ASR: HF-split ONNX (encoder + decoder) with greedy/beam-free decoding.

Why this shape:
- The HF Whisper ONNX export splits the model into ``encoder_model.onnx`` (runs once
  per 30 s window) and ``decoder_model.onnx`` (autoregressive). This variant of the
  decoder has *no* past-key-value inputs, so we recompute the prefix each step
  (``input_ids`` = full prefix). That is O(n^2) in decoder work but far simpler and
  has no cache-lifetime correctness traps; Whisper-tiny is small enough that this is
  the minimal design that still works correctly.
- The log-mel spectrogram is produced by the reference ``WhisperFeatureExtractor``
  (from ``preprocessor_config.json``) so the mel matches the model exactly --
  hand-rolling it would be fragile. Only the model weights run in ONNX Runtime.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from ..audio import _linear_resample

logger = logging.getLogger("camaron.asr")

_MAX_NEW_TOKENS = 2048
_WHISPER_SR = 16000


class WhisperASR:
    def __init__(
        self,
        root: Path,
        providers: list[str] | None = None,
        inter_op: int | None = None,
        intra_op: int | None = None,
    ) -> None:
        self.root = Path(root)
        enc = self._locate("encoder_model.onnx")
        dec = self._locate("decoder_model.onnx")
        tok_dir = str(self._locate("tokenizer.json").parent)
        pre = self._locate("preprocessor_config.json")

        self.tokenizer = WhisperTokenizer.from_pretrained(tok_dir)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(pre))
        self.sos = int(self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
        self.eos = _safe_eos(self.tokenizer)

        options = ort.SessionOptions()
        options.log_severity_level = 3  # ORT warnings/errors only
        if inter_op is not None:
            options.inter_op_num_threads = inter_op
        if intra_op is not None:
            options.intra_op_num_threads = intra_op

        # One session per graph, shared across threads (ORT sessions are thread-safe).
        self.encoder = ort.InferenceSession(str(enc), options, providers=providers or None)
        self.decoder = ort.InferenceSession(str(dec), options, providers=providers or None)

    def _locate(self, name: str) -> Path:
        for base in (self.root, self.root / "onnx"):
            candidate = base / name
            if candidate.exists():
                return candidate
        found = list(self.root.rglob(name))
        if not found:
            raise FileNotFoundError(f"{name} not found under {self.root}")
        return found[0]

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        audio = _linear_resample(audio.astype(np.float32), sample_rate, _WHISPER_SR)
        features = (
            self.feature_extractor(audio, sampling_rate=_WHISPER_SR, return_tensors="np")[
                "input_features"
            ]
            .astype(np.float32)
        )
        encoder_hidden = self.encoder.run(["last_hidden_state"], {"input_features": features})[0]

        prefix = [self.sos]
        for _ in range(max_new_tokens):
            logits = self.decoder.run(
                ["logits"],
                {
                    "input_ids": np.array([prefix], dtype=np.int64),
                    "encoder_hidden_states": encoder_hidden,
                },
            )[0]
            token = _sample(logits[0, -1], temperature, top_p)
            if token == self.eos:
                break
            prefix.append(int(token))

        # Drop the leading SOS and decode; ``skip_special_tokens`` hides <|endofprompt|> etc.
        return self.tokenizer.decode(prefix[1:], skip_special_tokens=True)


def _safe_eos(tokenizer: WhisperTokenizer) -> int:
    try:
        return int(tokenizer.eos_token_id)
    except Exception:
        return int(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))


def _sample(logits: np.ndarray, temperature: float, top_p: float) -> int:
    """Greedy when temperature <= 0; otherwise temperature + nucleus (top-p) sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits / max(temperature, 1e-5)
    z = z - np.max(z)
    probs = np.exp(z)
    probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cum, top_p) + 1)
        keep = np.zeros_like(probs)
        keep[order[:cutoff]] = 1.0
        probs = probs * keep
        probs /= probs.sum()
    ids = np.arange(len(probs))
    return int(np.random.choice(ids, p=probs))
