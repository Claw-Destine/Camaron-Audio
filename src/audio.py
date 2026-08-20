"""Audio codecs with no hard codec dependencies.

WAV and raw PCM are handled with the standard library (always available). Everything
else (mp3, flac, m4a, ...) is delegated to an external ``ffmpeg`` binary if one is on
``PATH``; if not, a clear error is raised rather than silently producing silence.
This keeps the base install lean while still supporting the OpenAI codec set in
production (where ffmpeg is present).
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger("camaron.audio")

# extension (lower, no dot) -> ffmpeg demuxer name
_FF_DEMUX = {"wav": "wav", "mp3": "mp3", "flac": "flac", "m4a": "aac", "aac": "aac",
             "ogg": "ogg", "opus": "ogg", "webm": "webm", "pcm": "f32le", "raw": "f32le"}


def _suffix(filename: str | None) -> str:
    if not filename:
        return ""
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _linear_resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Resample a real signal by linear interpolation between samples."""
    if src == dst:
        return x
    n_out = int(round(len(x) * dst / src))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    # positions in source-sample units for each destination sample
    pos = np.arange(n_out, dtype=np.float64) * (src / dst)
    lo = np.floor(pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, len(x) - 1)
    frac = (pos - lo).astype(np.float32)
    return (x[lo] * (1 - frac) + x[hi] * frac).astype(np.float32)


def decode_audio(data: bytes, filename: str | None = None) -> tuple[np.ndarray, int]:
    """Decode to mono ``float32`` in [-1, 1] plus the native sample rate."""
    ext = _suffix(filename)
    if ext in ("pcm", "raw", ""):
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return arr, 16000
    if ext == "wav":
        return _decode_wav(data)
    if not _have_ffmpeg():
        raise CodecError(f"decoding .{ext or 'audio'} needs ffmpeg, which is not installed")
    return _ffmpeg_decode(data, ext, target_sr=None)


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise CodecError(f"unsupported WAV sample width: {sw}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, sr


def _ffmpeg_decode(data: bytes, ext: str, target_sr: int | None) -> tuple[np.ndarray, int]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", _FF_DEMUX[ext], "-i", "-",
           "-f", "f32le"]
    if target_sr:
        cmd += ["-ac", "1", "-ar", str(target_sr)]
    else:
        cmd += ["-ac", "1"]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
    arr = np.frombuffer(out, dtype=np.float32)
    return arr, int(target_sr or 16000)


def encode_audio(waveform: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    """Encode a mono ``float32`` waveform to the requested format; return (bytes, mime)."""
    fmt = (fmt or "wav").lower().lstrip(".")
    if fmt in ("wav", "wave", "riff"):
        return _encode_wav(waveform, sr), "audio/wav"
    if fmt in ("pcm", "raw", "l16"):
        arr = np.clip(waveform, -1, 1)
        return (arr * 32767).astype(np.int16).tobytes(), "audio/pcm"
    if not _have_ffmpeg():
        raise CodecError(f"encoding .{fmt} needs ffmpeg, which is not installed")
    return _ffmpeg_encode(waveform, sr, fmt), _mime(fmt)


def _encode_wav(waveform: np.ndarray, sr: int) -> bytes:
    arr = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def _ffmpeg_encode(waveform: np.ndarray, sr: int, fmt: str) -> bytes:
    raw = np.clip(waveform, -1, 1).astype(np.float32).tobytes()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "-",
           "-f", _FF_DEMUX.get(fmt, fmt), "-"]
    out = subprocess.run(cmd, input=raw, capture_output=True, check=True).stdout
    if not out:
        raise CodecError(f"ffmpeg produced no output for .{fmt}")
    return out


def _mime(fmt: str) -> str:
    return {"mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}.get(fmt, f"audio/{fmt}")


class CodecError(RuntimeError):
    """Raised when a requested codec cannot be produced or consumed."""
