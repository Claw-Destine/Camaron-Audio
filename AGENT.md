# Camaron Audio

OpenAI-compatible audio AI service (STT + TTS + model listing) backed by ONNX Runtime.

## Overview

Camaron Audio is a self-hosted service that exposes OpenAI's `/audio` API endpoints.
It loads locally stored ONNX models (or references them on Hugging Face) and serves
inference requests through a well-documented, drop-in-compatible HTTP interface.

## Goals

### Core
- Expose OpenAI-compatible `/v1/audio/*` and `/v1/models` endpoints
- ONNX-only: the service handles only models in ONNX format (via onnxruntime)
- Model discovery: scan `MODEL_PATH` for subdirectories, auto-detect valid models
- Each model directory contains the ONNX file(s) + a `manifest.yaml`
- HF URL support: manifests may reference a Hugging Face URL instead of a local file;
  the service downloads to the standard HF cache and loads from there on first use
- Auto-detect available execution providers (CUDA, TensorRT, CPU) via onnxruntime
- Multi-platform: runs where onnxruntime runs

### Non-goals (for now)
- Model training or fine-tuning
- Streaming TTS (full response only)
- Multi-tenant / full auth system (single API-key header auth)
- Queue-based scheduling (thread-pool concurrency)

## Future Goals
- Support API-compatible modes for other providers (e.g. ElevenLabs) behind the same engine

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind — find the most minimal, better-working design.
- Don't introduce slop: fragile case-patching code, dead code, useless code, or code far more complicated than it needs to be.
- Comment inference code where model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain *why* a shape, ordering, cache boundary, or memory choice exists.

## Tech Stack

| Component       | Choice                                                  |
|-----------------|---------------------------------------------------------|
| Language        | Python 3.12+                                            |
| Web framework   | FastAPI + uvicorn                                       |
| Inference       | onnxruntime (CUDA / TensorRT / CPU EP via auto-detect)  |
| HF downloads    | huggingface_hub                                         |
| CLI parsing     | argparse (stdlib)                                       |
| Config          | Environment variables + CLI flags                       |
| Tests           | pytest                                                  |
| Test app        | Build-free single-page HTML/JS (in-repo, no Node)       |

## API

### Endpoints

| Method | Path                        | Description                          |
|--------|-----------------------------|--------------------------------------|
| POST   | `/v1/audio/speech`          | Text → speech                        |
| POST   | `/v1/audio/transcriptions`  | Speech → text (any language)         |
| POST   | `/v1/audio/translations`    | Speech → text (always English)       |
| GET    | `/v1/models`                | List available models                |

### `POST /v1/audio/speech` (TTS)

Request (JSON body):
```json
{
  "model": "kokoro-82m-v1.0",
  "input": "Hello world",
  "voice": "alloy",
  "response_format": "wav|pcm|mp3|flac",
  "speed": 1.0,
  "temperature": 0.8,
  "top_p": 0.9
}
```
Response: binary audio in the requested codec (`wav` | `pcm` | `mp3` | `flac`). No streaming — full audio in one response.
Voice: accepts an OpenAI voice name (resolved via `voice_map`), a raw Kokoro voice id, or omits it — in which case `default_voice` is used.

### `POST /v1/audio/transcriptions` (STT)

Request (multipart/form-data): `model`, `file` (audio upload), `language`, `prompt`,
`response_format` (`text|json|verbose_json|srt|vtt`), `temperature`, `initial_prompt`
Response: plain text or JSON depending on `response_format`.

### `POST /v1/audio/translations` (STT → English)

Same request shape as transcriptions. Forces `language=en` server-side.
Equivalent to transcriptions with `language=en`.

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {"id": "whisper-tiny.en", "object": "model", "owned_by": "camaron-audio"},
    {"id": "kokoro-82m-v1.0", "object": "model", "owned_by": "camaron-audio"}
  ]
}
```
Model ID is the folder name under `MODEL_PATH` (or `name` from manifest if set).

### Error format

All errors use OpenAI's JSON envelope:
```json
{"error": {"message": "...", "type": "invalid_request_error", "code": null}}
```
Auth failures return 401; unknown model returns 404; bad request format returns 400.

## Model Storage & Discovery

### Directory layout
```
MODEL_PATH/
  whisper-tiny.en/
    model.onnx
    manifest.yaml
  kokoro-82m-v1.0/
    model.onnx
    tokenizer.onnx
    manifest.yaml
```

### Discovery rules
- Each subdirectory of `MODEL_PATH` is a candidate
- Valid model = at least one ONNX file + a `manifest.yaml`
- Invalid directories: logged with WARNING, skipped
- `GET /v1/models` reflects only valid models

### `manifest.yaml` (schema)

```yaml
# --- Required ---
name: whisper-tiny.en        # model ID in API requests (defaults to folder name)
task: asr                    # "asr" | "tts"
onnx: model.onnx             # primary ONNX file (path relative to model dir)

# --- Optional ---
hf_url: https://huggingface.co/onnx-community/whisper-tiny.en
  # When set, and onnx file not present locally, download to HF cache and load from there.
  # (No file copying into MODEL_PATH.)

languages: [en]              # supported languages (ASR only)
voices: [af_heart, am_michael, bf_emma, bm_georgie]  # allowed raw Kokoro voice IDs (TTS only)
voice_map:                   # TTS only: OpenAI voice name -> raw Kokoro voice ID
  alloy: af_heart
  echo: am_michael
  fable: af_sarah
  nova: af_nina
  onyx: bf_emma
  shimmer: bm_georgie
default_voice: af_heart      # TTS only: used when voice is omitted or unknown
sample_rate: 16000           # audio sample rate in Hz (defaults to 16000 ASR / 24000 TTS)

# Inference parameter bounds exposed to the API
params:
  temperature: {default: 0.0, min: 0.0, max: 2.0}
  top_p: {default: 1.0, min: 0.0, max: 1.0}

# Auxiliary ONNX files (e.g. tokenizer for TTS)
onnx_files:
  - name: tokenizer.onnx

# Extra metadata (free form, not parsed by service)
metadata:
  author: community
  license: Apache-2.0
```

## HF URL Flow

1. Manifest has `hf_url` and local ONNX file is absent
2. Service uses `huggingface_hub.snapshot_download()` to cache files into
   the standard HF cache directory (`~/.cache/huggingface/`)
3. ONNX files are loaded from the HF cache path — not copied to `MODEL_PATH`
4. On subsequent starts, cache hit means no re-download
5. If model is used from a local ONNX file (hf_url absent), HF is never touched

### TTS voice tables
Kokoro selects the voice by a 256-dim style row *for the given token count*, so each
voice ships as a `voices/<id>.<npy|pt|bin>` table (one row per context length). The
upstream voices are `hexgrad/Kokoro-82M` `voices/*.pt`; convert them once to `.npy`
(`scripts/prepare_models.py`, or `tools/convert_voices.py`) so the running service
stays **torch-free**. `voice_map` maps OpenAI voice names → a voice id that has a
table on disk.

## Configuration

Environment variables (CLI flags override):

| Variable                  | Default      | Description                              |
|---------------------------|--------------|------------------------------------------|
| `CAMARON_MODEL_PATH`      | `./models`   | Root directory for model discovery       |
| `CAMARON_HOST`            | `0.0.0.0`    | Bind address                             |
| `CAMARON_PORT`            | `8080`       | HTTP port                                |
| `CAMARON_API_KEY`         | *(unset)*    | Required Bearer token (unset = no auth)  |
| `CAMARON_THREAD_POOL_SIZE`| `4`          | Inference thread pool size               |
| `CAMARON_LOG_LEVEL`       | `INFO`       | Logging level (DEBUG/INFO/WARNING)       |

Auth: when `CAMARON_API_KEY` is set, every request must carry
`Authorization: Bearer <key>`. Missing or wrong key → 401 with OpenAI error envelope.
When unset, auth is disabled (local dev mode); a WARNING is logged at startup
flagging that the service is running unauthenticated.

## Inference & Concurrency

- ONNX Runtime sessions created at model load time (cached — one per model)
- Sessions are thread-safe; concurrent requests share the same session
- Async FastAPI handlers dispatch inference to a `concurrent.futures.ThreadPoolExecutor`
- Pool size from `CAMARON_THREAD_POOL_SIZE` (default 4)
- onnxruntime auto-detects CUDA → TensorRT → CPU; no explicit provider wiring needed

## Project Structure

```
camaron-audio/
  AGENT.md                   # this spec
  pyproject.toml             # dependencies, tool config
  README.md
  src/
    __init__.py
    __main__.py              # CLI entry: `python -m src` or `camaron-audio`
    config.py                # env var + CLI flag parsing → Config dataclass
    discovery.py             # MODEL_PATH scan → list[ModelInfo]
    manifest.py              # manifest.yaml load/validate
    huggingface.py           # snapshot_download wrapper, cache path resolution
    inference/
      __init__.py
      engine.py              # ONNX session lifecycle, thread-safe inference runner
      asr.py                 # Whisper-family decode (logits → text)
      tts.py                 # Kokoro-family encode (text → waveform)
    api/
      __init__.py
      app.py                 # FastAPI app factory, router registration, startup/shutdown
      speech.py              # POST /v1/audio/speech
      transcriptions.py      # POST /v1/audio/transcriptions
      translations.py        # POST /v1/audio/translations  (thin wrapper on transcriptions)
      models.py              # GET  /v1/models
      errors.py              # OpenAI error envelope, exception handlers
  models/                    # (gitignored) default MODEL_PATH
  .gitignore
  tests/
    conftest.py              # fixtures: running server, OpenAI client
    test_speech.py           # E2E TTS via OpenAI client
    test_transcriptions.py   # E2E ASR via OpenAI client
    test_models.py           # E2E model listing
    test_discovery.py        # unit: directory scan
    test_manifest.py         # unit: YAML parse, validation
  test_app/                  # build-free single-page HTML test harness (no Node)
  Dockerfile                 # parameterized: cpu | cuda | rocm
```

## Testing

### E2E / functional
- Use the official OpenAI Python client (`openai` package) to exercise the running service
- ASR model for tests: `https://huggingface.co/onnx-community/whisper-tiny.en`
- TTS model for tests: `https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX`
- Assert: correct HTTP status, correct content-type, valid audio/text content, model listing

### Test application (`test_app/`)
Build-free single-page HTML/JS (`test_app/index.html`), served by any static file
server (e.g. `python -m http.server --directory test_app`) or opened directly:
- **Echo**: record → STT → show transcript → TTS → play audio
- **Conversation**: chat UI that chains STT → LLM (via OpenAI API) → TTS

## Docker

One `Dockerfile` at the repo root, parameterized by a `RUNTIME` build arg:

```dockerfile
ARG RUNTIME=cpu   # cpu | cuda | rocm
```

`RUNTIME` selects the base image and the `onnxruntime` pip package:

| RUNTIME | Base image                              | pip package          |
|---------|-----------------------------------------|----------------------|
| `cpu`   | `python:3.12-slim`                      | `onnxruntime`        |
| `cuda`  | `nvidia/cuda:12.4-runtime-ubuntu22.04`  | `onnxruntime-gpu`    |
| `rocm`  | `rocm/dev-ubuntu-22.04`                 | `onnxruntime-rocm`   |

Build:
```bash
docker build -t camaron-audio:cpu .                          # CPU (default)
docker build --build-arg RUNTIME=cuda -t camaron-audio:gpu .  # NVIDIA
docker build --build-arg RUNTIME=rocm -t camaron-audio:rocm . # AMD
```

Run (models mounted as a volume so they persist across container restarts):
```bash
docker run -p 8080:8080 \
  -v ./models:/models \
  -e CAMARON_MODEL_PATH=/models \
  -e CAMARON_API_KEY=sk-my-key \
  camaron-audio:gpu
```

The HF cache (`HF_HOME`) is also mountable as a volume if you want models to
survive image rebuilds without re-downloading.

## Development Guidelines

- **Do not push any changes to the remote repo.** The user will push after review.
- Follow quality rules above — minimal, elegant, no slop.
- One Python package, monorepo, no sub-package splits.
