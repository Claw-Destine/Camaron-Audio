# Camaron Audio

![](test_app/camaron-audio.png)


Tiny shrimps are humble servants of The Mighty Claw. This one is here to help with speech related chores

---

OpenAI-compatible, self-hosted audio service — Speech-to-Text (ASR), Text-to-Speech
(TTS), and model listing — backed by **ONNX Runtime** so it runs on CPU or GPU with no
PyTorch required at runtime.

It speaks the OpenAI `/audio` API, so anything built for OpenAI's audio endpoints
(the Python `openai` client, for example) works against it with a base-URL swap.

## Endpoints

| Method | Path                        | Description                          |
|--------|-----------------------------|--------------------------------------|
| POST   | `/v1/audio/speech`          | Text → speech                        |
| POST   | `/v1/audio/transcriptions`  | Speech → text (any language)         |
| POST   | `/v1/audio/translations`    | Speech → text (always English)       |
| GET    | `/v1/models`                | List available models                |

## Quick start

```bash
# 1. Create an environment (Python 3.12+)
uv venv .venv
source .venv/bin/activate

# 2. Install (add --with tts for TTS phonemization)
uv pip install -e . --with tts --with dev

# 3. Place models under ./models  (or point the service at one via a manifest)
#    Each model is a folder containing ONNX file(s) + manifest.yaml, e.g.:
#      models/kokoro-82m-v1.0/{model.onnx,tokenizer.json,manifest.yaml,voices/}
#      models/whisper-tiny.en/{encoder_model.onnx,decoder_model.onnx,tokenizer.json,preprocessor_config.json,manifest.yaml}

# 4. Run
python -m src --model-path ./models --port 8080

# 5. Use it (OpenAI client)
python - <<'PY'
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
print(c.audio.transcriptions.create(model="whisper-tiny.en", file=open("x.wav","rb")))
open("out.wav","wb").write(c.audio.speech.create(model="kokoro-82m-v1.0",
    input="Hello", voice="af_heart").content)
PY
```

## Configuration

| Env var                   | Default    | Description                          |
|---------------------------|------------|--------------------------------------|
| `CAMARON_MODEL_PATH`      | `./models` | Model discovery root                 |
| `CAMARON_HOST`            | `0.0.0.0`  | Bind address                         |
| `CAMARON_PORT`            | `8080`     | HTTP port                            |
| `CAMARON_API_KEY`         | *(unset)*  | Bearer token (unset = no auth + WARN)|
| `CAMARON_THREAD_POOL_SIZE`| `4`        | Inference worker threads             |
| `CAMARON_LOG_LEVEL`       | `INFO`     | Logging level                        |

CLI flags (`--model-path`, `--host`, `--port`, `--api-key`, `--log-level`) override
environment variables.

## Models

The service discovers models by scanning `CAMARON_MODEL_PATH` for subdirectories that
contain at least one ONNX file and a `manifest.yaml`. See `AGENT.md` for the manifest
schema. A manifest may point `hf_url` at a Hugging Face repo; missing files are then
pulled into the standard HF cache on first use.

Supported model families:
- **ASR** — Whisper ONNX (HF split `encoder_model.onnx` / `decoder_model.onnx`)
- **TTS** — Kokoro ONNX (single `model.onnx` + phoneme tokenizer + per-voice `voices/`)

## Docker

```bash
docker build -t camaron-audio .                                   # CPU
docker build --build-arg RUNTIME=cuda -t camaron-audio:gpu .      # NVIDIA
docker run -p 8080:8080 -v ./models:/models -e CAMARON_MODEL_PATH=/models camaron-audio
```

## Tests

```bash
pytest                          # unit tests
pytest -m e2e                   # end-to-end (downloads test models, generates audio)
```

## Test app
`test_app/` is a SvelteKit single-page harness (echo + LLM conversation). See
`test_app/README.md`.
