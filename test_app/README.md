# Test application (in-browser harness)

A single, build-free `index.html` (vanilla JS, no bundler, no Node) that exercises the
Camaron service over the OpenAI-compatible audio API. It needs only a browser.

## Run

1. Start the service (in one terminal):
   ```bash
   python -m src --model-path ./models --port 8080
   ```
2. Serve this page (in another) — any static file server works:
   ```bash
   python -m http.server 5173 --directory test_app
   ```
3. Open http://localhost:5173 in a browser (mic + audio require a secure context:
   `localhost` qualifies).

## What it does
- **Echo** — record → `POST /v1/audio/transcriptions` → show the transcript →
  `POST /v1/audio/speech` → play the audio back. A full STT→TTS round trip.
- **Conversation** — chat. Each turn goes: your text (or a voice note you transcribe)
  → your own OpenAI-compatible LLM (`chat/completions`) → reply spoken via the service's
  `speech` endpoint. All three URLs/keys are configured in the settings card.

## Notes
- Mic audio is captured as 16-bit WAV (16 kHz) natively in the browser and sent as a
  multipart upload — the service decodes it with no external codec.
- Playback uses the Web Audio API on the `wav` returned by `/v1/audio/speech`.
- The audio service sends permissive CORS headers so a page on a different origin can
  call it; set the base URL in the settings card if you run it elsewhere.
