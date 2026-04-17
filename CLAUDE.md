# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

MinbarAI listens to an Imam's Arabic Khutbah (sermon) via microphone and displays a live German translation on a screen for the congregation. It runs entirely offline/locally on a Windows PC.

## Environment Setup

```bash
# Activate venv before running anything
venv\Scripts\activate        # Windows cmd
source venv/Scripts/activate # Git Bash / this shell

pip install -r requirements.txt
```

## Running the App

```bash
# Real-time microphone translation (main use case)
python src/realtime.py              # uses default mic
python src/realtime.py 0           # specify device ID

# List available microphone devices
python src/list_devices.py

# Translate a pre-recorded audio file
python src/main.py audio_samples/test.mp3

# Test translation only (type Arabic, get German)
python src/test_translator.py

# Check all dependencies are installed
python src/install_language.py
```

## Architecture

The pipeline is: **Microphone → VAD → Whisper (local STT) → Translation API → Display**

- `src/realtime.py` — the main entry point for live use. Captures audio via `sounddevice`, uses energy-based Voice Activity Detection (VAD) to detect sentence boundaries, queues complete speech segments, and dispatches them to a background thread for Whisper transcription + translation.
- `src/display.py` — runs a lightweight HTTP server on `localhost:8080`. The display screen (a browser or second monitor) polls this page every 2 seconds via `<meta http-equiv="refresh">`. `update_display()` updates both the in-memory state and writes `display.html` to disk as a backup.
- `src/transcribe.py` / `src/main.py` — file-based transcription pipeline, used for testing with pre-recorded audio.

## Key Constants (in `realtime.py`)

| Constant | Default | Effect |
|---|---|---|
| `VAD_THRESHOLD` | `0.02` | RMS energy level to classify audio as speech |
| `SILENCE_DURATION` | `1.5s` | Silence needed to end a sentence |
| `MAX_SENTENCE_LENGTH` | `15s` | Force-flushes buffer if speaker doesn't pause |
| `MIN_BUFFER_SIZE` | `6 chunks` | Prevents very short noise bursts from triggering Whisper |

## Current Translation Backend

Uses the `translate` library (wraps MyMemory API by default) — no API key needed. The `Translator(from_lang="ar", to_lang="de")` call in each module handles this. The `arabic_reshaper` + `python-bidi` libraries are used to correctly render right-to-left Arabic text in terminal output.

## Whisper Model

All modules load `WhisperModel("small", device="cpu", compute_type="int8")`. The model downloads automatically on first run (~500MB). To change model size, update this call — `tiny` is faster, `medium`/`large` are more accurate but slower on CPU.
