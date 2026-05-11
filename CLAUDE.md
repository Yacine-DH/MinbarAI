# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

MinbarAI listens to an Imam's Arabic Khutbah (sermon) via microphone and displays a live German translation as a desktop overlay for the congregation. It runs on Windows.

## Environment Setup

```powershell
# Activate venv before running anything
.\venv\Scripts\Activate.ps1   # PowerShell
# or:
venv\Scripts\activate         # cmd
source venv/Scripts/activate  # Git Bash

pip install -r requirements.txt
```

Also need **Ollama** installed and `translategemma:4b` pulled:
```powershell
ollama pull translategemma:4b
```

API key required in `.env` at project root:
```
ELEVENLABS_API_KEY=sk_xxx
```

## Running the App

```powershell
python src/ui.py            # default mic
python src/ui.py 18         # specific device

# Utilities
python src/list_devices.py  # list mics + host APIs
python src/test_gemma.py    # sanity-check Gemma via Ollama
python src/test_elevenlabs_key.py
python src/test_scribe_v2.py
```

## Architecture

Pipeline: **Mic → Silero VAD → Scribe v2 (cloud) → Buffer → TranslateGemma 4B (local Ollama) → Helsinki MT (fallback) → PyQt6 overlay**

```
src/
├── ui.py                       # entry point — boots QApplication + MainWindow
├── backend/
│   ├── audio.py                # sounddevice + Silero VAD → audio_queue
│   ├── scribe_batch.py         # ElevenLabs Scribe v2 HTTP wrapper
│   ├── gemma_translate.py      # Ollama HTTP client (translategemma:4b)
│   ├── local_translate.py      # Helsinki opus-mt-ar-de (Marian MT) fallback
│   ├── history.py              # JSON persistence with khutba sessions
│   └── workers.py              # TranscribeWorker + TranslateWorker QThreads + buffering
└── frontend/
    ├── main_window.py          # overlay window (display pacing, fade, edge-resize)
    └── history_window.py       # khutba browser (combo box + entries list)
```

## Key constants

**`src/backend/workers.py`** (`TranslateWorker`):

| Constant | Default | Effect |
|---|---|---|
| `MAX_WORDS` | 8 | Flush buffer when combined ≥ N words |
| `SILENCE_TIMEOUT` | 0.7s | Flush after silence between transcripts |
| `PUNCTUATION` | `. ? ! ؟` | Punctuation that triggers flush |
| `MIN_GEMMA_INTERVAL` | 0.0s | Rate-limit guard between Gemma calls |

**`src/backend/audio.py`** (Silero VAD):

| Constant | Default | Effect |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Sample rate |
| `BLOCK_SIZE` | 512 | Silero VAD requirement at 16kHz |
| `SPEECH_THRESHOLD` | 0.5 | Speech probability cutoff |
| `SILENCE_DURATION` | 0.8s | Silence before flushing a speech segment |
| `MAX_SENTENCE_SECONDS` | 15 | Force-flush after this many seconds |

**`src/frontend/main_window.py`** (`MainWindow`):

| Constant | Default | Effect |
|---|---|---|
| `MIN_DISPLAY_MS` | 3000 | Min time each translation stays on screen |
| `FADE_MS` | 220 | Per-side fade duration (out + in) |
| `RESIZE_MARGIN` | 8 | Edge-detection margin in pixels |

## Persistence

`history.json` at project root (gitignored). Schema:
```json
{
  "khutbas": [
    {
      "id": "2026-05-10T14:30:00",
      "started_at": "2026-05-10T14:30:00",
      "ended_at": "2026-05-10T15:12:48",
      "entries": [{"ts": "...", "ar": "...", "de": "..."}]
    }
  ]
}
```

A new khutba is appended each app launch. `ended_at` is written on `closeEvent`. Legacy flat-list format auto-migrates on load.

## Threading

| Thread | Purpose |
|---|---|
| Main (Qt loop) | Render, mouse, keyboard |
| sounddevice callback | Push audio to VAD |
| VAD daemon | Push speech segments to `audio_queue` |
| `TranscribeWorker` QThread | `audio_queue` → Scribe → `arabic_queue` |
| `TranslateWorker` QThread | `arabic_queue` → buffer → Gemma/Helsinki → emit `(ar, de)` |
| Loader threads (3x) | Lazy-load Helsinki, Gemma, ElevenLabs at startup |

Qt signal/slot connections marshal results back onto the main thread.

## Legacy / not in active pipeline

- `src/main.py` — old entry shim, just re-runs `ui.MainWindow`
- `src/realtime.py`, `display.py`, `scribe_realtime.py` — browser-overlay alternative entry path
- Test scripts: `test_gemma.py`, `test_scribe_v2.py`, `test_elevenlabs_key.py`, `test_translate.py`, `test_translator.py`

## Notes for code edits

- PowerShell is the primary shell; `.\venv\Scripts\Activate.ps1` activates venv. `source` does not exist in PowerShell.
- The Qt overlay uses `WA_TranslucentBackground` + `FramelessWindowHint` — a margin with `rgba(0,0,0,1)` (alpha=1, not 0) is required so the resize edges still receive mouse events on Windows.
- App quit must use `os._exit(0)` after `QApplication.quit()` because `Qt.WindowType.Tool` windows do not trigger `quitOnLastWindowClosed`, and background daemon threads (HF download, Ollama HTTP) can block normal exit.
