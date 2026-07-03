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
```

## Architecture

Pipeline: **Mic → Silero VAD → Scribe v2 (cloud) → Buffer → Quran verse matcher → [TranslateGemma (remote 12B → local 4B) + Helsinki MT candidates → QE rerank] → postprocess → PyQt6 overlay**

Recited Quran verses bypass MT entirely: `quran_match.py` fuzzy-matches the
transcript against all 6236 verses and serves the canonical Bubenheim & Elyas
German. Everything else is translated by both MT engines; a cross-lingual
embedding reranker picks the better candidate.

```
src/
├── ui.py                       # entry point — boots QApplication + MainWindow
├── backend/
│   ├── audio.py                # sounddevice + Silero VAD → audio_queue
│   ├── scribe_batch.py         # ElevenLabs Scribe v2 HTTP wrapper
│   ├── gemma_translate.py      # Ollama client, ordered endpoints: remote 12B tunnel → local 4B
│   ├── local_translate.py      # Helsinki opus-mt-ar-de (Marian MT) second candidate
│   ├── quran_match.py          # verse matcher → canonical Bubenheim German (data/*.json)
│   ├── rerank.py               # cross-lingual QE: picks best MT candidate (MiniLM)
│   ├── postprocess.py          # Gott→Allah, Bote→Gesandter, language-leak guard
│   ├── history.py              # JSON persistence with khutba sessions
│   └── workers.py              # TranscribeWorker + TranslateWorker QThreads + buffering
└── frontend/
    ├── main_window.py          # overlay window (display pacing, fade, edge-resize)
    └── history_window.py       # khutba browser (combo box + entries list)
```

Remote GPU boost (optional): run `notebooks/remote_ollama.ipynb` on
Colab/Kaggle (T4, internet on), copy the printed `REMOTE_OLLAMA_HOST=...`
line into `.env`. Health-checked every 30 s; dead tunnel falls back to
local 4B automatically. Optional `.env` keys: `REMOTE_OLLAMA_HOST`,
`REMOTE_MODEL_ID` (default `translategemma:12b`).

## Accuracy evaluation

```powershell
python tests\eval_translation.py            # full pipeline (matcher + MT + rerank)
python tests\eval_translation.py --no-matcher --tag baseline
python -m pytest tests\test_quran_match.py -q
```
47 cases, metric = MiniLM cosine vs reference German. History appended to
`tests/eval_results.json`. Current: **0.950 overall** (Quran 1.0, khutbah
0.88). Research notes: `docs/translation_research.md`. Quran data files in
`data/` come from fawazahmed0/quran-api (jsDelivr CDN).

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

## Notes for code edits

- PowerShell is the primary shell; `.\venv\Scripts\Activate.ps1` activates venv. `source` does not exist in PowerShell.
- The Qt overlay uses `WA_TranslucentBackground` + `FramelessWindowHint` — a margin with `rgba(0,0,0,1)` (alpha=1, not 0) is required so the resize edges still receive mouse events on Windows.
- App quit must use `os._exit(0)` after `QApplication.quit()` because `Qt.WindowType.Tool` windows do not trigger `quitOnLastWindowClosed`, and background daemon threads (HF download, Ollama HTTP) can block normal exit.
