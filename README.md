# 🕌 MinbarAI

> Real-time Arabic to German speech translation — built for the mosque community.

---

## What is MinbarAI?

MinbarAI listens to the Imam's Khutbah (sermon) in Arabic and displays a live German translation as a desktop overlay for the congregation.

The current pipeline uses **ElevenLabs Scribe v2** for Arabic transcription (cloud) and **TranslateGemma 4B** running locally via Ollama for translation. **Helsinki-NLP Marian MT** stays bundled as a local fallback when Gemma is unavailable.

---

## How it works

```
Microphone → Silero VAD → ElevenLabs Scribe v2 → buffer (punct / 8 words / 0.7s silence)
                                                    │
                            ┌───────────────────────┘
                            ▼
        TranslateGemma 4B (Ollama, local)  ──fail──►  Helsinki opus-mt-ar-de (local)
                            │
                            ▼
                  PyQt6 overlay (fade transition, 3s minimum display)
                            │
                            ▼
                  Khutba history (JSON, per-session)
```

1. Microphone captures Imam's voice (16kHz mono)
2. Silero VAD detects speech segments based on probability
3. Each segment is sent to ElevenLabs Scribe v2 for Arabic transcription
4. Arabic transcripts are buffered until punctuation, 8 words, or 0.7s silence
5. Buffered Arabic is translated by TranslateGemma 4B (local, via Ollama) — Helsinki MT is the fallback
6. Each German translation appears in the overlay with a smooth fade transition and stays on screen ≥ 3s
7. Every transcribed + translated line is saved in `history.json` grouped by Khutba session

---

## Tech Stack

| Component | Technology | Where it runs |
|---|---|---|
| Voice activity detection | Silero VAD (PyTorch) | local CPU |
| Speech-to-Text | ElevenLabs Scribe v2 (`scribe_v2`) | cloud (API key) |
| Translation (primary) | TranslateGemma 4B (`translategemma:4b`) via Ollama | local |
| Translation (fallback) | Helsinki-NLP `opus-mt-ar-de` (Marian MT) | local CPU |
| Audio capture | `sounddevice` | local |
| UI | `PyQt6` — frameless, transparent, resizable overlay | local |
| Persistence | JSON file `history.json` (one entry per khutba) | local |
| Language | Python 3.11+ | — |

---

## Project Structure

```
src/
├── ui.py                       # entry point
├── backend/
│   ├── audio.py                # mic + Silero VAD
│   ├── scribe_batch.py         # ElevenLabs Scribe v2 STT
│   ├── gemma_translate.py      # TranslateGemma via Ollama
│   ├── local_translate.py      # Helsinki Marian MT fallback
│   ├── history.py              # JSON persistence + khutba schema
│   └── workers.py              # QThread workers + buffering logic
└── frontend/
    ├── main_window.py          # overlay window (display pacing, fade, resize)
    └── history_window.py       # khutba browser window
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Yacine-DH/MinbarAI.git
cd MinbarAI

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1     # PowerShell
# or:
venv\Scripts\activate           # Windows cmd
source venv/Scripts/activate    # Git Bash

# Install Python dependencies
pip install -r requirements.txt
```

### Set up Ollama + TranslateGemma

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull the model:
   ```powershell
   ollama pull translategemma:4b
   ```
3. Ollama runs in the background and exposes its API at `http://localhost:11434`.

### Set up API keys

Create `.env` at the project root:

```
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The `.env` file is already in `.gitignore`.

### Find your microphone device ID

```powershell
python src/list_devices.py
```

Pick a device whose host API is **MME** or **Windows WASAPI** — avoid **Windows WDM-KS** (especially for Bluetooth mics).

### Run

```powershell
python src/ui.py            # default mic
python src/ui.py 18         # specific device id
```

### Controls

| Key / button | Action |
|---|---|
| `S` or ⚙ icon | Toggle settings panel (opacity, font sizes) |
| `H` or 🕘 icon | Open Khutba history window |
| `Esc` or ✕ icon | Quit (cleanly closes Ollama session, audio threads) |
| Click + drag center | Move the overlay |
| Drag edge / corner | Resize the overlay |

---

## Tunables (in `src/backend/workers.py`)

| Constant | Default | Effect |
|---|---|---|
| `MAX_WORDS` | 8 | Max words in buffer before flushing to translator |
| `SILENCE_TIMEOUT` | 0.7s | Silence between transcripts that triggers flush |
| `PUNCTUATION` | `. ? ! ؟` | Punctuation that triggers immediate flush |
| `MIN_GEMMA_INTERVAL` | 0.0s | Rate-limit guard between Gemma calls (local = no throttle needed) |

In `src/frontend/main_window.py`:

| Constant | Default | Effect |
|---|---|---|
| `MIN_DISPLAY_MS` | 3000 | Each translation stays on-screen ≥ this many ms before swap |
| `FADE_MS` | 220 | Fade-out + fade-in duration per swap |

---

## Motivation

This project was born out of a real need — helping German-speaking members of a local mosque follow the Friday Khutbah in their language. MinbarAI is a step toward making mosque services more accessible and inclusive.

---

## Author

Built with ❤️ for the community.
