# 🕌 MinbarAI

> Real-time Arabic to German speech translation — built for the mosque community.

---

## What is MinbarAI?

MinbarAI listens to the Imam's Khutbah (sermon) in Arabic and displays a live German translation as a desktop overlay for the congregation.

The pipeline uses **ElevenLabs Scribe v2** for Arabic transcription (cloud), a **Quran verse matcher** that serves the canonical Bubenheim & Elyas German for recited verses, and **TranslateGemma** (remote 12B with automatic fallback to local 4B via Ollama) plus **Helsinki-NLP Marian MT** as parallel candidates — a cross-lingual reranker picks the better translation per chunk.

**Translation accuracy: 0.958** (semantic similarity vs reference, 47-case eval — Quran 1.00, khutbah speech 0.90).

---

## How it works

```
Microphone → Silero VAD → ElevenLabs Scribe v2 → buffer (8 words / 0.7s silence)
                                                    │
                                                    ▼
                                  ┌── Quran verse matcher (6236 verses) ──┐
                            match │                                       │ no match
                                  ▼                                       ▼
              canonical Bubenheim & Elyas German         TranslateGemma (remote 12B → local 4B)
              + sura:verse reference badge                + Helsinki opus-mt-ar-de
                                  │                                       │
                                  │                        cross-lingual QE rerank picks best
                                  │                                       │
                                  │                        postprocess (Allah, Gesandter, leak guard)
                                  └───────────────────┬───────────────────┘
                                                      ▼
                                  PyQt6 overlay (fade, 3s minimum display)
                                                      │
                                                      ▼
                                  Khutba history (JSON, per-session)
```

1. Microphone (selectable in the UI) captures the Imam's voice; Silero VAD segments speech
2. Segments go to ElevenLabs Scribe v2 for Arabic transcription; transcripts are buffered
3. **Recited Quran verses bypass MT entirely** — a fuzzy matcher (diacritic-stripped, bigram-indexed) recognizes full verses, partial recitation of long verses, and multi-verse passages, then displays the complete canonical German with a 📖 sura:verse badge
4. Everything else is translated by both TranslateGemma and Helsinki; a multilingual-MiniLM reranker compares each candidate against the Arabic source and picks the semantically closer one
5. Post-processing enforces Islamic German terminology (Allah, Gesandter) and rejects wrong-language output
6. Every line is saved in `history.json` grouped by Khutba session

### Optional remote GPU boost

Run `notebooks/remote_ollama.ipynb` on Google Colab or Kaggle (free T4 GPU), copy the printed `REMOTE_OLLAMA_HOST=...` line into `.env` — the app then uses **translategemma:12b** (~2s/chunk) and health-checks the tunnel every 30s. If the session dies mid-khutbah, it falls back to the local 4B automatically, no restart needed.

---

## Tech Stack

| Component | Technology | Where it runs |
|---|---|---|
| Voice activity detection | Silero VAD (PyTorch) | local CPU |
| Speech-to-Text | ElevenLabs Scribe v2 (`scribe_v2`) | cloud (API key) |
| Quran verse matching | rapidfuzz + bigram index over Tanzil text, Bubenheim & Elyas German | local CPU |
| Translation (primary) | TranslateGemma via Ollama — remote `12b` (Colab/Kaggle T4) → local `4b` | remote GPU / local |
| Translation (2nd candidate) | Helsinki-NLP `opus-mt-ar-de` (Marian MT) | local CPU |
| Candidate reranking | `paraphrase-multilingual-MiniLM-L12-v2` cross-lingual cosine | local CPU |
| Audio capture | `sounddevice` (WASAPI, live device switching) | local |
| UI | `PyQt6` — frameless, transparent, resizable overlay | local |
| Persistence | JSON file `history.json` (one entry per khutba) | local |
| Language | Python 3.11+ | — |

---

## Project Structure

```
src/
├── ui.py                       # entry point
├── backend/
│   ├── audio.py                # mic + Silero VAD, device enumeration/switching
│   ├── scribe_batch.py         # ElevenLabs Scribe v2 STT
│   ├── quran_match.py          # verse matcher → canonical German (data/*.json)
│   ├── gemma_translate.py      # Ollama client, ordered endpoints (remote 12B → local 4B)
│   ├── local_translate.py      # Helsinki Marian MT second candidate
│   ├── rerank.py               # cross-lingual QE reranker
│   ├── postprocess.py          # terminology fixes + language-leak guard
│   ├── history.py              # JSON persistence + khutba schema
│   └── workers.py              # QThread workers + buffering logic
└── frontend/
    ├── main_window.py          # overlay window (display pacing, fade, resize)
    └── history_window.py       # khutba browser window

data/          # Quran Arabic + Bubenheim & Elyas German (6236 verses)
notebooks/     # remote_ollama.ipynb — Colab/Kaggle 12B server
tests/         # eval_translation.py (47-case accuracy harness) + unit tests
docs/          # translation_research.md — datasets, decisions, eval history
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

# optional — remote 12B booster (see notebooks/remote_ollama.ipynb)
REMOTE_OLLAMA_HOST=https://xxxx.trycloudflare.com
REMOTE_MODEL_ID=translategemma:12b
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
| `S` or ⚙ icon | Toggle settings panel (opacity, font sizes, Scribe model, **microphone picker**) |
| `H` or 🕘 icon | Open Khutba history window |
| `M` | Mute (drops audio before it reaches the cloud — saves tokens) |
| `Esc` or ✕ icon | Quit (cleanly closes Ollama session, audio threads) |
| Click + drag center | Move the overlay |
| Drag edge / corner | Resize the overlay |

Tip: with **VB-Audio Virtual Cable** installed you can pick "CABLE Output" as the microphone and feed recorded khutbahs (YouTube etc.) directly into the pipeline — enable Windows "Listen to this device" on CABLE Output to hear it yourself.

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

## Accuracy evaluation

```powershell
python tests\eval_translation.py           # full pipeline (matcher + MT + rerank)
python tests\eval_translation.py --no-matcher --tag baseline
python -m pytest tests\test_quran_match.py -q
```

47 cases (Quran verses, partial recitation, khutbah formulas), metric = multilingual-MiniLM cosine vs reference German:

| Config | Overall | Quran | Khutbah |
|---|---|---|---|
| Baseline (4B only) | 0.744 | 0.756 | 0.739 |
| Local pipeline (matcher + rerank) | 0.950 | 1.000 | 0.883 |
| + remote 12B | **0.958** | **1.000** | 0.901 |

Full history in `tests/eval_results.json`, research notes in `docs/translation_research.md`.

---

## Motivation

This project was born out of a real need — helping German-speaking members of a local mosque follow the Friday Khutbah in their language. MinbarAI is a step toward making mosque services more accessible and inclusive.

---

## Author

Built with ❤️ for the community.
