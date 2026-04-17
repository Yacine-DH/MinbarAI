# 🕌 MinbarAI

> Real-time Arabic to German speech translation — built for the mosque community.

---

## What is MinbarAI?

MinbarAI listens to the Imam's Khutbah (sermon) in Arabic and displays a live German translation as a desktop overlay for the congregation. No delays, no manual work — just spoken Arabic in, German text out.

Runs entirely offline and locally on a standard Windows PC. No API keys, no cloud, no subscription.

---

## How it works

```
Microphone → VAD → Whisper (local, Arabic STT) → MyMemory API → PyQt6 Overlay
```

1. A microphone captures the Imam's voice
2. Energy-based Voice Activity Detection (VAD) detects sentence boundaries
3. `faster-whisper` transcribes the Arabic speech to text locally on CPU
4. The Arabic text is sent to the MyMemory translation API (free, no signup)
5. The German translation appears live in a transparent desktop overlay

---

## Tech Stack

| Component | Technology |
|---|---|
| Speech-to-Text | `faster-whisper` (Whisper `tiny` model, runs locally) |
| Translation | MyMemory API via `requests` (free, no API key) |
| Audio capture | `sounddevice` |
| UI | `PyQt6` — transparent always-on-top overlay |
| Language | Python 3.11+ |

---

## Project Status

- [x] Project setup & environment
- [x] Arabic speech-to-text with faster-whisper (local)
- [x] Arabic → German translation via MyMemory API
- [x] Real-time microphone input with VAD
- [x] PyQt6 transparent overlay (always on top, draggable)
- [x] User controls — opacity, font size (press `S`)
- [ ] Mosque deployment

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Yacine-DH/MinbarAI.git
cd MinbarAI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows cmd
source venv/Scripts/activate  # Git Bash

# Install dependencies
pip install -r requirements.txt
```

### Find your microphone device ID

```bash
python src/list_devices.py
```

### Run

```bash
python src/main.py <device_id>
# Example:
python src/main.py 4
```

### Controls

| Key | Action |
|---|---|
| `S` | Toggle settings panel (opacity, font size) |
| `Escape` | Quit |
| Click + drag | Move the overlay anywhere on screen |

---

## Motivation

This project was born out of a real need — helping German-speaking members of a local mosque follow the Friday Khutbah in their language. MinbarAI is a step toward making mosque services more accessible and inclusive.

---

## Author

Built with ❤️ for the community.
