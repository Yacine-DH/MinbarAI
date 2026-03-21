# 🕌 MinbarAI

> Real-time Arabic to German speech translation — built for the mosque community.

---

## What is MinbarAI?

MinbarAI listens to the Imam's Khutbah (sermon) in Arabic and displays a live German translation on a screen for the congregation. No delays, no manual work — just spoken Arabic in, German text out.

Built with a local AI speech recognition model (Whisper) and a translation API, MinbarAI runs entirely on a standard Windows PC with no special hardware required.

---

## How it works

```
Microphone → Whisper (Arabic STT) → Translation API → Live Display Screen
```

1. A microphone captures the Imam's voice
2. OpenAI's Whisper model transcribes Arabic speech to text locally
3. The text is sent to a translation API (DeepL / Google Translate)
4. The German translation appears live on a display screen

---

## Tech Stack

| Component | Technology |
|---|---|
| Speech-to-Text | `faster-whisper` (OpenAI Whisper) |
| Translation | DeepL API |
| Audio capture | `sounddevice` / `pyaudio` |
| Display | Python `tkinter` or browser-based UI |
| Language | Python 3.11+ |

---

## Project Status

🚧 Currently in development — Phase 1 (file transcription) in progress.

- [x] Project setup & environment
- [ ] Phase 1 — Transcribe Arabic audio file
- [ ] Phase 2 — Translate to German
- [ ] Phase 3 — Real-time microphone input
- [ ] Phase 4 — Live display screen
- [ ] Phase 5 — Mosque deployment

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourname/minbar-ai.git
cd minbar-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Motivation

This project was born out of a real need — helping German-speaking members of a local mosque follow the Friday Khutbah in their language. MinbarAI is a first step toward making mosque services more accessible and inclusive.

---

## Author

Built with ❤️ for the community.