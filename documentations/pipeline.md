# MinbarAI — Pipeline Diagram

## High-Level Flow

```
┌─────────────┐
│  Imam mic   │  Arabic speech (Khutbah)
└──────┬──────┘
       │ analog audio
       ▼
┌─────────────────────┐
│  sounddevice        │  16kHz mono, 512-sample blocks (32ms)
│  InputStream        │
└──────┬──────────────┘
       │ float32 frames
       ▼
┌─────────────────────┐
│  Silero VAD         │  speech probability ≥ 0.5
│  (snakers4/         │  silence flush ≥ 0.8s
│   silero-vad)       │  max segment 15s
└──────┬──────────────┘
       │ speech segments (numpy float32)
       ▼
┌─────────────────────┐
│  audio_queue        │  thread-safe handoff
│  (queue.Queue)      │
└──────┬──────────────┘
       │
       ▼ TranscribeWorker (QThread)
┌─────────────────────┐
│  ElevenLabs Scribe  │  cloud HTTP API
│  v2  (model=        │  ~1-3s per segment
│  scribe_v2,         │  language_code="ara"
│  language=ara)      │
└──────┬──────────────┘
       │ Arabic transcript
       ▼
┌─────────────────────┐
│  arabic_queue       │  thread-safe handoff
└──────┬──────────────┘
       │
       ▼ TranslateWorker (QThread)
┌─────────────────────┐
│  Buffer             │  Flush triggers:
│                     │   - punctuation (. ? ! ؟)
│                     │   - ≥ 8 words
│                     │   - 0.7s silence (queue empty)
└──────┬──────────────┘
       │ combined Arabic string
       ▼
┌─────────────────────┐    fail    ┌─────────────────────┐
│  TranslateGemma 4B  │───────────►│  Helsinki opus-mt-  │
│  via Ollama         │            │  ar-de (Marian MT)  │
│  (local, http://    │            │  local CPU          │
│   localhost:11434)  │            │                     │
└──────┬──────────────┘            └──────────┬──────────┘
       │ German                               │ German
       └──────────────────┬───────────────────┘
                          ▼
              ┌─────────────────────────┐
              │  result signal          │  Qt signal/slot
              │  (arabic, german)       │
              └──────┬──────────────────┘
                     │
                     ▼
              ┌─────────────────────────┐
              │  history.json           │  append entry
              │  (current khutba)       │  + save_history()
              └──────┬──────────────────┘
                     │
                     ▼
              ┌─────────────────────────┐
              │  MainWindow.update_text │  display pacing:
              │                         │   - if shown ≥ 3s: swap
              │                         │   - else: queue pending
              └──────┬──────────────────┘
                     │
                     ▼
              ┌─────────────────────────┐
              │  Fade transition        │  220ms fade out
              │  (QGraphicsOpacity-     │  → setText
              │   Effect)               │  → 220ms fade in
              └──────┬──────────────────┘
                     │
                     ▼
              ┌─────────────────────────┐
              │  PyQt6 overlay          │  frameless, transparent,
              │  - german_label         │  always-on-top,
              │  - arabic_label         │  resizable, draggable
              └─────────────────────────┘
```

## Latency Budget (current)

| Stage | Time | Notes |
|---|---|---|
| Mic capture | ~32ms | 512-sample block at 16kHz |
| VAD silence wait | ~800ms | `SILENCE_DURATION` in `audio.py` |
| Scribe v2 (cloud) | ~1000–2500ms | network roundtrip + transcription |
| Translation buffer wait | up to ~700ms | until silence trigger or 8 words |
| TranslateGemma 4B | ~500–2000ms | local CPU/GPU inference |
| Display pacing lock | 0–3000ms | min display time gate (only delays swap, not data) |
| Fade transition | 440ms total | 220ms out + 220ms in |
| **Total perceived lag** | **~3–6s** | after sentence end |

## Khutba History

- Persistence: `history.json` at project root (gitignored)
- Schema:
  ```json
  {
    "khutbas": [
      {
        "id": "2026-05-10T14:30:00",
        "started_at": "2026-05-10T14:30:00",
        "ended_at": "2026-05-10T15:12:48",
        "entries": [
          { "ts": "2026-05-10T14:31:22", "ar": "...", "de": "..." }
        ]
      }
    ]
  }
  ```
- New khutba = new app session
- Auto-migrates legacy flat-list format

## Module map

| Module | Responsibility |
|---|---|
| `backend/audio.py` | mic + Silero VAD → `audio_queue` |
| `backend/scribe_batch.py` | Scribe v2 HTTP API wrapper |
| `backend/gemma_translate.py` | Ollama HTTP client (TranslateGemma 4B) |
| `backend/local_translate.py` | Helsinki Marian MT (fallback) |
| `backend/history.py` | load/save + legacy migration |
| `backend/workers.py` | `TranscribeWorker`, `TranslateWorker`, buffering |
| `frontend/main_window.py` | overlay + display pacing + fade + resize |
| `frontend/history_window.py` | khutba browser (combo box + entry list) |
| `ui.py` | entry point: `QApplication` + `MainWindow.show()` |
