# MinbarAI — Pipeline Diagram

## High-Level Flow

```
┌─────────────┐
│  Imam mic   │  Arabic speech (Khutbah)
└──────┬──────┘
       │ analog audio
       ▼
┌─────────────────────┐
│  sounddevice        │  16kHz mono, 256ms blocks
│  InputStream        │
└──────┬──────────────┘
       │ float32 frames
       ▼
┌─────────────────────┐
│  audio_callback     │  RMS energy calc
│  (VAD)              │  threshold 0.02
└──────┬──────────────┘
       │ speech chunks
       ▼
┌─────────────────────┐
│  speech_buffer      │  accumulate until:
│                     │   - 1.5s silence, OR
│                     │   - 15s max length
└──────┬──────────────┘
       │ complete utterance
       ▼
┌─────────────────────┐
│  audio_queue        │  thread-safe handoff
│  (queue.Queue)      │
└──────┬──────────────┘
       │
       ▼ background thread
┌─────────────────────┐
│  WhisperModel       │  ⚠️ BOTTLENECK
│  small / int8 / cpu │  ~3-8s per chunk
│  language="ar"      │
└──────┬──────────────┘
       │ Arabic text
       ▼
┌─────────────────────┐
│  Translator         │  ⚠️ BOTTLENECK
│  translate lib      │  network call
│  ar → de            │  MyMemory API
└──────┬──────────────┘
       │ German text
       ▼
┌─────────────────────┐
│  update_display()   │  in-mem state
│                     │  + display.html
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  HTTP server :8080  │  meta refresh 2s
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Browser / screen   │  congregation read
└─────────────────────┘
```

## Latency Budget (current)

| Stage | Time | Notes |
|---|---|---|
| Mic capture | ~256ms | block size |
| VAD wait (silence) | 1500ms | `SILENCE_DURATION` |
| Whisper STT | 3000–8000ms | **CPU bound, biggest cost** |
| Translation API | 200–800ms | network roundtrip |
| Display refresh | 0–2000ms | meta-refresh poll |
| **Total perceived lag** | **~5–12s** | after sentence end |

## Optimization Targets

1. **Whisper STT** — biggest win. Options: `tiny` model (5x faster, less accurate), GPU/CUDA, `faster-whisper` batched inference, streaming partial results.
2. **Translation** — local NMT (Marian/NLLB-200) removes network + works offline fully. Currently API-bound.
3. **Display refresh** — replace 2s meta-refresh with WebSocket/SSE push (instant update).
4. **VAD silence wait** — `SILENCE_DURATION=1.5s` adds fixed latency. Lower = faster but more cuts mid-sentence.
