# MinbarAI — Architecture (Mermaid)

Cloud calls shown in **amber**, local models in **green**. Source: [architecture.mmd](architecture.mmd).

```mermaid
flowchart TD
    Mic([🎤 Imam Microphone<br/>Arabic Khutbah]):::input

    subgraph Capture["Audio Capture"]
        SD[sounddevice InputStream<br/>16kHz mono, 512-sample blocks]
        VAD[Silero VAD<br/>speech prob ≥ 0.5<br/>silence flush 0.8s]:::local
        SD --> VAD
    end

    Q1[[audio_queue<br/>queue.Queue]]:::queue
    VAD -->|speech segment| Q1

    subgraph Transcription["TranscribeWorker (QThread)"]
        SCRIBE[ElevenLabs Scribe v2<br/>scribe_v2 · ara<br/>cloud HTTP]:::cloud
    end
    Q1 --> SCRIBE

    Q2[[arabic_queue<br/>queue.Queue]]:::queue
    SCRIBE -->|Arabic text| Q2

    subgraph Translation["TranslateWorker (QThread)"]
        BUF[Buffer<br/>flush on:<br/>punct OR ≥8 words OR 0.7s silence]
        GEMMA[TranslateGemma 4B<br/>Ollama localhost:11434]:::local
        HELSINKI[Helsinki opus-mt-ar-de<br/>Marian MT, CPU]:::local
        BUF --> GEMMA
        GEMMA -.->|fail| HELSINKI
    end
    Q2 --> BUF

    subgraph Persist["Persistence"]
        HIST[(history.json<br/>khutbas[])]:::store
    end
    GEMMA -->|de| HIST
    HELSINKI -->|de| HIST

    subgraph UI["MainWindow"]
        PACE[Display pacing<br/>≥ 3s on screen]
        FADE[Fade transition<br/>220ms × 2]
        LBL[arabic_label + german_label]
        PACE --> FADE --> LBL
    end
    GEMMA --> PACE
    HELSINKI --> PACE

    HISTWIN([🕘 HistoryWindow<br/>khutba browser]):::output
    HIST --> HISTWIN

    Overlay([🖥️ Transparent overlay<br/>congregation reads German]):::output
    LBL --> Overlay

    classDef input fill:#1e3a8a,stroke:#60a5fa,color:#fff
    classDef output fill:#14532d,stroke:#4ade80,color:#fff
    classDef queue fill:#78350f,stroke:#fbbf24,color:#fff
    classDef cloud fill:#7c2d12,stroke:#fb923c,color:#fff,stroke-width:3px
    classDef local fill:#064e3b,stroke:#34d399,color:#fff,stroke-width:2px
    classDef store fill:#3f3f46,stroke:#a1a1aa,color:#fff
```

## Components

| Layer | Module | Where it runs |
|---|---|---|
| Audio capture | `backend/audio.py` | local CPU |
| VAD | Silero VAD (PyTorch) | local CPU |
| STT | ElevenLabs Scribe v2 | **cloud** |
| Translation primary | TranslateGemma 4B via Ollama | local |
| Translation fallback | Helsinki Marian MT | local CPU |
| Persistence | JSON file | local disk |
| UI | PyQt6 overlay | local |

## Cloud vs Local

| Cloud (requires internet + API key) | Local |
|---|---|
| ElevenLabs Scribe v2 | Silero VAD |
| | TranslateGemma 4B (Ollama) |
| | Helsinki Marian MT |

If Ollama / Gemma is unavailable, Helsinki kicks in automatically. If ElevenLabs fails, transcription stops (no local STT fallback in the current pipeline).

## Threading model

| Thread | Owner | Job |
|---|---|---|
| Main (Qt event loop) | `ui.py` | Render, mouse, keyboard, signals |
| sounddevice callback | OS audio driver | Push raw audio into VAD |
| VAD worker | `backend/audio.py` | Run Silero, push segments to `audio_queue` |
| `TranscribeWorker` (QThread) | `backend/workers.py` | Pull from `audio_queue`, call Scribe, push to `arabic_queue` |
| `TranslateWorker` (QThread) | `backend/workers.py` | Pull from `arabic_queue`, buffer, call Gemma/Helsinki, emit result |
| Loader threads (3×) | `MainWindow.__init__` | Lazy-load Helsinki, Gemma client, ElevenLabs client at startup |

All signals between QThreads and `MainWindow` are connected via Qt's `pyqtSignal` so updates land on the main thread safely.
