# MinbarAI — Architecture (Mermaid)

Bottlenecks shown in **red**. Source: [architecture.mmd](architecture.mmd).

```mermaid
flowchart TD
    Mic([🎤 Imam Microphone<br/>Arabic Khutbah]):::input

    subgraph Capture["Audio Capture (main thread)"]
        SD[sounddevice InputStream<br/>16kHz mono, 256ms blocks]
        CB[audio_callback<br/>RMS energy VAD<br/>threshold=0.02]
        BUF[(speech_buffer<br/>list of np arrays)]
        SD --> CB --> BUF
    end

    DECIDE{End of sentence?<br/>silence ≥ 1.5s<br/>OR len ≥ 15s}
    BUF --> DECIDE

    Q[[audio_queue<br/>queue.Queue]]:::queue
    DECIDE -->|yes: flush| Q
    DECIDE -->|no| BUF

    subgraph Worker["Background Thread (process_audio)"]
        WHISPER[WhisperModel<br/>small / int8 / CPU<br/>language=ar]:::bottleneck
        TXT[Arabic text segments]
        TRANS[Translator<br/>translate lib<br/>MyMemory API ar→de]:::bottleneck
        GER[German text]
        WHISPER --> TXT --> TRANS --> GER
    end
    Q --> WHISPER

    subgraph Display["Display Layer"]
        UPD[update_display ar, de]
        STATE[(in-memory state<br/>+ display.html)]
        HTTP[HTTP server :8080<br/>meta refresh 2s]:::bottleneck
        UPD --> STATE --> HTTP
    end
    GER --> UPD

    Browser([🖥️ Browser / 2nd monitor<br/>Congregation reads German]):::output
    HTTP --> Browser

    classDef input fill:#1e3a8a,stroke:#60a5fa,color:#fff
    classDef output fill:#14532d,stroke:#4ade80,color:#fff
    classDef queue fill:#78350f,stroke:#fbbf24,color:#fff
    classDef bottleneck fill:#7f1d1d,stroke:#f87171,color:#fff,stroke-width:3px
```

## Bottleneck breakdown

| Node | Why slow | Fix |
|---|---|---|
| WhisperModel | `small` int8 on CPU, 3–8s per sentence | `tiny` model, or GPU+`float16`, or streaming partials |
| Translator (MyMemory) | network call, not offline, rate-limited | local Marian-NMT `Helsinki-NLP/opus-mt-ar-de` or NLLB-200 |
| HTTP meta-refresh 2s | client polls every 2s, up to 2s lag | WebSocket / SSE push from server |

Other tunables (not bottlenecks but affect feel): `SILENCE_DURATION=1.5s` adds fixed lag before each translation; `MIN_BUFFER_SIZE=6` prevents short bursts.
