# MinbarAI Documentation

System docs for understanding pipeline + finding optimization points.

## Files

- [pipeline.md](pipeline.md) — ASCII pipeline diagram + latency budget + optimization targets
- [architecture.mmd](architecture.mmd) — Mermaid source. Bottlenecks marked red
- [architecture.md](architecture.md) — Mermaid embedded for GitHub/markdown viewers

## Bottlenecks (red nodes)

1. **Whisper STT** — CPU `small` model = 3–8s per utterance. Switch `tiny` or GPU.
2. **Translation API** — network roundtrip, breaks offline guarantee. Use local Marian/NLLB.
3. **HTTP meta-refresh 2s** — adds up to 2s display lag. Use WebSocket/SSE push.

See [pipeline.md](pipeline.md) for full latency budget.
