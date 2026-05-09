"""Send VAD-chunked audio to ElevenLabs Scribe v2 (batch HTTP API)."""
import io
import os
import threading
import wave

import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

SAMPLE_RATE = 16000

_client = None
_ready = threading.Event()


def load():
    global _client
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print("[scribe_batch] ELEVENLABS_API_KEY missing", flush=True)
        return
    _client = ElevenLabs(api_key=api_key)
    _ready.set()
    print("[scribe_batch] ready (model=scribe_v2)", flush=True)


def transcribe(audio: np.ndarray) -> str:
    _ready.wait()
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    buf.name = "chunk.wav"

    try:
        result = _client.speech_to_text.convert(
            file=buf,
            model_id="scribe_v2",
            language_code="ara",
        )
        return (result.text or "").strip()
    except Exception as exc:
        print(f"[scribe_batch] error: {exc}", flush=True)
        return ""
