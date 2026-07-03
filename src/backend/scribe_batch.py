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

# Active Scribe model. HTTP convert endpoint accepts:
#   "scribe_v1"              — legacy, cheapest
#   "scribe_v1_experimental" — experimental v1 variant
#   "scribe_v2"              — current default, best accuracy
# NOTE: "scribe_v2_realtime" is only available on the realtime websocket API,
# not this HTTP batch endpoint — using it here returns HTTP 400.
# Override at runtime via SCRIBE_MODEL in .env.
SCRIBE_MODEL = os.getenv("SCRIBE_MODEL", "scribe_v2")

AVAILABLE_MODELS = ("scribe_v2", "scribe_v1", "scribe_v1_experimental")


def set_model(name: str):
    """Swap the active Scribe model at runtime (UI-driven)."""
    global SCRIBE_MODEL
    if name not in AVAILABLE_MODELS:
        print(f"[scribe_batch] ignoring unknown model: {name!r}", flush=True)
        return
    SCRIBE_MODEL = name
    print(f"[scribe_batch] model switched → {SCRIBE_MODEL}", flush=True)


def get_model() -> str:
    return SCRIBE_MODEL


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
    print(f"[scribe_batch] ready (model={SCRIBE_MODEL})", flush=True)


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
            model_id=SCRIBE_MODEL,
            language_code="ara",
        )
        return (result.text or "").strip()
    except Exception as exc:
        print(f"[scribe_batch] error: {exc}", flush=True)
        return ""
