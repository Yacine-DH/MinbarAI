import threading
import numpy as np
from faster_whisper import WhisperModel

_model = None
_lock = threading.Lock()
_ready = threading.Event()


def load():
    global _model
    try:
        print("[transcribe] loading whisper tiny...", flush=True)
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("[transcribe] warmup...", flush=True)
        _model.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="ar",
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
        )
        _ready.set()
        print("[transcribe] ready.", flush=True)
    except Exception as e:
        import traceback
        print(f"[transcribe] LOAD FAILED: {e}", flush=True)
        traceback.print_exc()


def transcribe(audio: np.ndarray) -> str:
    _ready.wait()
    with _lock:
        segments, _ = _model.transcribe(
            audio,
            language="ar",
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return "".join(s.text for s in segments).strip()
