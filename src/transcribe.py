import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe(audio: np.ndarray) -> str:
    segments, _ = model.transcribe(audio, language="ar")
    return "".join(segment.text for segment in segments).strip()
