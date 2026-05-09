"""Record N seconds from mic, send to Scribe v2 (batch), print transcript.

Usage:
    python src/test_scribe_v2.py            # default mic, 10 seconds
    python src/test_scribe_v2.py 18         # device 18, 10s
    python src/test_scribe_v2.py 18 15      # device 18, 15s
"""
import io
import os
import sys
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

SAMPLE_RATE = 16000

device = None
seconds = 10
if len(sys.argv) > 1:
    device = int(sys.argv[1])
if len(sys.argv) > 2:
    seconds = int(sys.argv[2])

api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    print("ELEVENLABS_API_KEY missing")
    sys.exit(1)

print(f"Recording {seconds}s from device={device}...")
audio = sd.rec(
    int(seconds * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    device=device,
)
sd.wait()
print("Recording done.")

# Wrap PCM as in-memory WAV
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())
buf.seek(0)
buf.name = "clip.wav"

print("Sending to Scribe v2...")
client = ElevenLabs(api_key=api_key)
result = client.speech_to_text.convert(
    file=buf,
    model_id="scribe_v2",
    language_code="ara",
)

print("\n=== TRANSCRIPT ===")
print(result.text)
print("==================")
