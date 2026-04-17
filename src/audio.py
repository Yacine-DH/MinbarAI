import queue
import threading
import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.02
SILENCE_DURATION = 0.8
MAX_SENTENCE_SECONDS = 15
MIN_CHUNKS = 3
BLOCK_SIZE = int(SAMPLE_RATE * 0.256)  # 256ms per chunk

audio_queue = queue.Queue()

_buffer = []
_last_speech_time = [None]

def _callback(indata, frames, time_info, status):
    audio = indata.copy().flatten().astype(np.float32)
    energy = np.sqrt(np.mean(audio**2))
    now = time.time()

    if energy > VAD_THRESHOLD:
        _buffer.append(audio)
        _last_speech_time[0] = now

        if len(_buffer) * frames >= MAX_SENTENCE_SECONDS * SAMPLE_RATE:
            audio_queue.put(np.concatenate(_buffer))
            _buffer.clear()
            _last_speech_time[0] = None
    else:
        if _buffer and _last_speech_time[0] is not None:
            silence = now - _last_speech_time[0]
            if len(_buffer) >= MIN_CHUNKS and silence >= SILENCE_DURATION:
                audio_queue.put(np.concatenate(_buffer))
                _buffer.clear()
                _last_speech_time[0] = None

def start(device: int = None):
    def _run():
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device,
            blocksize=BLOCK_SIZE,
            callback=_callback
        ):
            while True:
                time.sleep(1)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
