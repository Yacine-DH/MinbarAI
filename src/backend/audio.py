import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch

SAMPLE_RATE = 16000
BLOCK_SIZE = 512          # Silero VAD requirement at 16kHz (32ms chunks)
SPEECH_THRESHOLD = 0.5    # probability cutoff
SILENCE_DURATION = 0.8    # seconds of silence to flush segment
MAX_SENTENCE_SECONDS = 15
MIN_CHUNKS = 3            # ~96ms minimum to avoid noise bursts

audio_queue = queue.Queue()

_vad_model = None
_buffer = []
_last_speech_time = [None]
_muted = False


def set_muted(value: bool):
    """When muted, the VAD callback drops incoming audio so nothing reaches
    Scribe/Gemma — saves tokens while there's noise or no one is speaking."""
    global _muted
    _muted = bool(value)
    if _muted:
        _buffer.clear()
        _last_speech_time[0] = None
        if _vad_model is not None:
            _vad_model.reset_states()


def is_muted() -> bool:
    return _muted


def _load_vad():
    global _vad_model
    _vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        verbose=False
    )


def _callback(indata, frames, time_info, status):
    if _vad_model is None or _muted:
        return

    audio = indata.copy().flatten().astype(np.float32)
    chunk_tensor = torch.from_numpy(audio)

    with torch.no_grad():
        speech_prob = _vad_model(chunk_tensor, SAMPLE_RATE).item()

    now = time.time()

    if speech_prob >= SPEECH_THRESHOLD:
        _buffer.append(audio)
        _last_speech_time[0] = now

        if len(_buffer) * BLOCK_SIZE >= MAX_SENTENCE_SECONDS * SAMPLE_RATE:
            audio_queue.put(np.concatenate(_buffer))
            _buffer.clear()
            _last_speech_time[0] = None
            _vad_model.reset_states()
    else:
        if _buffer and _last_speech_time[0] is not None:
            silence = now - _last_speech_time[0]
            if len(_buffer) >= MIN_CHUNKS and silence >= SILENCE_DURATION:
                audio_queue.put(np.concatenate(_buffer))
                _buffer.clear()
                _last_speech_time[0] = None
                _vad_model.reset_states()


def start(device=None):
    _load_vad()

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
