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


_stream = None
_stream_lock = threading.Lock()
_current_device = None


def list_input_devices():
    """[(index, name), ...] of input-capable devices, one host API only.

    Windows exposes every device 3-4x (MME, DirectSound, WASAPI, ...);
    WASAPI is preferred — full device names, lowest latency."""
    apis = sd.query_hostapis()
    preferred = next(
        (i for i, a in enumerate(apis) if "WASAPI" in a["name"]),
        next((i for i, a in enumerate(apis) if a["devices"]), 0),
    )
    devices = []
    for idx, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["hostapi"] == preferred:
            devices.append((idx, d["name"]))
    return devices


def current_device():
    return _current_device


def default_input_index():
    """System default input mapped onto the list_input_devices() host API
    (default may point at an MME entry whose name is truncated to 31 chars)."""
    try:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None
        name = sd.query_devices(idx)["name"]
        for d, n in list_input_devices():
            if n == name or n.startswith(name) or name.startswith(n):
                return d
    except Exception:
        pass
    return None


def _open_stream(device):
    global _stream, _current_device
    with _stream_lock:
        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None
        extra = None
        if device is not None:
            api_name = sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
            if "WASAPI" in api_name:
                # shared-mode WASAPI refuses non-native sample rates unless
                # PortAudio is allowed to convert (we need 16 kHz for Silero)
                extra = sd.WasapiSettings(auto_convert=True)
        _stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device,
            blocksize=BLOCK_SIZE,
            callback=_callback,
            extra_settings=extra,
        )
        _stream.start()
        _current_device = device
        name = sd.query_devices(device)["name"] if device is not None else "default"
        print(f"[audio] input stream on '{name}'", flush=True)


def switch_device(device) -> bool:
    """Swap the input device live. Returns False (and keeps the old device
    if possible) when the new one cannot be opened."""
    _buffer.clear()
    _last_speech_time[0] = None
    if _vad_model is not None:
        _vad_model.reset_states()
    old = _current_device
    try:
        _open_stream(device)
        return True
    except Exception as exc:
        print(f"[audio] failed to open device {device}: {exc}", flush=True)
        try:
            _open_stream(old)
        except Exception:
            pass
        return False


def start(device=None):
    _load_vad()
    _open_stream(device)
