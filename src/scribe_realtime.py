"""Streams microphone audio to ElevenLabs Scribe v2 Realtime via WebSocket.

Public interface:
    start(device=None) -> None    # starts mic + WebSocket in background thread
    partial_queue:   queue.Queue[str]   # in-progress transcripts (live)
    committed_queue: queue.Queue[str]   # finalized sentence transcripts
"""
import asyncio
import base64
import os
import queue
import threading

import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs import (
    AudioFormat,
    ElevenLabs,
    RealtimeAudioOptions,
    RealtimeEvents,
)

load_dotenv()

SAMPLE_RATE = 16000
BLOCKSIZE = int(SAMPLE_RATE * 0.256)

partial_queue: "queue.Queue[str]" = queue.Queue()
committed_queue: "queue.Queue[str]" = queue.Queue()

_audio_q: "queue.Queue[bytes]" = queue.Queue()


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[scribe] audio status: {status}", flush=True)
    _audio_q.put(bytes(indata))


async def _run(device):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print("[scribe] ELEVENLABS_API_KEY missing in .env", flush=True)
        return

    eleven = ElevenLabs(api_key=api_key)

    print("[scribe] connecting to scribe_v2_realtime...", flush=True)
    connection = await eleven.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=SAMPLE_RATE,
        )
    )

    stop_event = asyncio.Event()

    def on_session_started(data):
        print(f"[scribe] session started: {data}", flush=True)

    def on_partial(data):
        text = (data.get("text") or "").strip()
        if text:
            partial_queue.put(text)

    def on_committed(data):
        text = (data.get("text") or "").strip()
        if text:
            committed_queue.put(text)

    def on_error(error):
        print(f"[scribe] error: {error}", flush=True)
        stop_event.set()

    def on_close():
        print("[scribe] connection closed", flush=True)
        stop_event.set()

    connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
    connection.on(RealtimeEvents.ERROR, on_error)
    connection.on(RealtimeEvents.CLOSE, on_close)

    loop = asyncio.get_running_loop()

    async def pump_audio():
        while not stop_event.is_set():
            chunk = await loop.run_in_executor(None, _audio_q.get)
            b64 = base64.b64encode(chunk).decode("utf-8")
            await connection.send(
                {"audio_base_64": b64, "sample_rate": SAMPLE_RATE}
            )

    pump_task = asyncio.create_task(pump_audio())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=_audio_callback,
            blocksize=BLOCKSIZE,
            device=device,
        ):
            print("[scribe] mic streaming", flush=True)
            await stop_event.wait()
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        await connection.close()


def start(device=None):
    """Launch Scribe streaming in a background thread."""
    def _runner():
        try:
            asyncio.run(_run(device))
        except Exception as exc:
            print(f"[scribe] thread crashed: {exc}", flush=True)

    threading.Thread(target=_runner, daemon=True).start()
