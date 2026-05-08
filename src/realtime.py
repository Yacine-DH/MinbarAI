import asyncio
import base64
import os
import queue
import sys

import sounddevice as sd
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from dotenv import load_dotenv
from elevenlabs import (
    AudioFormat,
    ElevenLabs,
    RealtimeAudioOptions,
    RealtimeEvents,
)
from translate import Translator

from display import start_http_server, update_display

load_dotenv()

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.256
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_SECONDS)

DEVICE = None
if len(sys.argv) > 1:
    try:
        DEVICE = int(sys.argv[1])
        print(f"Using device #{DEVICE}", flush=True)
    except ValueError:
        print("Invalid device number. Usage: python realtime.py <device_id>", flush=True)
        print("Run 'python list_devices.py' to see available devices", flush=True)
        sys.exit(1)
else:
    DEVICE = sd.default.device[0]
    print(f"Using default device #{DEVICE}", flush=True)

API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY or API_KEY == "PASTE_YOUR_KEY_HERE":
    print("❌ ELEVENLABS_API_KEY missing. Set it in .env", flush=True)
    sys.exit(1)

audio_q: queue.Queue[bytes] = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"\n📢 Audio status: {status}", flush=True)
    audio_q.put(bytes(indata))


async def stream_to_scribe():
    eleven = ElevenLabs(api_key=API_KEY)
    translator = Translator(from_lang="ar", to_lang="de")

    update_display("في انتظار الخطبة...", "Warte auf die Khutbah...")

    print("Connecting to Scribe v2 Realtime...", flush=True)
    connection = await eleven.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=SAMPLE_RATE,
        )
    )

    stop_event = asyncio.Event()

    def on_session_started(data):
        print(f"✓ Session started: {data}", flush=True)

    def on_partial_transcript(data):
        text = (data.get("text") or "").strip()
        if text:
            shown = get_display(reshape(text))
            print(f"… {shown}", flush=True)

    def on_committed_transcript(data):
        arabic = (data.get("text") or "").strip()
        if not arabic:
            return
        try:
            german = translator.translate(arabic)
        except Exception as exc:
            german = f"[translation error: {exc}]"

        shown = get_display(reshape(arabic))
        print("\n" + "=" * 60, flush=True)
        print(f"✅ AR: {shown}", flush=True)
        print(f"📥 DE: {german}", flush=True)
        print("=" * 60 + "\n", flush=True)

        update_display(arabic, german)

    def on_error(error):
        print(f"⚠️  Scribe error: {error}", flush=True)
        stop_event.set()

    def on_close():
        print("Connection closed", flush=True)
        stop_event.set()

    connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial_transcript)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed_transcript)
    connection.on(RealtimeEvents.ERROR, on_error)
    connection.on(RealtimeEvents.CLOSE, on_close)

    loop = asyncio.get_running_loop()

    async def pump_audio():
        while not stop_event.is_set():
            chunk = await loop.run_in_executor(None, audio_q.get)
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
            callback=audio_callback,
            blocksize=BLOCKSIZE,
            device=DEVICE,
        ):
            print("=" * 60, flush=True)
            print("🎤 MINBAR AI — Scribe v2 Realtime → German", flush=True)
            print("🌐 Web Display: http://localhost:8080/", flush=True)
            print("=" * 60, flush=True)
            print("✅ Microphone READY — speak Arabic\n", flush=True)
            await stop_event.wait()
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        await connection.close()


def main():
    start_http_server()
    try:
        asyncio.run(stream_to_scribe())
    except KeyboardInterrupt:
        print("\n\n✓ Stopped!", flush=True)


if __name__ == "__main__":
    main()
