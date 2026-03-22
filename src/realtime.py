import sounddevice as sd
import numpy as np
import queue
import threading
import sys
from faster_whisper import WhisperModel
from translate import Translator

# Settings
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5  # captures 5 seconds at a time

# Get device from command line argument
DEVICE = None
if len(sys.argv) > 1:
    try:
        DEVICE = int(sys.argv[1])
        print(f"Using device #{DEVICE}", flush=True)
    except ValueError:
        print(f"Invalid device number. Usage: python realtime.py <device_id>", flush=True)
        print(f"Run 'python list_devices.py' to see available devices", flush=True)
        sys.exit(1)
else:
    DEVICE = sd.default.device[0]  # Use default if not specified
    print(f"Using default device #{DEVICE}", flush=True)

# Load models
print("Loading models...", flush=True)
sys.stdout.flush()

try:
    model = WhisperModel("small", device="cpu", compute_type="int8")
    translator = Translator(from_lang="ar", to_lang="de")
    print("✓ Models loaded!", flush=True)
except Exception as e:
    print(f"❌ Error loading models: {e}", flush=True)
    sys.exit(1)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}", flush=True)
    audio_queue.put(indata.copy())

def process_audio():
    print("Audio processing thread started", flush=True)
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=10)
            audio_flat = audio_chunk.flatten().astype(np.float32)
            
            print("Transcribing...", flush=True)
            segments, _ = model.transcribe(audio_flat, language="ar")
            
            for segment in segments:
                arabic = segment.text.strip()
                if arabic:
                    german = translator.translate(arabic)
                    print(f"\n🇸🇦 AR: {arabic}")
                    print(f"🇩🇪 DE: {german}\n", flush=True)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Error processing audio: {e}", flush=True)

# Run processing in background thread
thread = threading.Thread(target=process_audio, daemon=True)
thread.start()

# Start listening
try:
    print("\n🎤 Starting microphone...", flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback,
                        blocksize=SAMPLE_RATE * CHUNK_SECONDS, device=DEVICE):
        print(f"✓ Ready! Listening for Arabic on device #{DEVICE}...", flush=True)
        print("Press Ctrl+C to stop\n", flush=True)
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\n\n✓ Stopped!", flush=True)
except Exception as e:
    print(f"❌ Microphone error: {e}", flush=True)
    sys.exit(1)