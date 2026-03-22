import sounddevice as sd
import numpy as np
import queue
import threading
import sys
from faster_whisper import WhisperModel
from translate import Translator
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# Settings
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.03  # Much higher threshold - only loud speech (was 0.005)
CHUNK_SECONDS = 3  # Accumulate 3 seconds of detected speech before processing

# Debug mode - show every frame
DEBUG = True

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
speech_buffer = []  # Buffer for accumulating speech frames
frame_count = [0]  # Track frames

def audio_callback(indata, frames, time, status):
    frame_count[0] += 1
    
    if status:
        print(f"\n📢 Audio status: {status}", flush=True)
    
    # Convert to mono if needed
    audio_data = indata.copy().flatten().astype(np.float32)
    
    # Calculate energy (RMS) for voice activity detection
    energy = np.sqrt(np.mean(audio_data**2))
    
    # Show every frame to keep output flowing
    if frame_count[0] % 5 == 0:  # Every 5 frames
        if energy > VAD_THRESHOLD:
            bar = "█" * min(int(energy * 50), 15)
            print(f"🎤 VOICE: {bar} (Energy: {energy:.4f})", flush=True)
        else:
            print(f"🔇 (listening...)", end="\r", flush=True)
    
    if energy > VAD_THRESHOLD:  # Speech detected
        speech_buffer.append(audio_data)
        
        # When buffer reaches ~3 seconds of speech, queue it for processing
        if len(speech_buffer) >= int(CHUNK_SECONDS * SAMPLE_RATE / frames):
            print(f"\n⏳ BUFFERED {len(speech_buffer)} chunks - sending to Whisper...\n", flush=True)
            audio_queue.put(np.concatenate(speech_buffer))
            speech_buffer.clear()
    else:
        # If we have buffered speech and now have silence, process it immediately
        if speech_buffer:
            print(f"\n⏳ SILENCE DETECTED - sending buffered {len(speech_buffer)} chunks to Whisper...\n", flush=True)
            audio_queue.put(np.concatenate(speech_buffer))
            speech_buffer.clear()

def process_audio():
    print("\n[Started background processor]", flush=True)
    proc_count = [0]
    
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=10)
            proc_count[0] += 1
            
            print(f"\n{'='*60}")
            print(f"🔍 PROCESSING AUDIO #{proc_count[0]} WITH WHISPER", flush=True)
            print(f"{'='*60}\n", flush=True)
            
            segments, _ = model.transcribe(audio_chunk, language="ar")
            
            found_speech = False
            for segment in segments:
                arabic = segment.text.strip()
                if arabic:
                    found_speech = True
                    print(f"✅ DETECTED ARABIC: {arabic}\n", flush=True)
                    
                    german = translator.translate(arabic)
                    
                    # Properly reshape Arabic text for display (connected letters, right-to-left)
                    reshaped_arabic = reshape(arabic)
                    display_arabic = get_display(reshaped_arabic)
                    
                    print(f"📤 YOUR INPUT (Arabic):\n    {display_arabic}\n", flush=True)
                    print(f"📥 TRANSLATION (German):\n    {german}\n", flush=True)
                    print(f"{'='*60}\n", flush=True)
            
            if not found_speech:
                print(f"❌ No speech detected in chunk #{proc_count[0]}\n", flush=True)
                
        except queue.Empty:
            print(".", end="", flush=True)
            continue
        except Exception as e:
            print(f"\n⚠️  Error: {e}", flush=True)

# Run processing in background thread
thread = threading.Thread(target=process_audio, daemon=True)
thread.start()

# Start listening
try:
    print("\n" + "="*60, flush=True)
    print("🎤 MINBAR AI - REAL-TIME ARABIC → GERMAN TRANSLATOR", flush=True)
    print("="*60, flush=True)
    print("Listening on: JBL Tune 720BT (Device #18)", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Use 256ms blocks for responsive detection, but never block during processing
    blocksize = int(SAMPLE_RATE * 0.256)  # 256ms blocks
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback,
                        blocksize=blocksize, device=DEVICE):
        print("✅ Microphone READY", flush=True)
        print("\n🎙️  SPEAK ARABIC - WATCH BELOW FOR WHAT YOU SAY AND THE TRANSLATION:\n", flush=True)
        print("="*60 + "\n", flush=True)
        
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\n\n✓ Stopped!", flush=True)
except Exception as e:
    print(f"❌ Microphone error: {e}", flush=True)
    sys.exit(1)