import sounddevice as sd
import numpy as np
import queue
import threading
import sys
from faster_whisper import WhisperModel
from translate import Translator
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from display import update_display, start_http_server

# Settings
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.02  # Lower threshold slightly for better sensitivity
SILENCE_DURATION = 1.5  # Require 1.5 seconds of silence (natural pause)
MAX_SENTENCE_LENGTH = 15  # Maximum seconds to buffer before forcing translation
MIN_BUFFER_SIZE = 6  # Minimum chunks to buffer before checking for end-of-sentence

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
last_speech_time = [None]  # Track when the last speech was detected
frame_count = [0]  # Track frames
import time

def audio_callback(indata, frames, time_info, status):
    frame_count[0] += 1
    current_time = time.time()
    
    if status:
        print(f"\n📢 Audio status: {status}", flush=True)
    
    # Convert to mono if needed
    audio_data = indata.copy().flatten().astype(np.float32)
    
    # Calculate energy (RMS) for voice activity detection
    energy = np.sqrt(np.mean(audio_data**2))
    
    # Show every 40 frames to keep output flowing
    if frame_count[0] % 40 == 0:  # Every 40 frames (~10 seconds at 256ms blocks)
        if energy > VAD_THRESHOLD:
            bar = "█" * min(int(energy * 50), 15)
            print(f"🎤 LISTENING ({len(speech_buffer)} chunks buffered): {bar} (Energy: {energy:.4f})", flush=True)
    
    if energy > VAD_THRESHOLD:  # Speech detected
        speech_buffer.append(audio_data)
        last_speech_time[0] = current_time  # Update last speech time
        
        # Max buffer size: ~30 seconds of speech (force translation if too long)
        max_time = MAX_SENTENCE_LENGTH * SAMPLE_RATE
        if len(speech_buffer) * frames >= max_time:
            print(f"\n⏳ MAX SENTENCE LENGTH REACHED ({len(speech_buffer)} chunks) - Force sending to Whisper...\n", flush=True)
            audio_queue.put(np.concatenate(speech_buffer))
            speech_buffer.clear()
            last_speech_time[0] = None
    else:
        # Check if we have buffered speech and enough silence has passed
        if speech_buffer and last_speech_time[0] is not None:
            silence_duration = current_time - last_speech_time[0]
            
            # Only consider end-of-sentence if we have at least MIN_BUFFER_SIZE chunks
            if len(speech_buffer) >= MIN_BUFFER_SIZE and silence_duration >= SILENCE_DURATION:
                print(f"\n⏳ COMPLETE SENTENCE ({len(speech_buffer)} chunks, {silence_duration:.1f}s silence) - Sending to Whisper...\n", flush=True)
                audio_queue.put(np.concatenate(speech_buffer))
                speech_buffer.clear()
                last_speech_time[0] = None

def process_audio():
    print("\n[Started background processor]", flush=True)
    # Initialize with default display
    update_display("في انتظار الخطبة...", "Warte auf die Khutbah...")
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
                    
                    # Update the web display
                    update_display(arabic, german)
            
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

# Start HTTP server in background
start_http_server()

# Start listening
try:
    print("\n" + "="*60, flush=True)
    print("🎤 MINBAR AI - REAL-TIME ARABIC → GERMAN TRANSLATOR", flush=True)
    print("="*60, flush=True)
    print("Listening on: JBL Tune 720BT (Device #18)", flush=True)
    print("🌐 Web Display: http://localhost:8080/", flush=True)
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