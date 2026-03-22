from faster_whisper import WhisperModel
from translate import Translator
import sys

# Load Whisper model
print("Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio_samples/test.mp3"

# Create translator (Arabic to German)
translator = Translator(from_lang="ar", to_lang="de")

print(f"Transcribing + Translating: {audio_path}")
print("=" * 60)

# Transcribe and translate segment by segment
segments, info = model.transcribe(audio_path, language="ar")

for segment in segments:
    arabic_text = segment.text.strip()
    
    if arabic_text:
        # Translate this segment
        german_text = translator.translate(arabic_text)
        
        # Show both
        print(f"\n🇸🇦 Arabic: {arabic_text}")
        print(f"🇩🇪 German: {german_text}")

print("\n" + "=" * 60)
print("✓ Done!")
