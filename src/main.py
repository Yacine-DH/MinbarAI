"""
MinbarAI - Arabic to German Speech Translation
Complete pipeline: Transcribe Arabic audio → Translate to German
"""
import sys
import os
from transcribe import *
from faster_whisper import WhisperModel
from translate import Translator

def main(audio_file: str = "audio_samples/test.mp3"):
    """
    Main pipeline: transcribe Arabic audio and translate to German.
    
    Args:
        audio_file: Path to the audio file to process
    """
    print("=" * 60)
    print("MinbarAI - Arabic to German Speech Translation")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        return False
    
    # Load model
    print(f"\n📋 Loading Whisper model...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    # Create translator
    translator = Translator(from_lang="ar", to_lang="de")
    
    # Transcribe and translate
    print(f"🎤 Transcribing Arabic audio: {audio_file}")
    print("-" * 60)
    
    segments, info = model.transcribe(audio_file, language="ar")
    print(f"Detected language: {info.language}\n")
    
    for segment in segments:
        arabic_text = segment.text.strip()
        
        if arabic_text:
            # Translate
            german_text = translator.translate(arabic_text)
            
            # Display
            print(f"🇸🇦 {arabic_text}")
            print(f"🇩🇪 {german_text}\n")
    
    print("=" * 60)
    print("✓ Translation complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Use command line argument if provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio_samples/test.mp3"
    
    success = main(audio_file)
    sys.exit(0 if success else 1)
