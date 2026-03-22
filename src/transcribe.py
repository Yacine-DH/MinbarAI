from faster_whisper import WhisperModel

# Load the model (downloads ~150MB the first time, be patient)
model = WhisperModel("small", device="cpu", compute_type="int8")

# Point this to your test audio file
audio_path = "audio_samples/test.mp3"

print("Transcribing... please wait")

segments, info = model.transcribe(audio_path, language="ar")

print(f"Detected language: {info.language}")
print("--- Transcription ---")

for segment in segments:
    print(segment.text)
