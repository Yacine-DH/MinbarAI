"""Standalone Gemini sanity test. Prints exact error if it fails."""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print(f"Key prefix: {key[:10] if key else 'MISSING'}...")

client = genai.Client(api_key=key)

print("Trying gemini-2.5-flash-lite...")
try:
    r = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Translate to German: مرحبا",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=100),
    )
    print("OK:", r.text)
except Exception as exc:
    print(f"FAIL: {type(exc).__name__}: {exc}")

print("\nListing available models...")
try:
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(f"  - {m.name}")
except Exception as exc:
    print(f"List failed: {exc}")
