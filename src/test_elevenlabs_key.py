"""Verify .env loads ElevenLabs API key correctly."""
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("ELEVENLABS_API_KEY")

if not key:
    print("FAIL: ELEVENLABS_API_KEY not found. Check .env file exists in project root.")
elif key == "PASTE_YOUR_KEY_HERE":
    print("FAIL: Placeholder still present. Paste real key into .env.")
elif not key.startswith("sk_"):
    print(f"WARN: Key loaded but does not start with 'sk_'. Got: {key[:6]}...")
else:
    print(f"OK: Key loaded. Starts with: {key[:6]}... (length {len(key)})")
