"""Gemini Flash-Lite Arabic→German translator. Available as alt option."""
import os
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_ID = "gemini-2.5-flash-lite"

_client = None
_ready = threading.Event()

PROMPT_PREFIX = (
    "Translate the following Arabic text into clear, fluent German. "
    "Output only the German translation. No explanations, no quotes, no preamble.\n\n"
    "Arabic: "
)


def load():
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_GEMINI_KEY_HERE":
        print("[gemini] GEMINI_API_KEY missing — fallback only", flush=True)
        return
    try:
        _client = genai.Client(api_key=api_key)
        _ready.set()
        print(f"[gemini] ready (model={MODEL_ID})", flush=True)
    except Exception as exc:
        print(f"[gemini] init failed: {exc}", flush=True)


def is_ready() -> bool:
    return _ready.is_set()


def translate(text: str) -> str:
    if not _ready.is_set():
        raise RuntimeError("gemini not ready")
    response = _client.models.generate_content(
        model=MODEL_ID,
        contents=PROMPT_PREFIX + text,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512,
        ),
    )
    out = (response.text or "").strip()
    if not out:
        raise RuntimeError("gemini returned empty")
    return out
