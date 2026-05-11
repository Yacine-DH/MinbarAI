"""TranslateGemma 4B local translator via Ollama HTTP API."""
import threading

import ollama

MODEL_ID = "translategemma:4b"
OLLAMA_HOST = "http://localhost:11434"

PROMPT_TEMPLATE = (
    "You are a professional Arabic (ar) to German (de-DE) translator. "
    "Your goal is to accurately convey the meaning and nuances of the "
    "original Arabic text while adhering to German grammar, vocabulary, "
    "and cultural sensitivities. Produce only the German translation, "
    "without any additional explanations or commentary. "
    "Please translate the following Arabic text into German:\n\n\n"
    "{text}"
)

_client = None
_ready = threading.Event()


def load():
    global _client
    try:
        _client = ollama.Client(host=OLLAMA_HOST)
        models = _client.list()
        names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
        if not any(MODEL_ID in n for n in names):
            print(f"[gemma] model '{MODEL_ID}' not found in Ollama. Run: ollama pull {MODEL_ID}", flush=True)
            return
        _ready.set()
        print(f"[gemma] ready (model={MODEL_ID})", flush=True)
    except Exception as exc:
        print(f"[gemma] init failed: {exc}", flush=True)


def is_ready() -> bool:
    return _ready.is_set()


def translate(text: str) -> str:
    if not _ready.is_set():
        raise RuntimeError("gemma not ready")
    prompt = PROMPT_TEMPLATE.format(text=text)
    response = _client.chat(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_predict": 512},
    )
    out = (response.get("message", {}).get("content") or "").strip()
    if not out:
        raise RuntimeError("gemma returned empty")
    return out
