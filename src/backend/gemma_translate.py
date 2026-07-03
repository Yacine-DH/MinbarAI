"""TranslateGemma translator via Ollama HTTP API.

Supports an ordered list of endpoints: an optional remote GPU endpoint
(Colab/Kaggle tunnel running translategemma:12b, see
notebooks/remote_ollama.ipynb) tried first, then local Ollama with the 4B
model. A background health monitor keeps endpoint state fresh so a dead
tunnel never stalls a live chunk.

.env keys:
    REMOTE_OLLAMA_HOST=https://xxxx.trycloudflare.com   (optional)
    REMOTE_MODEL_ID=translategemma:12b                  (optional, default 12b)
"""
import os
import threading
import time

import ollama
from dotenv import load_dotenv

from backend import postprocess

load_dotenv()

LOCAL_HOST = "http://localhost:11434"
LOCAL_MODEL = "translategemma:4b"
REMOTE_HOST = (os.getenv("REMOTE_OLLAMA_HOST") or "").rstrip("/")
REMOTE_MODEL = os.getenv("REMOTE_MODEL_ID", "translategemma:12b")

HEALTH_INTERVAL = 30.0   # seconds between endpoint health checks
REQUEST_TIMEOUT = 20.0   # per-call timeout (remote tunnels can hang)

PROMPT_TEMPLATE = (
    "You are a professional Arabic (ar) to German (de-DE) translator. "
    "Your goal is to accurately convey the meaning and nuances of the "
    "original Arabic text while adhering to German grammar, vocabulary, "
    "and cultural sensitivities. Produce only the German translation, "
    "without any additional explanations or commentary. "
    "Please translate the following Arabic text into German:\n\n\n"
    "{text}"
)


class _Endpoint:
    def __init__(self, name, host, model):
        self.name = name
        self.host = host
        self.model = model
        self.client = None
        self.alive = False

    def check(self):
        """Probe the endpoint; alive only if reachable AND model present."""
        try:
            if self.client is None:
                self.client = ollama.Client(host=self.host, timeout=REQUEST_TIMEOUT)
            resp = self.client.list()
            models = getattr(resp, "models", None) or resp.get("models", [])
            names = []
            for m in models:
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if name is None and isinstance(m, dict):
                    name = m.get("model") or m.get("name") or ""
                names.append(name or "")
            was = self.alive
            self.alive = any(self.model in n for n in names)
            if self.alive and not was:
                print(f"[gemma] endpoint '{self.name}' up ({self.model} @ {self.host})", flush=True)
            if not self.alive and was:
                print(f"[gemma] endpoint '{self.name}' lost", flush=True)
        except Exception:
            if self.alive:
                print(f"[gemma] endpoint '{self.name}' unreachable", flush=True)
            self.alive = False
        return self.alive


_endpoints = []
if REMOTE_HOST:
    _endpoints.append(_Endpoint("remote", REMOTE_HOST, REMOTE_MODEL))
_endpoints.append(_Endpoint("local", LOCAL_HOST, LOCAL_MODEL))

_ready = threading.Event()


def _monitor():
    while True:
        for ep in _endpoints:
            ep.check()
        if any(ep.alive for ep in _endpoints):
            _ready.set()
        else:
            _ready.clear()
        time.sleep(HEALTH_INTERVAL)


def load():
    for ep in _endpoints:
        ep.check()
    if any(ep.alive for ep in _endpoints):
        _ready.set()
        active = next(ep for ep in _endpoints if ep.alive)
        print(f"[gemma] ready (active={active.name}, model={active.model})", flush=True)
    else:
        print(f"[gemma] no endpoint available. Local: ollama pull {LOCAL_MODEL}", flush=True)
    threading.Thread(target=_monitor, daemon=True).start()


def is_ready() -> bool:
    return _ready.is_set()


def active_endpoint() -> str:
    for ep in _endpoints:
        if ep.alive:
            return ep.name
    return "none"


def translate(text: str) -> str:
    if not _ready.is_set():
        raise RuntimeError("gemma not ready")
    prompt = PROMPT_TEMPLATE.format(text=text)
    last_exc = None
    for ep in _endpoints:
        if not ep.alive:
            continue
        try:
            response = ep.client.chat(
                model=ep.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 512},
            )
            msg = getattr(response, "message", None)
            if msg is None and isinstance(response, dict):
                msg = response.get("message", {})
            content = getattr(msg, "content", None) if msg is not None else None
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            out = (content or "").strip()
            if not out:
                raise RuntimeError("empty response")
            return postprocess.clean(out)
        except Exception as exc:
            # mark dead immediately so the next chunk skips the wait;
            # the monitor thread revives it when it comes back
            ep.alive = False
            last_exc = exc
            print(f"[gemma] '{ep.name}' failed mid-call → next endpoint: {exc}", flush=True)
    raise RuntimeError(f"all gemma endpoints failed: {last_exc}")
