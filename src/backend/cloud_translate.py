"""Thin client for the cloud translation server (server/app.py).

When TRANSLATE_SERVER_URL is set in .env, TranslateWorker sends each chunk
to the server (which runs matcher + MT + rerank) instead of the local
stack. Health-checked every 30 s; when the server is down the worker falls
back to whatever is available locally.

.env:
    TRANSLATE_SERVER_URL=https://xxxx.trycloudflare.com
"""
import json
import os
import threading
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

SERVER_URL = (os.getenv("TRANSLATE_SERVER_URL") or "").rstrip("/")
HEALTH_INTERVAL = 30.0
REQUEST_TIMEOUT = 20.0

_alive = threading.Event()


def _check() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as resp:
            ok = json.load(resp).get("ok", False)
    except Exception:
        ok = False
    was = _alive.is_set()
    if ok and not was:
        print(f"[cloud] translate server up ({SERVER_URL})", flush=True)
        _alive.set()
    if not ok and was:
        print("[cloud] translate server lost", flush=True)
        _alive.clear()
    return ok


def _monitor():
    while True:
        time.sleep(HEALTH_INTERVAL)
        _check()


def load():
    if not SERVER_URL:
        print("[cloud] TRANSLATE_SERVER_URL not set — cloud mode off", flush=True)
        return
    _check()
    threading.Thread(target=_monitor, daemon=True).start()


def is_ready() -> bool:
    return _alive.is_set()


def translate(text: str, context_ref: str = None) -> dict:
    """Returns {'german', 'ref', 'source', 'qe'}; raises on failure."""
    body = json.dumps({"text": text, "context_ref": context_ref}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/translate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.load(resp)
    except Exception:
        _alive.clear()  # skip the server until the monitor revives it
        raise
