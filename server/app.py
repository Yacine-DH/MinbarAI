"""MinbarAI cloud translation server.

Runs the full translation stack (Quran verse matcher, TranslateGemma via
local Ollama, Helsinki MT, QE rerank, postprocess) behind one HTTP endpoint
so the mosque PC only needs mic + VAD + overlay.

Deploy: notebooks/cloud_server.ipynb (Colab/Kaggle). Locally:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Env:
    OLLAMA_MODEL_ID   model served by the co-located Ollama (default
                      translategemma:4b; the notebook sets translategemma:12b)
"""
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import formula_match, gemma_translate, local_translate, postprocess, quran_match, rerank  # noqa: E402

app = FastAPI(title="MinbarAI translate server")

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _gemini_translate(text: str):
    """Rescue engine for fragments the strict TranslateGemma template
    refuses. Only called after the QE gate fails — a handful of calls per
    khutbah, well inside the free tier."""
    if not GEMINI_KEY:
        return None
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
    prompt = (
        "Translate this Arabic mosque-sermon fragment into German. It may be "
        "rhymed classical prose, a hadith, or an incomplete phrase — translate "
        "faithfully anyway. Use established Islamic German terminology "
        "(Allah, Gesandter, Gottesfurcht). Reply with the German translation "
        "only, no commentary.\n\n" + text
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:
        print(f"[server] gemini rescue failed: {exc}", flush=True)
        return None


@app.on_event("startup")
def _load():
    quran_match.load()
    formula_match.load()
    rerank.load()
    gemma_translate.load()
    threading.Thread(target=local_translate.load, daemon=True).start()


class TranslateRequest(BaseModel):
    text: str
    context_ref: str | None = None  # previous verse ref during recitation


class TranslateResponse(BaseModel):
    german: str
    ref: str = ""       # sura:verse when the text is a Quran quotation
    source: str = ""    # "quran" | "gemma" | "helsinki"
    qe: float = 0.0     # rerank score when MT was used


@app.get("/health")
def health():
    return {
        "ok": True,
        "quran": quran_match.is_ready(),
        "gemma": gemma_translate.is_ready(),
        "helsinki": local_translate.is_ready(),
        "rerank": rerank.is_ready(),
    }


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        return TranslateResponse(german="")

    if quran_match.is_ready():
        m = quran_match.match(text, context_ref=req.context_ref)
        if m:
            return TranslateResponse(german=m.german, ref=m.ref, source="quran", qe=1.0)

    if formula_match.is_ready():
        m = formula_match.match(text)
        if m:
            return TranslateResponse(german=m.german, ref=m.ref, source="formula", qe=1.0)

    # Gemma 12B is the primary engine. As a competing candidate Helsinki
    # wins the rerank too often on hard rhetoric with worse output (e.g.
    # جُنة shield → "Paradies"), so it is only consulted when gemma fails,
    # refuses, leaks another script, or semantically disagrees with the
    # source (QE gate).
    QE_GATE = 0.40

    def _qe(out):
        return rerank.score(text, out) if rerank.is_ready() else 1.0

    gemma_out, gemma_qe = None, -1.0
    if gemma_translate.is_ready():
        try:
            out = gemma_translate.translate(text)
            if postprocess.looks_german(out) and not postprocess.is_refusal(out):
                gemma_out, gemma_qe = out, _qe(out)
        except Exception as exc:
            print(f"[server] gemma failed: {exc}", flush=True)

    if gemma_out is not None and gemma_qe >= QE_GATE:
        return TranslateResponse(german=gemma_out, source="gemma", qe=round(gemma_qe, 3))

    # rescue ladder for the ~5% the strict template refuses or bungles
    # (rhymed saj', hadith fragments). Gemini outranks Helsinki by fiat —
    # the QE score is too noisy on short poetic fragments to arbitrate.
    g = _gemini_translate(text)
    if g and postprocess.looks_german(g) and not postprocess.is_refusal(g):
        g = postprocess.clean(g)
        g_qe = _qe(g)
        if g_qe >= max(gemma_qe, 0.30):
            return TranslateResponse(german=g, source="gemini", qe=round(g_qe, 3))

    if gemma_out is not None and gemma_qe >= 0.30:
        return TranslateResponse(german=gemma_out, source="gemma", qe=round(gemma_qe, 3))

    if local_translate.is_ready():  # never block on a still-loading model
        try:
            out = local_translate.translate(text)
            return TranslateResponse(german=out, source="helsinki", qe=round(_qe(out), 3))
        except Exception as exc:
            print(f"[server] helsinki failed: {exc}", flush=True)
    if gemma_out is not None:
        return TranslateResponse(german=gemma_out, source="gemma", qe=round(gemma_qe, 3))
    return TranslateResponse(german="")
