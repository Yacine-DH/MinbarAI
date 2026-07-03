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
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import gemma_translate, local_translate, postprocess, quran_match, rerank  # noqa: E402

app = FastAPI(title="MinbarAI translate server")


@app.on_event("startup")
def _load():
    quran_match.load()
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

    candidates = []
    if gemma_translate.is_ready():
        try:
            out = gemma_translate.translate(text)
            if postprocess.looks_german(out):
                candidates.append(("gemma", out))
        except Exception as exc:
            print(f"[server] gemma failed: {exc}", flush=True)
    if local_translate.is_ready() or not candidates:
        try:
            candidates.append(("helsinki", local_translate.translate(text)))
        except Exception as exc:
            print(f"[server] helsinki failed: {exc}", flush=True)

    if not candidates:
        return TranslateResponse(german="")

    if len(candidates) > 1 and rerank.is_ready():
        (label, german), qe = rerank.pick(text, candidates)
        return TranslateResponse(german=german, source=label, qe=round(qe, 3))
    label, german = candidates[0]
    return TranslateResponse(german=german, source=label)
