"""Cross-lingual quality estimation for candidate translations.

Embeds the Arabic source and each German candidate with a multilingual
sentence encoder and scores semantic agreement (cosine). Used to pick the
better of Gemma/Helsinki per chunk and to reject degenerate output
(hallucinations, language leaks) that string checks miss.
"""
import threading

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_lock = threading.Lock()
_ready = threading.Event()


def load():
    global _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
        _model.encode(["warmup"])
        _ready.set()
        print("[rerank] ready", flush=True)
    except Exception as exc:
        print(f"[rerank] init failed — reranking disabled: {exc}", flush=True)


def is_ready() -> bool:
    return _ready.is_set()


def score(source_ar: str, candidate_de: str) -> float:
    """Cross-lingual semantic agreement in [-1, 1]."""
    with _lock:
        emb = _model.encode([source_ar, candidate_de], normalize_embeddings=True)
    return float((emb[0] * emb[1]).sum())


def pick(source_ar: str, candidates: list) -> tuple:
    """Return (best_candidate, its_score). Candidates: list of (label, text)."""
    with _lock:
        texts = [source_ar] + [c[1] for c in candidates]
        emb = _model.encode(texts, normalize_embeddings=True)
    src = emb[0]
    best_i, best_s = 0, -1.0
    for i in range(len(candidates)):
        s = float((src * emb[1 + i]).sum())
        if s > best_s:
            best_i, best_s = i, s
    return candidates[best_i], best_s
