"""Liturgical formula matcher.

Khutbahs are framed by fixed liturgical texts (khutbat al-hajah, shahada,
salawat calls, standard du'a, the closing istighfar). Like Quran verses,
these have established German renderings — serving them canonically beats
machine translation. Checked after the Quran matcher, before MT.

Data: data/formulas.json ({"formulas": [{"id", "ar", "de"}, ...]}).
"""
import json
import threading
from dataclasses import dataclass
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

from backend.quran_match import normalize

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_FILES = [_DATA_DIR / "formulas.json", _DATA_DIR / "hadith.json"]

MIN_WORDS = 2
SCORE_THRESHOLD = 88.0

_formulas = []   # (id, norm_ar, german)
_exact = {}
_ready = threading.Event()


def load():
    global _formulas, _exact
    if fuzz is None:
        print("[formula] rapidfuzz missing — formula matching disabled", flush=True)
        return
    entries = []
    for f in DATA_FILES:
        try:
            entries.extend(json.loads(f.read_text(encoding="utf-8"))["formulas"])
        except Exception as exc:
            print(f"[formula] {f.name} not loaded: {exc}", flush=True)
    if not entries:
        print("[formula] no data — disabled", flush=True)
        return
    formulas = []
    exact = {}
    for e in entries:
        norm = normalize(e["ar"])
        formulas.append((e["id"], norm, e["de"]))
        exact.setdefault(norm, len(formulas) - 1)
    _formulas = formulas
    _exact = exact
    _ready.set()
    print(f"[formula] ready ({len(formulas)} formulas)", flush=True)


def is_ready() -> bool:
    return _ready.is_set()


@dataclass
class Match:
    ref: str      # "formula:<id>"
    german: str
    score: float


def match(text: str):
    if not _ready.is_set():
        return None
    norm = normalize(text)
    words = norm.split()
    if len(words) < MIN_WORDS:
        return None

    idx = _exact.get(norm)
    if idx is not None:
        fid, _, german = _formulas[idx]
        return Match(ref=f"formula:{fid}", german=german, score=100.0)

    best = None
    for fid, fnorm, german in _formulas:
        if len(norm) > len(fnorm) * 1.35:
            # chunk clearly longer than the formula: only accept when the
            # formula makes up most of the chunk (imam appends a few words)
            score = fuzz.ratio(norm, fnorm)
        else:
            # chunk within / part of the formula (long formulas arrive
            # split across buffer flushes)
            score = fuzz.partial_ratio(norm, fnorm)
        if best is None or score > best[0]:
            best = (score, fid, german)

    score, fid, german = best
    if score < SCORE_THRESHOLD:
        return None
    return Match(ref=f"formula:{fid}", german=german, score=score)
