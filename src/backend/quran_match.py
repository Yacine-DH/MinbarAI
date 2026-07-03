"""Quran verse matcher.

Detects when a transcribed Arabic chunk is a quotation from the Quran and
returns the canonical German translation (Bubenheim & Elyas) instead of
running it through MT. Recited verses are the highest-accuracy content in a
khutbah when served from the canonical text, and the lowest-accuracy when
machine-translated (classical Arabic).

Data files (gitignored, downloaded once):
    data/quran_ar_simple.json   -- Tanzil "simple" Arabic text (6236 verses)
    data/quran_de_bubenheim.json -- Bubenheim & Elyas German (6236 verses)
"""
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:  # matcher disabled if rapidfuzz missing
    fuzz = None

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AR_FILE = DATA_DIR / "quran_ar_simple.json"
DE_FILE = DATA_DIR / "quran_de_bubenheim.json"

# minimum normalized words in a chunk before we attempt a match
MIN_WORDS = 3
# fuzzy score (0-100) required to accept a verse match
SCORE_THRESHOLD = 80.0
# how many bigram-index candidates to score exactly
MAX_CANDIDATES = 12

_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")
_NON_ARABIC = re.compile(r"[^ء-ي\s]")

_verses = []          # list of (ref, norm_text, german)
_bigram_index = {}    # bigram -> set of verse indices
_exact = {}           # norm_text -> verse index (catches short verses)
_basmala_norm = ""
_basmala_de = "Im Namen Allahs, des Allerbarmers, des Barmherzigen."
_ready = threading.Event()


def normalize(text: str) -> str:
    """Strip diacritics/tatweel, unify letter variants, keep Arabic letters only."""
    text = _DIACRITICS.sub("", text)
    text = _NON_ARABIC.sub(" ", text)
    text = re.sub("[أإآٱ]", "ا", text)  # alef variants
    text = text.replace("ى", "ي")  # alef maqsura -> ya
    text = text.replace("ة", "ه")  # ta marbuta -> ha
    text = text.replace("ؤ", "و")  # waw hamza -> waw
    text = text.replace("ئ", "ي")  # ya hamza -> ya
    return " ".join(text.split())


def _bigrams(words):
    return [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]


def load():
    """Parse data files and build the bigram index. Safe to call from a loader thread."""
    global _verses, _bigram_index, _exact, _basmala_norm
    if fuzz is None:
        print("[quran] rapidfuzz not installed — verse matching disabled", flush=True)
        return
    try:
        ar = json.loads(AR_FILE.read_text(encoding="utf-8"))["quran"]
        de = json.loads(DE_FILE.read_text(encoding="utf-8"))["quran"]
    except Exception as exc:
        print(f"[quran] data files missing/corrupt — verse matching disabled: {exc}", flush=True)
        return

    basmala = normalize("بسم الله الرحمن الرحيم")
    verses = []
    index = {}
    exact = {}
    for a, g in zip(ar, de):
        ref = f"{a['chapter']}:{a['verse']}"
        norm = normalize(a["text"])
        # dataset prefixes verse 1 of each sura with the basmala; strip it so
        # recitations without it still match (1:1 is the basmala itself)
        if a["verse"] == 1 and a["chapter"] != 1 and norm.startswith(basmala + " "):
            norm = norm[len(basmala) + 1:]
        idx = len(verses)
        verses.append((ref, norm, g["text"]))
        exact.setdefault(norm, idx)
        for bg in _bigrams(norm.split()):
            index.setdefault(bg, set()).add(idx)

    _verses = verses
    _bigram_index = index
    _exact = exact
    _basmala_norm = basmala
    _ready.set()
    print(f"[quran] ready ({len(verses)} verses indexed)", flush=True)


def is_ready() -> bool:
    return _ready.is_set()


def _ref_span(ref: str):
    """'2:255' -> (2, 255, 255); '2:255-256' -> (2, 255, 256)."""
    chapter, verses = ref.split(":", 1)
    if "-" in verses:
        lo, hi = verses.split("-", 1)
        # cross-chapter spans ('113:5-114:1') — treat as first chapter only
        if ":" in hi:
            hi = lo
    else:
        lo = hi = verses
    return int(chapter), int(lo), int(hi)


def _follows(prev_ref: str, ref: str) -> bool:
    """True when `ref` is the verse immediately after the end of `prev_ref`."""
    try:
        ca, _, ha = _ref_span(prev_ref)
        cb, lb, _ = _ref_span(ref)
    except (ValueError, AttributeError):
        return False
    return ca == cb and lb == ha + 1


def refs_overlap(a: str, b: str) -> bool:
    """True when two matches refer to overlapping verse ranges (continued recitation)."""
    try:
        ca, la, ha = _ref_span(a)
        cb, lb, hb = _ref_span(b)
    except (ValueError, AttributeError):
        return False
    return ca == cb and la <= hb and lb <= ha


@dataclass
class Match:
    ref: str        # e.g. "2:255", or "2:255-256" when spanning two verses
    german: str
    score: float


def match(text: str, context_ref: str = None):
    """Return a Match if `text` looks like a Quran quotation, else None.

    `context_ref` is the ref of the previous match when a recitation is in
    progress; it relaxes the fragment rules so mid-verse continuation chunks
    still resolve to the same verse.
    """
    if not _ready.is_set():
        return None
    norm = normalize(text)
    if not norm:
        return None

    # recitation often opens with the basmala; match the rest independently
    # and re-attach the basmala translation
    prefix_de = ""
    if norm == _basmala_norm:
        idx = _exact[_basmala_norm]  # 1:1
        return Match(ref=_verses[idx][0], german=_verses[idx][2], score=100.0)
    if norm.startswith(_basmala_norm + " "):
        norm = norm[len(_basmala_norm) + 1:]
        prefix_de = _basmala_de + " "

    # exact text of a verse (catches verses too short for fuzzy matching)
    idx = _exact.get(norm)
    if idx is not None:
        ref, _, german = _verses[idx]
        return Match(ref=ref, german=prefix_de + german, score=100.0)

    words = norm.split()
    if len(words) < MIN_WORDS:
        return None

    counts = Counter()
    for bg in _bigrams(words):
        for idx in _bigram_index.get(bg, ()):
            counts[idx] += 1
    if not counts:
        return None

    best = None
    for idx, _ in counts.most_common(MAX_CANDIDATES):
        ref, vnorm, german = _verses[idx]
        # partial_ratio handles both directions: chunk inside a long verse,
        # or a short verse inside a chunk with surrounding speech
        score = fuzz.partial_ratio(norm, vnorm)
        cand = (score, ref, german, idx)
        if best is None or score > best[0]:
            best = cand

    if best is None:
        return None
    score, ref, german, idx = best
    if score < SCORE_THRESHOLD:
        return None

    # a chunk may cover several consecutive verses (short suras) or straddle
    # a verse boundary — greedily extend the window while full-ratio improves
    lo = hi = idx
    best_full = fuzz.ratio(norm, _verses[idx][1])
    improved = True
    while improved:
        improved = False
        for nlo, nhi in ((lo - 1, hi), (lo, hi + 1)):
            if nlo < 0 or nhi >= len(_verses):
                continue
            joined = " ".join(v[1] for v in _verses[nlo:nhi + 1])
            s = fuzz.ratio(norm, joined)
            if s > best_full + 1:
                best_full, lo, hi = s, nlo, nhi
                improved = True
                break

    # acceptance: either the chunk sits inside the matched window (partial
    # recitation of a long verse), or the window explains most of the chunk.
    # rejects sermon speech that merely opens with a verse-like phrase.
    joined_norm = " ".join(v[1] for v in _verses[lo:hi + 1])
    contained = len(norm) <= len(joined_norm) * 1.2
    if not contained and best_full < 70:
        return None
    if contained and len(norm) < len(joined_norm) * 0.85:
        # true fragment of a longer passage. accept only a clean quotation
        # (high fuzzy score) that either starts at the verse head (recitation
        # start), covers nearly the whole verse, or continues an ongoing
        # recitation — otherwise sermon speech reusing Quranic wording would
        # surface an unrelated full-verse translation.
        window_ref = _verses[lo][0] if hi == lo else f"{_verses[lo][0]}-{_verses[hi][0].split(':')[1]}"
        continuing = context_ref and (
            refs_overlap(window_ref, context_ref) or _follows(context_ref, _verses[lo][0])
        )
        if not continuing:
            aln = fuzz.partial_ratio_alignment(norm, joined_norm)
            if aln is None or aln.score < 93:
                return None
            coverage = (aln.dest_end - aln.dest_start) / max(len(joined_norm), 1)
            if aln.dest_start > 12 and coverage < 0.85:
                return None

    if hi > lo:
        first_ref, last_ref = _verses[lo][0], _verses[hi][0]
        first_ch = first_ref.split(":")[0]
        last_ch, last_v = last_ref.split(":")
        ref = f"{first_ref}-{last_v}" if first_ch == last_ch else f"{first_ref}-{last_ref}"
        german = " ".join(v[2] for v in _verses[lo:hi + 1])
        score = max(score, best_full)

    return Match(ref=ref, german=prefix_de + german, score=score)
