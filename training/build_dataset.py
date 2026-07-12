"""Build the ar->de fine-tuning dataset for TranslateGemma.

Sources:
  * Quran: 4 German editions (Bubenheim, Khoury, Zaidan, Abu Rida) vs the
    simple Arabic text — fawazahmed0/quran-api CDN (~25k pairs)
  * Liturgical formulas + hadith canon from data/*.json
  * A small curated terminology set (Islamic German renderings)

Eval integrity: every Arabic string used by tests/eval_translation.py
(QURAN_REFS verses, KHUTBAH_CASES, IMAM_MT_CASES) is excluded.

Output: training/dataset.jsonl  — {"ar": ..., "de": ...} per line, shuffled.
Run:    python training/build_dataset.py
"""
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from backend.quran_match import normalize  # noqa: E402

CDN = "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions"
GERMAN_EDITIONS = [
    "deu-frankbubenheima",
    "deu-adeltheodorkhou",
    "deu-amirzaidan",
    "deu-aburidamuhammad",
]

TERMINOLOGY_PAIRS = [
    ("اتقوا الله في السر والعلن", "Fürchtet Allah im Verborgenen und im Offenen."),
    ("التقوى وصية الله للأولين والآخرين", "Die Gottesfurcht ist Allahs Gebot an die Früheren und die Späteren."),
    ("أطيعوا الله وأطيعوا الرسول", "Gehorcht Allah und gehorcht dem Gesandten."),
    ("الدنيا مزرعة الآخرة", "Das Diesseits ist das Saatfeld des Jenseits."),
    ("اللهم اجعلنا من عبادك الصالحين", "O Allah, mach uns zu Deinen rechtschaffenen Dienern."),
    ("إن الله يحب المحسنين", "Allah liebt die Gutes Tuenden."),
    ("الجنة تحت أقدام الأمهات", "Das Paradies liegt zu Füßen der Mütter."),
    ("خير الناس أنفعهم للناس", "Der beste Mensch ist der, der den Menschen am meisten nützt."),
    ("الصبر مفتاح الفرج", "Geduld ist der Schlüssel zur Erleichterung."),
    ("الدعاء هو العبادة", "Das Bittgebet ist die Anbetung."),
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def eval_exclusions():
    """Normalized Arabic of everything the eval harness scores."""
    import eval_translation as ev
    excl = set()
    ar = json.loads((ROOT / "data" / "quran_ar_simple.json").read_text(encoding="utf-8"))["quran"]
    by_ref = {f"{v['chapter']}:{v['verse']}": v["text"] for v in ar}
    for ref in ev.QURAN_REFS:
        excl.add(normalize(by_ref[ref]))
    for case in list(ev.KHUTBAH_CASES) + list(ev.IMAM_MT_CASES):
        excl.add(normalize(case[0]))
    return excl


def main():
    excl = eval_exclusions()
    pairs = []

    arabic = fetch(f"{CDN}/ara-quransimple.min.json")["quran"]
    for ed in GERMAN_EDITIONS:
        german = fetch(f"{CDN}/{ed}.min.json")["quran"]
        kept = 0
        for a, g in zip(arabic, german):
            ar_t, de_t = a["text"].strip(), g["text"].strip()
            if not ar_t or not de_t or normalize(ar_t) in excl:
                continue
            # skip ultra-short verses (single word) — no signal, high ambiguity
            if len(normalize(ar_t).split()) < 3:
                continue
            pairs.append({"ar": ar_t, "de": de_t})
            kept += 1
        print(f"{ed}: {kept} pairs")

    for fname in ("formulas.json", "hadith.json"):
        entries = json.loads((ROOT / "data" / fname).read_text(encoding="utf-8"))["formulas"]
        for e in entries:
            if normalize(e["ar"]) not in excl:
                pairs.append({"ar": e["ar"], "de": e["de"]})
        print(f"{fname}: +{len(entries)}")

    for ar_t, de_t in TERMINOLOGY_PAIRS:
        if normalize(ar_t) not in excl:
            pairs.append({"ar": ar_t, "de": de_t})

    random.seed(42)
    random.shuffle(pairs)
    out = ROOT / "training" / "dataset.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"TOTAL: {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
