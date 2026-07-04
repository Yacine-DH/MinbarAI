"""Translation accuracy evaluation for the MinbarAI pipeline.

Builds a test set of Arabic->German pairs:
  * Quran verses (full + partial 8-word chunks, as the buffer produces them)
    with Bubenheim & Elyas as reference
  * Khutbah-style sentences (sermon formulas, du'a, admonitions) with
    curated reference translations

Scores each system output against the reference with:
  * cosine similarity of multilingual sentence embeddings (primary "accuracy")
  * chrF (secondary, surface-level)

Usage:
    python tests/eval_translation.py                # full pipeline (matcher + gemma)
    python tests/eval_translation.py --no-matcher   # gemma only (baseline)
    python tests/eval_translation.py --helsinki     # helsinki fallback only
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend import quran_match  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_FILE = ROOT / "tests" / "eval_results.json"

# well-known verses an imam actually recites: Fatiha, Ayat al-Kursi, Ikhlas,
# Asr, khutbah-closing (16:90), taqwa verses (3:102, 4:1, 33:70-71), 33:56
QURAN_REFS = [
    "1:1", "1:2", "1:5", "1:6", "1:7",
    "2:255",
    "112:1", "112:2", "112:3", "112:4",
    "103:1", "103:2", "103:3",
    "16:90", "3:102", "4:1", "33:70", "33:71", "33:56",
    "59:18", "2:183", "17:23", "31:13", "49:13",
]

# khutbah-style sentences with curated references
KHUTBAH_CASES = [
    ("الحمد لله رب العالمين نحمده ونستعينه ونستغفره",
     "Alles Lob gebührt Allah, dem Herrn der Welten. Wir loben Ihn, suchen Seine Hilfe und bitten Ihn um Vergebung."),
    ("أشهد أن لا إله إلا الله وحده لا شريك له",
     "Ich bezeuge, dass es keinen Gott außer Allah gibt, Er allein, ohne Teilhaber."),
    ("وأشهد أن محمدا عبده ورسوله",
     "Und ich bezeuge, dass Muhammad Sein Diener und Sein Gesandter ist."),
    ("أيها المسلمون اتقوا الله حق تقاته",
     "O ihr Muslime, fürchtet Allah, wie es Ihm gebührt."),
    ("أوصيكم ونفسي بتقوى الله عز وجل",
     "Ich empfehle euch und mir selbst die Furcht vor Allah, dem Mächtigen und Erhabenen."),
    ("إن الصلاة عماد الدين ومن أقامها فقد أقام الدين",
     "Das Gebet ist die Stütze der Religion. Wer es verrichtet, hat die Religion aufgerichtet."),
    ("اعلموا أن الله أمركم بأمر عظيم",
     "Wisst, dass Allah euch etwas Gewaltiges befohlen hat."),
    ("فيا عباد الله عليكم بالصدق فإنه يهدي إلى البر",
     "O Diener Allahs, haltet an der Wahrhaftigkeit fest, denn sie führt zur Frömmigkeit."),
    ("واعلموا أن الدنيا دار ممر والآخرة دار مقر",
     "Und wisst, dass das Diesseits ein Ort des Durchgangs ist und das Jenseits ein Ort des Verweilens."),
    ("اللهم اغفر للمسلمين والمسلمات والمؤمنين والمؤمنات",
     "O Allah, vergib den muslimischen Männern und Frauen und den gläubigen Männern und Frauen."),
    ("اللهم انصر الإسلام والمسلمين في كل مكان",
     "O Allah, verhilf dem Islam und den Muslimen überall zum Sieg."),
    ("ربنا آتنا في الدنيا حسنة وفي الآخرة حسنة وقنا عذاب النار",
     "Unser Herr, gib uns im Diesseits Gutes und im Jenseits Gutes und bewahre uns vor der Strafe des Feuers."),
    ("إن أصدق الحديث كتاب الله وخير الهدي هدي محمد صلى الله عليه وسلم",
     "Die wahrhaftigste Rede ist das Buch Allahs, und die beste Rechtleitung ist die Rechtleitung Muhammads, Allahs Segen und Friede auf ihm."),
    ("أيها الإخوة الكرام إن شهر رمضان شهر مبارك",
     "Werte Brüder, der Monat Ramadan ist ein gesegneter Monat."),
    ("من صام رمضان إيمانا واحتسابا غفر له ما تقدم من ذنبه",
     "Wer den Ramadan aus Glauben und in Erwartung des Lohnes fastet, dem werden seine vergangenen Sünden vergeben."),
    ("اتقوا الله واعلموا أنكم ملاقوه",
     "Fürchtet Allah und wisst, dass ihr Ihm begegnen werdet."),
    ("إن من أعظم النعم نعمة الأمن والإيمان",
     "Zu den größten Gaben gehören die Gabe der Sicherheit und des Glaubens."),
    ("علينا أن نربي أبناءنا على حب الله ورسوله",
     "Wir müssen unsere Kinder zur Liebe zu Allah und Seinem Gesandten erziehen."),
    ("بروا آباءكم وأمهاتكم فإن رضا الله في رضا الوالدين",
     "Seid gütig zu euren Vätern und Müttern, denn das Wohlgefallen Allahs liegt im Wohlgefallen der Eltern."),
    ("أقول قولي هذا وأستغفر الله لي ولكم",
     "Ich sage diese meine Worte und bitte Allah um Vergebung für mich und für euch."),
]


def build_cases():
    ar = json.loads((DATA_DIR / "quran_ar_simple.json").read_text(encoding="utf-8"))["quran"]
    de = json.loads((DATA_DIR / "quran_de_bubenheim.json").read_text(encoding="utf-8"))["quran"]
    by_ref = {f"{a['chapter']}:{a['verse']}": (a["text"], g["text"]) for a, g in zip(ar, de)}

    cases = []
    basmala_norm = quran_match.normalize("بسم الله الرحمن الرحيم")
    basmala_de = "Im Namen Allahs, des Allerbarmers, des Barmherzigen."
    for ref in QURAN_REFS:
        ar_text, de_text = by_ref[ref]
        # dataset embeds the basmala in verse 1 of each sura on the Arabic
        # side only; the correct display includes its translation
        norm = quran_match.normalize(ar_text)
        if ref != "1:1" and norm.startswith(basmala_norm + " "):
            de_text = f"{basmala_de} {de_text}"
        cases.append({"id": f"quran-{ref}", "type": "quran", "ar": ar_text, "ref_de": de_text})

    # partial chunks: 8-word slices as the ASR/buffer actually produces them
    # (plain words, no diacritics or ornaments)
    kursi_ar, kursi_de = by_ref["2:255"]
    words = quran_match.normalize(kursi_ar).split()
    cases.append({"id": "quran-2:255-chunk1", "type": "quran-partial",
                  "ar": " ".join(words[:8]), "ref_de": kursi_de})
    # mid-verse continuation: pipeline supplies the previous match as context
    cases.append({"id": "quran-2:255-chunk2", "type": "quran-partial",
                  "ar": " ".join(words[8:16]), "ref_de": kursi_de,
                  "context_ref": "2:255"})
    w41 = quran_match.normalize(by_ref["4:1"][0]).split()
    # skip the embedded basmala (words 0-3): reciting sura 4 mid-khutbah
    # starts at the verse itself
    cases.append({"id": "quran-4:1-chunk1", "type": "quran-partial",
                  "ar": " ".join(w41[4:12]), "ref_de": by_ref["4:1"][1]})

    for i, (ar_text, de_text) in enumerate(KHUTBAH_CASES, 1):
        cases.append({"id": f"khutbah-{i:02d}", "type": "khutbah", "ar": ar_text, "ref_de": de_text})
    return cases


def translate_gemma(text):
    """Mirrors TranslateWorker: gemma + helsinki candidates, QE rerank."""
    from backend import gemma_translate, postprocess, rerank
    if not gemma_translate.is_ready():
        gemma_translate.load()
    if not rerank.is_ready():
        rerank.load()
    candidates = []
    try:
        out = gemma_translate.translate(text)
        if postprocess.looks_german(out):
            candidates.append(("gemma", out))
    except Exception:
        pass
    candidates.append(("helsinki", translate_helsinki(text)))
    if len(candidates) > 1 and rerank.is_ready():
        (label, out), _ = rerank.pick(text, candidates)
        return out
    return candidates[0][1]


def translate_helsinki(text):
    from backend import local_translate
    if not local_translate._ready.is_set():
        local_translate.load()
    return local_translate.translate(text)


def translate_server(url, text, context_ref=None):
    import json as _json
    import urllib.request
    body = _json.dumps({"text": text, "context_ref": context_ref}).encode()
    req = urllib.request.Request(f"{url}/translate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return _json.load(resp)


def run(cases, use_matcher, engine, server=None):
    if use_matcher and not server:
        quran_match.load()
    outputs = []
    for c in cases:
        t0 = time.time()
        src, out = "mt", None
        if server:
            res = translate_server(server, c["ar"], c.get("context_ref"))
            out = res.get("german", "")
            src = f"server:{res.get('source')}" + (f"({res['ref']})" if res.get("ref") else "")
        elif use_matcher:
            m = quran_match.match(c["ar"], context_ref=c.get("context_ref"))
            if m:
                out, src = m.german, f"quran({m.ref},{m.score:.0f})"
        if out is None:
            out = translate_helsinki(c["ar"]) if engine == "helsinki" else translate_gemma(c["ar"])
        outputs.append({**c, "hyp_de": out, "source": src, "secs": round(time.time() - t0, 2)})
        print(f"  {c['id']:24s} [{src}] {out[:70]!r}", flush=True)
    return outputs


def score(outputs):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sacrebleu.metrics import CHRF

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    hyps = [o["hyp_de"] for o in outputs]
    refs = [o["ref_de"] for o in outputs]
    eh = model.encode(hyps, normalize_embeddings=True)
    er = model.encode(refs, normalize_embeddings=True)
    sims = (eh * er).sum(axis=1)

    chrf = CHRF()
    for o, s in zip(outputs, sims):
        o["cos"] = round(float(s), 4)
        o["chrf"] = round(chrf.sentence_score(o["hyp_de"], [o["ref_de"]]).score, 1)

    by_type = {}
    for o in outputs:
        by_type.setdefault(o["type"], []).append(o["cos"])
    print("\n=== Results ===")
    for t, vals in sorted(by_type.items()):
        print(f"  {t:14s} n={len(vals):2d}  mean cos={np.mean(vals):.4f}  min={min(vals):.4f}")
    overall = float(np.mean(sims))
    print(f"  {'OVERALL':14s} n={len(sims):2d}  mean cos={overall:.4f}")
    print(f"  mean chrF: {np.mean([o['chrf'] for o in outputs]):.1f}")
    worst = sorted(outputs, key=lambda o: o["cos"])[:5]
    print("\n  worst cases:")
    for o in worst:
        print(f"    {o['id']:24s} cos={o['cos']:.3f} [{o['source']}]")
        print(f"      hyp: {o['hyp_de'][:90]}")
        print(f"      ref: {o['ref_de'][:90]}")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-matcher", action="store_true")
    ap.add_argument("--helsinki", action="store_true")
    ap.add_argument("--server", default=None, help="translate via server URL (cloud pipeline)")
    ap.add_argument("--tag", default=None, help="label stored in results file")
    args = ap.parse_args()

    cases = build_cases()
    engine = "helsinki" if args.helsinki else "gemma"
    use_matcher = not args.no_matcher
    tag = args.tag or ("server" if args.server else f"{engine}{'+matcher' if use_matcher else ''}")
    print(f"Running eval: {tag} ({len(cases)} cases)")
    outputs = run(cases, use_matcher, engine, server=args.server)
    overall = score(outputs)

    hist = []
    if RESULTS_FILE.exists():
        hist = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    hist.append({"tag": tag, "overall_cos": round(overall, 4),
                 "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "outputs": outputs})
    RESULTS_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
