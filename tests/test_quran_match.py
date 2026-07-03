"""Unit tests for the Quran verse matcher.

Run: python -m pytest tests/test_quran_match.py -q
(needs data/quran_*.json downloaded)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend import quran_match as q  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def loaded():
    q.load()
    assert q.is_ready()


def _verse(ref):
    ar = json.loads((ROOT / "data" / "quran_ar_simple.json").read_text(encoding="utf-8"))["quran"]
    ch, v = ref.split(":")
    return next(x["text"] for x in ar if x["chapter"] == int(ch) and x["verse"] == int(v))


def test_full_verse():
    assert q.match(_verse("2:255")).ref == "2:255"


def test_verse_start_fragment():
    words = q.normalize(_verse("2:255")).split()
    m = q.match(" ".join(words[:8]))
    # opening of 2:255 is word-identical to 3:2 — either is correct
    assert m and m.ref in ("2:255", "3:2")


def test_mid_verse_fragment_needs_context():
    words = q.normalize(_verse("2:255")).split()
    frag = " ".join(words[8:16])
    assert q.match(frag) is None
    assert q.match(frag, context_ref="2:255").ref == "2:255"


def test_multi_verse_recitation_with_asr_noise():
    m = q.match("والعصر ان الانسان لفي خسر الا الذين امنو وعملو الصالحات")
    assert m.ref == "103:1-3"


def test_basmala_prefix_resolves_to_recited_verse():
    m = q.match("بسم الله الرحمن الرحيم قل هو الله أحد")
    assert m.ref == "112:1"
    assert m.german.startswith("Im Namen Allahs")


def test_basmala_alone_is_fatiha_1_1():
    assert q.match("بسم الله الرحمن الرحيم").ref == "1:1"


def test_short_verse_exact():
    assert q.match("الله الصمد").ref == "112:2"


def test_sermon_speech_not_matched():
    assert q.match("ايها المسلمون اتقوا الله حق تقاته") is None
    assert q.match("ان شهر رمضان شهر مبارك وخير الشهور") is None
    # opens like 1:2 but continues as sermon formula
    assert q.match("الحمد لله رب العالمين نحمده ونستعينه ونستغفره") is None
    # du'a wording shared with 33:35 must not surface that verse
    m = q.match("اللهم اغفر للمسلمين والمسلمات والمؤمنين والمؤمنات")
    assert m is None
    # tail fragment of 2:223 quoted in speech: reject (full-verse German would
    # mislead) — it is only shown during sequential recitation
    assert q.match("اتقوا الله واعلموا أنكم ملاقوه") is None


def test_refs_overlap():
    assert q.refs_overlap("2:255", "2:255")
    assert q.refs_overlap("2:255-256", "2:256")
    assert not q.refs_overlap("2:255", "2:256")
    assert not q.refs_overlap("2:255", "3:255")
