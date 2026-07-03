"""Post-processing for MT output.

MT models render الله as "Gott"; congregation display convention (and the
Bubenheim canonical text) uses "Allah". Generic-deity uses ("kein Gott außer
ihm", "ein Gott") must stay "Gott".
"""
import re

_DEITY = re.compile(r"\b(Gottes|Gott)\b")
_GENERIC_PRECEDERS = {
    "kein", "keinen", "keinem", "keines",
    "ein", "einen", "einem", "eines", "einer",
    "anderer", "anderen", "anderem", "falscher", "falschen",
}


_NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿ一-鿿぀-ヿ]")


def looks_german(text: str) -> bool:
    """Reject MT output that leaked another script (observed: Cyrillic)."""
    if not text.strip():
        return False
    return len(_NON_LATIN.findall(text)) <= len(text) * 0.1


def germanize_deity(text: str) -> str:
    def rep(m):
        before = text[:m.start()].split()
        prev = before[-1].strip('"„,.;:!?()').lower() if before else ""
        if prev in _GENERIC_PRECEDERS:
            return m.group(0)
        return "Allahs" if m.group(0) == "Gottes" else "Allah"

    return _DEITY.sub(rep, text)


# established Islamic-German renderings the MT models regularly miss
_TERMS = [
    (re.compile(r"\bBoten\b"), "Gesandten"),
    (re.compile(r"\bBote\b"), "Gesandter"),
    (re.compile(r"\bMohammed\b"), "Muhammad"),
]


def fix_terms(text: str) -> str:
    for pat, rep in _TERMS:
        text = pat.sub(rep, text)
    return text


def clean(text: str) -> str:
    return fix_terms(germanize_deity(text))
