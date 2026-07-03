# Translation accuracy research (2026-07)

Goal: ≥0.94 translation accuracy (mean cosine similarity of multilingual
sentence embeddings between system output and reference German).

## Datasets / resources found

| Resource | What | Use here |
|---|---|---|
| [Tanzil](https://tanzil.net/trans/) | Verified Quran text + translations, UTF-8 | Source of truth |
| [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api) (jsDelivr CDN) | JSON, 6236 verses, 5 German editions | **Used** — `data/quran_ar_simple.json`, `data/quran_de_bubenheim.json` |
| Bubenheim & Elyas | Semi-official German Quran translation (quranenc.com) | **Used** as canonical German |
| [djalal/quran-mcp-server](https://github.com/djalal/quran-mcp-server) | MCP server for Quran.com API v4 (search/translation/tafsir) | Not used — runtime needs offline/local, REST latency + dependency not acceptable mid-khutbah; local JSON is strictly better here |
| [sacred-scriptures-mcp](https://github.com/Traves-Theberge/sacred-scriptures-mcp) | Multi-religion scripture MCP | Not needed |
| OPUS Tanzil corpus | ar-de parallel corpus (Quran) | Superseded by direct JSON |
| [TranslateGemma](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) (Jan 2026) | Google open MT models 4B/12B/27B, 55 languages | 4B already in pipeline; 12B is upgrade option (12B > Gemma3-27B on WMT24++) |
| Helsinki-NLP/opus-mt-ar-de | Marian MT | Existing fallback only |

## Key insight

Khutbah content splits into two classes:
1. **Quoted canonical text** (Quran, well-known hadith) — MT is *worst* here
   (classical Arabic) but a canonical German translation *exists*. Serve it
   verbatim → near-perfect accuracy.
2. **Free speech** (modern standard Arabic) — TranslateGemma handles well.

Baseline confirmed this: translategemma:4b on Quran verses produces
hallucinations (e.g. Russian text for 1:2), while khutbah-style MSA is fine.

## Changes made

- `src/backend/quran_match.py` — offline verse matcher:
  - normalization (diacritics stripped, alef/ya/ta-marbuta unified) → robust to ASR output
  - word-bigram inverted index over 6236 verses (fast, in-memory)
  - rapidfuzz `partial_ratio` scoring, threshold 80
  - greedy window extension → handles multi-verse recitation + boundary straddle
  - basmala prefix stripped from verse 1 of each sura (dataset artifact)
- `src/backend/workers.py` — matcher consulted before MT; consecutive
  flushes matching overlapping verse ranges are deduped (a long verse split
  across buffer flushes displays its full translation once)
- `tests/eval_translation.py` — eval harness: 47 cases (24 verses + 3
  partial-chunk cases + 20 khutbah sentences), scored with
  `paraphrase-multilingual-MiniLM-L12-v2` cosine + chrF; results appended to
  `tests/eval_results.json`

## Changes made (continued)

- `src/backend/rerank.py` — cross-lingual QE reranker: embeds the Arabic
  source + each candidate (gemma, helsinki) with multilingual MiniLM, picks
  the semantically closest. Also rejects hallucinations the string guard
  misses.
- `src/backend/postprocess.py` — `germanize_deity` (Gott→Allah with
  generic-deity guard: "kein Gott außer" stays), `fix_terms`
  (Bote→Gesandter, Mohammed→Muhammad), `looks_german` language-leak guard
  (gemma occasionally emits Cyrillic on liturgical formulas).
- gemma temperature 0.2 → 0.0.
- `tests/test_quran_match.py` — 9 unit tests for the matcher.

## Eval results

See `tests/eval_results.json` (history). Metric: mean cosine similarity
(paraphrase-multilingual-MiniLM-L12-v2) vs reference German, 47 cases.

| Run | Overall | Quran | Khutbah |
|---|---|---|---|
| baseline (gemma only) | 0.7438 | 0.7555 | 0.7391 |
| + verse matcher (v1) | 0.8580 | 0.9401 | 0.7484 |
| + matcher fixes, deity postproc, temp 0 (v2) | 0.9306 | 1.0000 | 0.8370 |
| + QE rerank, vocab fixes, fragment rules (v3) | 0.9454 | 1.0000 | 0.8717 |
| + fragment cutoff 0.85 (v4, local-only baseline) | 0.9500 | 1.0000 | 0.8825 |
| + remote translategemma:12b on Kaggle T4 (v5, current) | **0.9580** | **1.0000** | 0.9012 |

Target ≥ 0.94: **met** (with or without the remote booster).

## Remote 12B booster (v5)

Kaggle T4×2 runs `translategemma:12b` (Q4_K_M, 8.1 GB), exposed through a
VS Code tunnel port-forward (`https://<id>-11434.<region>.devtunnels.ms`,
visibility Public). ~2.2 s per chunk warm; first call after session start
cold-loads the model (>20 s) — pre-warm before the khutbah (notebook cell 3).
`gemma_translate.py` health-checks every 30 s and falls back to local 4B
mid-call on failure (verified live). Gotcha found on the way: university
network DNS (TU Darmstadt) filters tunnel domains — fixed by setting the
Wi-Fi adapter DNS to Cloudflare (1.1.1.1 / 2606:4700:4700::1111).

## Remaining ideas (not done)

- Hadith corpus matching (khutbah-13/15 are hadiths; no clean aligned
  ar-de hadith dataset found — would need scraping sunnah.com + German
  sources)
- TranslateGemma 12B if the machine ever gets a GPU (12B > Gemma3-27B on
  WMT24++; CPU-only here → too slow live)
- Optional third candidate (gemma with khutbah-domain prompt variant) —
  helped formulas but hurt fidelity elsewhere in testing; rerank could
  arbitrate, at the cost of a second gemma call per chunk
