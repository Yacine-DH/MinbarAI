# MinbarAI — Cloud mission brief (for Claude running on Kaggle/Colab)

You are Claude Code running inside a Kaggle (or Colab) GPU session. This file
is your complete context. The owner connects his accounts, drops this file in
the working folder, and gives you a goal command — everything you need to
understand the project is here. Work autonomously; verify with the eval
harness after every meaningful change.

## What MinbarAI is

Windows desktop app that listens to an Imam's Arabic khutbah (Friday sermon)
via microphone and displays a live German translation as an overlay for the
congregation. Repo: `https://github.com/Yacine-DH/MinbarAI` (branch:
`cloud-pipeline`).

## Pipeline (already built, working, accuracy 0.958)

```
Mic → Silero VAD → ElevenLabs Scribe v2 (cloud ASR) → text buffer
  → Quran verse matcher (src/backend/quran_match.py)
      · fuzzy match against all 6236 verses (data/quran_ar_simple.json)
      · hit → canonical Bubenheim & Elyas German (data/quran_de_bubenheim.json), NO MT
  → miss → TranslateGemma (12B remote / 4B local, Ollama) + Helsinki opus-mt-ar-de
      → QE rerank (paraphrase-multilingual-MiniLM-L12-v2 cross-lingual cosine)
  → postprocess (Gott→Allah, Bote→Gesandter, language-leak guard)
  → PyQt6 overlay
```

Two server deployment modes already exist on this branch:
- `notebooks/remote_ollama.ipynb` — bare Ollama 12B + tunnel (`REMOTE_OLLAMA_HOST`)
- `notebooks/cloud_server.ipynb` + `server/app.py` — full FastAPI stack
  (matcher + MT + rerank server-side, `TRANSLATE_SERVER_URL`), thin client on
  the mosque PC. **This is the preferred mode.**

## Accuracy state (do not regress)

`tests/eval_translation.py` — 47 cases, metric = MiniLM cosine vs reference
German. History in `tests/eval_results.json`.

| Config | Overall | Quran | Khutbah |
|---|---|---|---|
| Baseline 4B only | 0.744 | 0.756 | 0.739 |
| Local pipeline | 0.950 | 1.000 | 0.883 |
| + remote 12B | 0.958 | 1.000 | 0.901 |

Quran = solved (canonical retrieval — fine-tuning cannot improve it).
**The only improvable number is khutbah speech: 0.901.**

## Your mission on the GPU box

1. **Serve**: get `server/app.py` running with `translategemma:12b`
   (follow `notebooks/cloud_server.ipynb` cells — zstd → ollama → pull →
   uvicorn :8000 → cloudflared tunnel → print `TRANSLATE_SERVER_URL=`).
2. **Fine-tune** TranslateGemma **4B first** (fits single T4 with QLoRA;
   the tuned 4B also upgrades the mosque PC's offline fallback):
   - Base: `google/translategemma-4b-it` (HF, requires license acceptance
     on the owner's HF account + `HF_TOKEN`)
   - Method: QLoRA 4-bit (peft/unsloth), r=16, lr ~1e-4, 1-2 epochs,
     max_seq ~512. Keep it conservative — MT models forget fast.
   - Prompt format for training pairs must match inference exactly — see
     `PROMPT_TEMPLATE` in `src/backend/gemma_translate.py`.
3. **Evaluate** after training: run `tests/eval_translation.py` pointed at
   the tuned model. Accept only if khutbah mean improves AND Quran/overall
   do not drop. If worse: reduce lr/epochs or cut synthetic data.
4. **Save to HuggingFace** (owner's account, `HF_TOKEN` secret):
   - LoRA adapter → `<user>/translategemma-4b-khutbah-lora`
   - Merged model → `<user>/translategemma-4b-khutbah`
   - GGUF Q4_K_M (llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`)
     → same repo, so both cloud and mosque PC can pull it
   - Push checkpoints DURING training (sessions die at 9-12 h; Kaggle wipes
     the disk — HF is the only persistence)
5. **Register in Ollama**: `ollama create translategemma-khutbah -f Modelfile`
   (FROM the GGUF; copy the template/params from `ollama show translategemma:4b
   --template/--parameters`). Then set `OLLAMA_MODEL_ID=translategemma-khutbah`
   for the server.

## Training data (build it, ~30-40k pairs)

| Source | How | ~Pairs |
|---|---|---|
| Quran ar→de, 4 German editions | already have Bubenheim in `data/`; pull Khoury, Zaidan, Abu Rida from `https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/deu_<name>.min.json` (editions list: `editions.min.json`; Arabic side: `ara-quransimple`) | ~25k |
| Religious terminology | generate short parallel sentences enforcing the glossary: الله→Allah, رسول→Gesandter, تقوى→Gottesfurcht, الدنيا→Diesseits, الآخرة→Jenseits, عباد الله→Diener Allahs, صلى الله عليه وسلم→(Segen und Frieden auf ihm) | ~1k |
| Khutbah formulas + du'a | translate common sermon openings/closings/du'a with the 12B, manually spot-check, keep only clean ones | ~1-2k |
| Optional MSA anchor | small general ar-de sample (e.g. Tatoeba/OPUS) to prevent domain overfit | ~5k |

**Eval integrity: never train on the sentences in `KHUTBAH_CASES` or the
`QURAN_REFS` list inside `tests/eval_translation.py`. Check for overlap and
drop collisions.** (Training on other Quran verses is fine — those eval cases
are served by the matcher anyway.)

MCP note: `djalal/quran-mcp-server` (Quran.com API) exists for dev-time verse
lookups if useful to you; runtime stays offline-local by design.

## Environment facts

- Kaggle free: T4 x2 (2×16 GB), 30 h GPU/week, ~12 h max session, disk wiped
  after session, Internet must be enabled in settings. Secrets via Kaggle
  "Add-ons → Secrets" (`HF_TOKEN`).
- Colab free: single T4, shorter/flakier sessions. Same notebooks work.
- 4B QLoRA fits one T4. 12B QLoRA needs both T4s (device_map=auto) — only
  attempt after 4B succeeds end-to-end.
- The owner's mosque PC has no GPU (Intel Iris Xe) — anything meant to run
  there must be CPU-viable (that's why 4B GGUF matters).
- Cloudflare quick tunnels rotate URLs per session; print the
  `TRANSLATE_SERVER_URL=...` line prominently so the owner can update `.env`.

## Working rules

- Commit training scripts/notebooks to the repo (`training/` directory),
  push to `cloud-pipeline`. Model weights go to HF only, never to git.
- Log every eval run (the harness appends to `tests/eval_results.json` —
  commit that too).
- If something is ambiguous, prefer the choice that cannot regress the
  live Friday pipeline: the app must always be able to fall back to stock
  `translategemma:4b` + Helsinki locally.
