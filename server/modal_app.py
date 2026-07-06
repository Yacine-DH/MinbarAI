"""MinbarAI translation server on Modal (serverless GPU).

Serves the full stack (Quran matcher + TranslateGemma 12B via Ollama +
Helsinki + QE rerank) at a permanent HTTPS URL. Scales to zero between
khutbahs — you pay GPU seconds only while translating, which fits inside
Modal's free monthly credits.

One-time setup:
    pip install modal
    modal setup                      # browser login (GitHub)
    modal deploy server/modal_app.py # prints the permanent URL

Then put in .env on the mosque PC:
    TRANSLATE_SERVER_URL=https://<workspace>--minbarai-translate-api.modal.run

First request after deploy pulls the 12B into the volume (~5 min, once).
Later cold starts: ~60-90 s (container boot + model load) — send one
warm-up request 10 minutes before the khutbah, e.g.:
    curl -X POST <URL>/translate -H "Content-Type: application/json" \
         -d '{"text": "الحمد لله رب العالمين"}'
"""
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

# fine-tuned 12B (khutbah domain, QLoRA on google/translategemma-12b-it):
# beats stock 0.9535 -> 0.9564 overall on the 74-case eval (v14).
# Rollback: set MODEL_ID = "translategemma:12b" and redeploy.
TUNED_GGUF = "https://huggingface.co/Yacinedh/translategemma-12b-khutbah/resolve/main/model-Q4_K_M.gguf"
MODEL_ID = "translategemma-khutbah-12b"
APP_NAME = "minbarai-translate"

MODELFILE_TEMPLATE = '''FROM {gguf}
TEMPLATE """{{{{- range $i, $_ := .Messages }}}}
{{{{- $last := eq (len (slice $.Messages $i)) 1 }}}}
{{{{- if or (eq .Role "user") (eq .Role "system") }}}}<start_of_turn>user
{{{{ .Content }}}}<end_of_turn>
{{{{ if $last }}}}<start_of_turn>model
{{{{ end }}}}
{{{{- else if eq .Role "assistant" }}}}<start_of_turn>model
{{{{ .Content }}}}{{{{ if not $last }}}}<end_of_turn>
{{{{ end }}}}
{{{{- end }}}}
{{{{- end }}}}"""
PARAMETER stop <end_of_turn>
PARAMETER top_k 64
PARAMETER top_p 0.95
'''

app = modal.App(APP_NAME)

# model blobs survive across containers — pulled exactly once
ollama_volume = modal.Volume.from_name("minbarai-ollama", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "fastapi",
        "uvicorn",
        "ollama",
        "rapidfuzz",
        "python-dotenv",
        "pydantic",
        "torch",
        "transformers",
        "sentencepiece",
        "sentence-transformers",
    )
    # bake the CPU models into the image so cold starts skip the downloads
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\"",
        "python -c \"from transformers import MarianMTModel, MarianTokenizer; "
        "MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ar-de'); "
        "MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ar-de', use_safetensors=False)\"",
    )
    .env({"OLLAMA_MODEL_ID": MODEL_ID, "OLLAMA_KEEP_ALIVE": "-1", "GEMMA_TIMEOUT": "45"})
    # project code + Quran data, laid out exactly like the repo so the
    # relative paths inside server/app.py and quran_match.py keep working
    .add_local_dir("src", remote_path="/root/minbarai/src")
    .add_local_dir("data", remote_path="/root/minbarai/data")
    .add_local_dir("server", remote_path="/root/minbarai/server")
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/.ollama": ollama_volume},
    secrets=[modal.Secret.from_name("minbarai-gemini")],  # rescue engine
    scaledown_window=600,   # stay warm 10 min after the last chunk
    timeout=900,
)
class Translator:
    @modal.enter()
    def start_ollama(self):
        self._ollama = subprocess.Popen(["ollama", "serve"])
        for _ in range(30):
            try:
                urllib.request.urlopen("http://localhost:11434", timeout=2)
                break
            except Exception:
                time.sleep(1)
        # provision only if the volume doesn't have it yet (first deploy)
        have = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        ).stdout
        if MODEL_ID not in have:
            if MODEL_ID.startswith("translategemma:"):
                print(f"[modal] pulling {MODEL_ID} into volume (one-time)...", flush=True)
                subprocess.run(["ollama", "pull", MODEL_ID], check=True)
            else:
                print("[modal] downloading tuned GGUF from HF (one-time)...", flush=True)
                gguf = Path("/root/.ollama/khutbah-12b.gguf")
                urllib.request.urlretrieve(TUNED_GGUF, gguf)
                Path("/root/Modelfile").write_text(MODELFILE_TEMPLATE.format(gguf=gguf))
                subprocess.run(["ollama", "create", MODEL_ID, "-f", "/root/Modelfile"], check=True)
            ollama_volume.commit()
        # load weights into the GPU now so the first chunk isn't slow and
        # gemma never times out against the fast Helsinki candidate
        print("[modal] warming model...", flush=True)
        subprocess.run(
            ["ollama", "run", MODEL_ID, "Translate to German: مرحبا"],
            capture_output=True, timeout=300,
        )
        print("[modal] warm.", flush=True)

    @modal.asgi_app()
    def api(self):
        import sys

        sys.path.insert(0, "/root/minbarai")
        sys.path.insert(0, "/root/minbarai/src")
        from server.app import app as fastapi_app

        return fastapi_app
