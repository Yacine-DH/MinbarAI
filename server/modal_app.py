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

import modal

MODEL_ID = "translategemma:12b"
APP_NAME = "minbarai-translate"

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
    .env({"OLLAMA_MODEL_ID": MODEL_ID, "OLLAMA_KEEP_ALIVE": "-1"})
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
        # pull only if the volume doesn't have it yet (first deploy)
        have = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        ).stdout
        if MODEL_ID not in have:
            print(f"[modal] pulling {MODEL_ID} into volume (one-time)...", flush=True)
            subprocess.run(["ollama", "pull", MODEL_ID], check=True)
            ollama_volume.commit()

    @modal.asgi_app()
    def api(self):
        import sys

        sys.path.insert(0, "/root/minbarai")
        sys.path.insert(0, "/root/minbarai/src")
        from server.app import app as fastapi_app

        return fastapi_app
