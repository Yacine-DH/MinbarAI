"""Experimental Modal app serving a FINE-TUNED TranslateGemma GGUF from HF.

Separate app from production (minbarai-translate) — safe to deploy/stop
freely for A/B evaluation. Flip TUNED_GGUF/BASE_MODEL to compare stock vs
tuned through the identical pipeline.

    modal deploy server/modal_app_tuned.py
    python tests/eval_translation.py --server <printed url> --tag <tag>
"""
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

# --- what to serve ---
TUNED_GGUF = "https://huggingface.co/Yacinedh/translategemma-4b-khutbah/resolve/main/model-Q4_K_M.gguf"
SERVE_TUNED = False                   # False -> serve STOCK_MODEL instead
STOCK_MODEL = "translategemma:4b"
TUNED_NAME = "translategemma-khutbah"

APP_NAME = "minbarai-translate-tuned"
app = modal.App(APP_NAME)

volume = modal.Volume.from_name("minbarai-ollama-tuned", create_if_missing=True)

# gemma turn template exactly as the stock Ollama translategemma ships it
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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "fastapi", "uvicorn", "ollama", "rapidfuzz", "python-dotenv",
        "pydantic", "torch", "transformers", "sentencepiece",
        "sentence-transformers",
    )
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\"",
        "python -c \"from transformers import MarianMTModel, MarianTokenizer; "
        "MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ar-de'); "
        "MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ar-de', use_safetensors=False)\"",
    )
    .env({
        "OLLAMA_MODEL_ID": TUNED_NAME if SERVE_TUNED else STOCK_MODEL,
        "OLLAMA_KEEP_ALIVE": "-1",
        "GEMMA_TIMEOUT": "45",
    })
    .add_local_dir("src", remote_path="/root/minbarai/src")
    .add_local_dir("data", remote_path="/root/minbarai/data")
    .add_local_dir("server", remote_path="/root/minbarai/server")
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/.ollama": volume},
    secrets=[modal.Secret.from_name("minbarai-gemini")],
    scaledown_window=600,
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
        have = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout

        if SERVE_TUNED and TUNED_NAME not in have:
            print("[modal-tuned] downloading tuned GGUF from HF (one-time)...", flush=True)
            gguf = Path("/root/.ollama/tuned.gguf")
            urllib.request.urlretrieve(TUNED_GGUF, gguf)
            Path("/root/Modelfile").write_text(MODELFILE_TEMPLATE.format(gguf=gguf))
            subprocess.run(["ollama", "create", TUNED_NAME, "-f", "/root/Modelfile"], check=True)
            volume.commit()
        if not SERVE_TUNED and STOCK_MODEL not in have:
            subprocess.run(["ollama", "pull", STOCK_MODEL], check=True)
            volume.commit()

        model = TUNED_NAME if SERVE_TUNED else STOCK_MODEL
        print(f"[modal-tuned] warming {model}...", flush=True)
        subprocess.run(["ollama", "run", model, "Translate to German: مرحبا"],
                       capture_output=True, timeout=300)
        print("[modal-tuned] warm.", flush=True)

    @modal.asgi_app()
    def api(self):
        import sys
        sys.path.insert(0, "/root/minbarai")
        sys.path.insert(0, "/root/minbarai/src")
        from server.app import app as fastapi_app
        return fastapi_app
