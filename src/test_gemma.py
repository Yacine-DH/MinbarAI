"""Sanity-check TranslateGemma via Ollama."""
from backend import gemma_translate

gemma_translate.load()
if not gemma_translate.is_ready():
    raise SystemExit("Gemma not ready - check Ollama is running and model pulled")

samples = [
    "السلام عليكم ورحمة الله وبركاته",
    "الحمد لله رب العالمين",
    "إن الله مع الصابرين",
]

for ar in samples:
    print(f"\nAR: {ar}")
    print(f"DE: {gemma_translate.translate(ar)}")
