import threading
import torch
from transformers import MarianMTModel, MarianTokenizer

_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-de"
_tokenizer = None
_model = None
_lock = threading.Lock()
_ready = threading.Event()


def load():
    global _tokenizer, _model
    try:
        print("[translate] loading tokenizer...", flush=True)
        _tokenizer = MarianTokenizer.from_pretrained(_MODEL_NAME)
        print("[translate] loading model...", flush=True)
        _model = MarianMTModel.from_pretrained(_MODEL_NAME, use_safetensors=False)
        _model.eval()
        print("[translate] warmup...", flush=True)
        with torch.no_grad():
            dummy = _tokenizer(["مرحبا"], return_tensors="pt")
            _model.generate(**dummy, num_beams=1, max_length=16)
        _ready.set()
        print("[translate] ready.", flush=True)
    except Exception as e:
        import traceback
        print(f"[translate] LOAD FAILED: {e}", flush=True)
        traceback.print_exc()


def translate(text: str) -> str:
    _ready.wait()
    with _lock:
        with torch.no_grad():
            inputs = _tokenizer(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=256
            )
            outputs = _model.generate(
                **inputs,
                num_beams=1,
                max_length=128,
                do_sample=False,
            )
            return _tokenizer.decode(outputs[0], skip_special_tokens=True)
