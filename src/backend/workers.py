"""QThread workers for the transcription + translation pipeline.

TranscribeWorker : audio chunk -> Scribe -> Arabic text -> arabic_queue
TranslateWorker  : arabic_queue -> buffer -> (Gemma | Helsinki fallback) -> emit (ar, de)
"""
import queue
import time

from PyQt6.QtCore import QThread, pyqtSignal

from backend import audio
from backend import scribe_batch as transcribe_module
from backend import gemma_translate
from backend import postprocess
from backend import quran_match
from backend import rerank
from backend import local_translate as translate_module

arabic_queue: "queue.Queue[str]" = queue.Queue()


class TranscribeWorker(QThread):
    partial = pyqtSignal(str)

    def run(self):
        while True:
            try:
                chunk = audio.audio_queue.get(timeout=1)
                arabic = transcribe_module.transcribe(chunk)
                if arabic:
                    self.partial.emit(arabic)
                    arabic_queue.put(arabic)
            except Exception:
                pass
            self.msleep(10)


class TranslateWorker(QThread):
    """Buffers Arabic transcripts, flushes when:
       * Silence > SILENCE_TIMEOUT seconds since last received
       * Buffer reaches MAX_WORDS words

    Translation chain: Gemma (local Ollama) -> Helsinki (local fallback).
    """

    result = pyqtSignal(str, str)

    MAX_WORDS = 8
    SILENCE_TIMEOUT = 0.7
    MIN_GEMMA_INTERVAL = 0.0   # local model — no rate limit needed

    def run(self):
        print("[translate] worker started", flush=True)
        buffer = []
        last_received = None
        self._last_gemma_call = 0.0
        self._last_verse_ref = None

        while True:
            try:
                arabic = arabic_queue.get(timeout=0.2)
                if arabic:
                    print(f"[translate] got arabic: {arabic!r} (buf={len(buffer)+1})", flush=True)
                    buffer.append(arabic)
                    last_received = time.time()

                    word_count = sum(len(s.split()) for s in buffer)

                    if word_count >= self.MAX_WORDS:
                        print(f"[translate] trigger: words={word_count}", flush=True)
                        self._flush(buffer)
                        buffer = []
                        last_received = None
            except queue.Empty:
                if (
                    buffer
                    and last_received
                    and (time.time() - last_received) > self.SILENCE_TIMEOUT
                ):
                    print(f"[translate] trigger: silence ({len(buffer)} items)", flush=True)
                    self._flush(buffer)
                    buffer = []
                    last_received = None
            except Exception as exc:
                print(f"[translate] worker error: {exc}", flush=True)
            self.msleep(10)

    def _gemma_ok_to_call(self):
        return (time.time() - self._last_gemma_call) >= self.MIN_GEMMA_INTERVAL

    def _flush(self, buffer):
        combined = " ".join(s.strip() for s in buffer if s and s.strip()).strip()
        if not combined:
            print("[translate] flush skipped: empty combined", flush=True)
            return

        print(f"[translate] flushing: {combined[:60]!r}...", flush=True)

        # Quran quotation? serve the canonical Bubenheim translation
        if quran_match.is_ready():
            m = quran_match.match(combined, context_ref=self._last_verse_ref)
            if m:
                if self._last_verse_ref and quran_match.refs_overlap(m.ref, self._last_verse_ref):
                    # continued recitation of a verse already displayed
                    print(f"[translate] quran {m.ref} continues {self._last_verse_ref} — skip", flush=True)
                    self._last_verse_ref = m.ref
                    return
                print(f"[translate] quran match {m.ref} (score={m.score:.0f})", flush=True)
                self._last_verse_ref = m.ref
                self.result.emit(combined, m.german)
                return
        self._last_verse_ref = None

        candidates = []
        if gemma_translate.is_ready() and self._gemma_ok_to_call():
            try:
                self._last_gemma_call = time.time()
                out = gemma_translate.translate(combined)
                if postprocess.looks_german(out):
                    candidates.append(("gemma", out))
                    print(f"[translate] gemma OK → {out[:60]!r}", flush=True)
                else:
                    print(f"[translate] gemma output not German → dropped: {out[:60]!r}", flush=True)
            except Exception as exc:
                print(f"[translate] gemma failed: {exc}", flush=True)
        else:
            print(f"[translate] gemma skipped (ready={gemma_translate.is_ready()})", flush=True)

        # Helsinki: fast local second opinion (only when loaded, or as last resort)
        if translate_module.is_ready() or not candidates:
            try:
                out = translate_module.translate(combined)
                candidates.append(("helsinki", out))
                print(f"[translate] Helsinki OK → {out[:60]!r}", flush=True)
            except Exception as exc:
                print(f"[translate] Helsinki failed: {exc}", flush=True)

        if not candidates:
            print("[translate] no translation produced", flush=True)
            return

        if len(candidates) > 1 and rerank.is_ready():
            (label, german), qe = rerank.pick(combined, candidates)
            print(f"[translate] rerank picked {label} (qe={qe:.2f})", flush=True)
        else:
            label, german = candidates[0]

        print("[translate] emitting result", flush=True)
        self.result.emit(combined, german)
