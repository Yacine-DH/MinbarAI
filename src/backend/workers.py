"""QThread workers for the transcription + translation pipeline.

TranscribeWorker : audio chunk -> Scribe -> Arabic text -> arabic_queue
TranslateWorker  : arabic_queue -> buffer -> (Gemma | Helsinki fallback) -> emit (ar, de)
"""
import queue
import time

from PyQt6.QtCore import QThread, pyqtSignal

from backend import audio
from backend import cloud_translate
from backend import scribe_batch as transcribe_module
from backend import quran_match  # refs_overlap only (pure string helper)

arabic_queue: "queue.Queue[str]" = queue.Queue()


class TranscribeWorker(QThread):
    partial = pyqtSignal(str)

    def run(self):
        while True:
            try:
                chunk = audio.audio_queue.get(timeout=1)
                if len(chunk) < audio.SAMPLE_RATE // 2:
                    continue  # <0.5s: Scribe rejects with audio_too_short
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

    Cloud-only: every chunk goes to the translation server
    (TRANSLATE_SERVER_URL), which runs verse matcher + MT + rerank.
    No models run on this machine; while the server is unreachable the
    overlay keeps showing the live Arabic transcript only.
    """

    result = pyqtSignal(str, str, str)  # (arabic, german, quran ref or "")

    MAX_WORDS = 8
    SILENCE_TIMEOUT = 0.7

    def run(self):
        print("[translate] worker started (cloud-only)", flush=True)
        buffer = []
        last_received = None
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

    def _flush(self, buffer):
        combined = " ".join(s.strip() for s in buffer if s and s.strip()).strip()
        if not combined:
            print("[translate] flush skipped: empty combined", flush=True)
            return

        print(f"[translate] flushing: {combined[:60]!r}...", flush=True)

        if not cloud_translate.is_ready():
            print("[translate] server unreachable — chunk dropped (transcript stays live)", flush=True)
            return

        try:
            res = cloud_translate.translate(combined, context_ref=self._last_verse_ref)
        except Exception as exc:
            print(f"[translate] server call failed — chunk dropped: {exc}", flush=True)
            return

        german, ref = res.get("german", ""), res.get("ref", "")
        if not german:
            print("[translate] server returned empty translation", flush=True)
            return
        if ref and self._last_verse_ref and quran_match.refs_overlap(ref, self._last_verse_ref):
            # continued recitation of a verse whose full translation is shown
            print(f"[translate] {ref} continues {self._last_verse_ref} — skip", flush=True)
            self._last_verse_ref = ref
            return
        self._last_verse_ref = ref or None
        print(f"[translate] cloud OK [{res.get('source')}] → {german[:60]!r}", flush=True)
        self.result.emit(combined, german, ref)
