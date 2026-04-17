from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import audio
from transcribe import transcribe
from translate import translate


class TranslationWorker(QThread):
    result = pyqtSignal(str, str)  # emits (arabic, german)

    def run(self):
        while True:
            try:
                audio_chunk = audio.audio_queue.get(timeout=1)
                arabic = transcribe(audio_chunk)
                if arabic:
                    german = translate(arabic)
                    self.result.emit(arabic, german)
            except Exception:
                pass
            self.msleep(10)


class MainWindow(QMainWindow):
    def __init__(self, device: int = None):
        super().__init__()
        self.setWindowTitle("MinbarAI")
        self.setStyleSheet("background-color: #000000;")
        self.resize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(30)

        self.german_label = QLabel("Warte auf input...")
        self.german_label.setFont(QFont("Arial", 52))
        self.german_label.setStyleSheet("color: #FFFFFF;")
        self.german_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.german_label.setWordWrap(True)

        self.arabic_label = QLabel("...")
        self.arabic_label.setFont(QFont("Arial", 28))
        self.arabic_label.setStyleSheet("color: #888888;")
        self.arabic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.arabic_label.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.arabic_label.setWordWrap(True)

        self.status_label = QLabel("🎤 Listening...")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setStyleSheet("color: #444444;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(self.german_label)
        layout.addWidget(self.arabic_label)
        layout.addStretch()
        layout.addWidget(self.status_label)

        audio.start(device=device)

        self.worker = TranslationWorker()
        self.worker.result.connect(self.update_text)
        self.worker.start()

    def update_text(self, arabic: str, german: str):
        self.german_label.setText(german)
        self.arabic_label.setText(arabic)
