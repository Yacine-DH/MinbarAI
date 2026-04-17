from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QFont

import audio
from transcribe import transcribe
from translate import translate


class TranslationWorker(QThread):
    result = pyqtSignal(str, str)

    def run(self):
        while True:
            try:
                chunk = audio.audio_queue.get(timeout=1)
                arabic = transcribe(chunk)
                if arabic:
                    german = translate(arabic)
                    self.result.emit(arabic, german)
            except Exception:
                pass
            self.msleep(10)


class MainWindow(QMainWindow):
    def __init__(self, device=None):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # hides from taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1280, 220)
        self._drag_pos = None

        central = QWidget()
        central.setStyleSheet("background: transparent;")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # --- Text display area ---
        text_widget = QWidget()
        text_widget.setObjectName("textArea")
        text_widget.setStyleSheet("""
            QWidget#textArea {
                background-color: rgba(0, 0, 0, 180);
                border-radius: 12px;
            }
        """)
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(40, 20, 40, 20)
        text_layout.setSpacing(10)

        self.german_label = QLabel("Warte auf input...")
        self.german_label.setFont(QFont("Arial", 52))
        self.german_label.setStyleSheet("color: #FFFFFF; background: transparent;")
        self.german_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.german_label.setWordWrap(True)

        self.arabic_label = QLabel("...")
        self.arabic_label.setFont(QFont("Arial", 28))
        self.arabic_label.setStyleSheet("color: #888888; background: transparent;")
        self.arabic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.arabic_label.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.arabic_label.setWordWrap(True)

        text_layout.addWidget(self.german_label)
        text_layout.addWidget(self.arabic_label)
        main_layout.addWidget(text_widget)

        # --- Controls panel (hidden by default, toggle with S) ---
        self.controls = QWidget()
        self.controls.setObjectName("controls")
        self.controls.setStyleSheet("""
            QWidget#controls {
                background-color: rgba(20, 20, 20, 210);
                border-radius: 8px;
            }
            QLabel { color: #AAAAAA; background: transparent; font-size: 11px; }
            QPushButton {
                background-color: #333333;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 14px;
            }
            QPushButton:hover { background-color: #555555; }
        """)
        ctrl_layout = QHBoxLayout(self.controls)
        ctrl_layout.setContentsMargins(16, 8, 16, 8)
        ctrl_layout.setSpacing(24)

        self.opacity_slider = self._make_slider(ctrl_layout, "Opacity", 30, 100, 90)
        self.opacity_slider.valueChanged.connect(
            lambda v: (self.setWindowOpacity(v / 100), self._reset_hide_timer())
        )

        self.de_slider = self._make_slider(ctrl_layout, "DE size", 24, 80, 52)
        self.de_slider.valueChanged.connect(
            lambda v: (self.german_label.setFont(QFont("Arial", v)), self._reset_hide_timer())
        )

        self.ar_slider = self._make_slider(ctrl_layout, "AR size", 12, 48, 28)
        self.ar_slider.valueChanged.connect(
            lambda v: (self.arabic_label.setFont(QFont("Arial", v)), self._reset_hide_timer())
        )

        ctrl_layout.addStretch()

        hint = QLabel("S = settings   Esc = quit")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctrl_layout.addWidget(hint)

        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(32)
        close_btn.clicked.connect(self.close)
        ctrl_layout.addWidget(close_btn)

        self.controls.hide()
        main_layout.addWidget(self.controls)

        # Auto-hide controls after 4 seconds of no interaction
        self._hide_timer = QTimer()
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(4000)
        self._hide_timer.timeout.connect(self.controls.hide)

        self.setWindowOpacity(0.9)

        audio.start(device=device)
        self.worker = TranslationWorker()
        self.worker.result.connect(self.update_text)
        self.worker.start()

    def _make_slider(self, layout, label: str, min_val: int, max_val: int, default: int) -> QSlider:
        container = QVBoxLayout()
        container.addWidget(QLabel(label))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setFixedWidth(130)
        container.addWidget(slider)
        layout.addLayout(container)
        return slider

    def update_text(self, arabic: str, german: str):
        self.german_label.setText(german)
        self.arabic_label.setText(arabic)

    def _reset_hide_timer(self):
        self._hide_timer.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            if self.controls.isVisible():
                self.controls.hide()
                self._hide_timer.stop()
            else:
                self.controls.show()
                self._reset_hide_timer()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
