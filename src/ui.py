import json
import queue
import threading
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QCursor

import audio
import local_translate as translate_module
import scribe_batch as transcribe_module

arabic_queue = queue.Queue()

HISTORY_PATH = Path(__file__).resolve().parent.parent / "history.json"


def load_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entries):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[history] save failed: {exc}", flush=True)


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
    result = pyqtSignal(str, str)

    def run(self):
        while True:
            try:
                arabic = arabic_queue.get(timeout=1)
                german = translate_module.translate(arabic)
                self.result.emit(arabic, german)
            except Exception:
                pass
            self.msleep(10)


class HistoryWindow(QMainWindow):
    def __init__(self, entries):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(720, 520)
        self._drag_pos = None

        central = QWidget()
        central.setObjectName("histRoot")
        central.setStyleSheet("""
            QWidget#histRoot {
                background-color: rgba(15, 15, 15, 230);
                border-radius: 14px;
            }
            QListWidget {
                background: transparent;
                color: #EEEEEE;
                border: none;
                font-size: 13px;
            }
            QListWidget::item {
                border-bottom: 1px solid #333333;
                padding: 10px 6px;
            }
            QPushButton {
                background-color: #333333; color: white; border: none;
                border-radius: 4px; padding: 4px 12px;
            }
            QPushButton:hover { background-color: #555555; }
            QLabel { color: #BBBBBB; }
        """)
        self.setCentralWidget(central)

        v = QVBoxLayout(central)
        v.setContentsMargins(16, 12, 16, 16)
        v.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("History")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        header.addWidget(clear_btn)
        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(32)
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn)
        v.addLayout(header)

        self.list = QListWidget()
        v.addWidget(self.list)

        self.entries = entries
        self.refresh()

    def refresh(self):
        self.list.clear()
        for e in reversed(self.entries):
            ts = e.get("ts", "")
            ar = e.get("ar", "")
            de = e.get("de", "")
            text = f"[{ts}]\nAR: {ar}\nDE: {de}"
            item = QListWidgetItem(text)
            self.list.addItem(item)

    def add_entry(self, entry):
        self.entries.append(entry)
        self.refresh()

    def clear(self):
        self.entries.clear()
        save_history(self.entries)
        self.refresh()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


# Resize edge detection margin in pixels
RESIZE_MARGIN = 8


class MainWindow(QMainWindow):
    def __init__(self, device=None):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1280, 260)
        self.setMinimumSize(420, 140)
        self.setMouseTracking(True)

        self._drag_pos = None
        self._resize_edge = None
        self._resize_start_geom = None
        self._resize_start_pos = None

        central = QWidget()
        # alpha=1 (not 0) so margin receives mouse events for resize.
        # Fully transparent pixels are click-through on Windows with WA_TranslucentBackground.
        central.setStyleSheet("background-color: rgba(0, 0, 0, 1);")
        central.setMouseTracking(True)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(
            RESIZE_MARGIN, RESIZE_MARGIN, RESIZE_MARGIN, RESIZE_MARGIN
        )
        main_layout.setSpacing(4)

        # --- Top icon bar ---
        topbar = QWidget()
        topbar.setStyleSheet("background: transparent;")
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(8, 4, 8, 0)
        top_layout.setSpacing(6)
        top_layout.addStretch()

        icon_style = """
            QPushButton {
                background-color: rgba(0, 0, 0, 160);
                color: #FFFFFF; border: none;
                border-radius: 14px; min-width: 28px; min-height: 28px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: rgba(80, 80, 80, 200); }
        """

        self.history_btn = QPushButton("🕘")
        self.history_btn.setToolTip("History (H)")
        self.history_btn.setStyleSheet(icon_style)
        self.history_btn.clicked.connect(self.toggle_history)
        top_layout.addWidget(self.history_btn)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setToolTip("Settings (S)")
        self.settings_btn.setStyleSheet(icon_style)
        self.settings_btn.clicked.connect(self.toggle_settings)
        top_layout.addWidget(self.settings_btn)

        self.exit_btn = QPushButton("✕")
        self.exit_btn.setToolTip("Quit (Esc)")
        self.exit_btn.setStyleSheet(icon_style)
        self.exit_btn.clicked.connect(self.close)
        top_layout.addWidget(self.exit_btn)

        main_layout.addWidget(topbar)

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

        self.german_label = QLabel("Übersetzungsmodell wird geladen...")
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
        main_layout.addWidget(text_widget, stretch=1)

        # --- Controls panel (hidden by default) ---
        self.controls = QWidget()
        self.controls.setObjectName("controls")
        self.controls.setStyleSheet("""
            QWidget#controls {
                background-color: rgba(20, 20, 20, 210);
                border-radius: 8px;
            }
            QLabel { color: #AAAAAA; background: transparent; font-size: 11px; }
            QPushButton {
                background-color: #333333; color: white; border: none;
                border-radius: 4px; padding: 4px 14px;
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

        hint = QLabel("S = settings   H = history   Esc = quit")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctrl_layout.addWidget(hint)

        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(32)
        close_btn.clicked.connect(self.close)
        ctrl_layout.addWidget(close_btn)

        self.controls.hide()
        main_layout.addWidget(self.controls)

        self._hide_timer = QTimer()
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(4000)
        self._hide_timer.timeout.connect(self.controls.hide)

        self.setWindowOpacity(0.9)

        # --- History setup ---
        self.history = load_history()
        self.history_window = HistoryWindow(self.history)

        # --- Workers ---
        threading.Thread(target=translate_module.load, daemon=True).start()
        threading.Thread(target=transcribe_module.load, daemon=True).start()

        audio.start(device=device)

        self.transcriber = TranscribeWorker()
        self.transcriber.partial.connect(self.update_arabic)
        self.transcriber.start()

        self.translator = TranslateWorker()
        self.translator.result.connect(self.update_text)
        self.translator.start()

    def _make_slider(self, layout, label, min_val, max_val, default):
        container = QVBoxLayout()
        container.addWidget(QLabel(label))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setFixedWidth(130)
        container.addWidget(slider)
        layout.addLayout(container)
        return slider

    def update_text(self, arabic, german):
        self.german_label.setText(german)
        self.arabic_label.setText(arabic)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "ar": arabic,
            "de": german,
        }
        self.history.append(entry)
        save_history(self.history)
        self.history_window.refresh()

    def update_arabic(self, arabic):
        self.arabic_label.setText(arabic)

    def _reset_hide_timer(self):
        self._hide_timer.start()

    def toggle_settings(self):
        if self.controls.isVisible():
            self.controls.hide()
            self._hide_timer.stop()
        else:
            self.controls.show()
            self._reset_hide_timer()

    def toggle_history(self):
        if self.history_window.isVisible():
            self.history_window.hide()
        else:
            self.history_window.refresh()
            self.history_window.show()
            self.history_window.raise_()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_S:
            self.toggle_settings()
        elif k == Qt.Key.Key_H:
            self.toggle_history()
        elif k == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        from PyQt6.QtWidgets import QApplication
        try:
            self.history_window.close()
        except Exception:
            pass
        QApplication.quit()
        import os
        os._exit(0)

    # --- Edge-detection resize ---
    def _edge_at(self, pos):
        m = RESIZE_MARGIN
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        left = x < m
        right = x > w - m
        top = y < m
        bottom = y > h - m
        if top and left: return "tl"
        if top and right: return "tr"
        if bottom and left: return "bl"
        if bottom and right: return "br"
        if left: return "l"
        if right: return "r"
        if top: return "t"
        if bottom: return "b"
        return None

    def _cursor_for(self, edge):
        return {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
            "l": Qt.CursorShape.SizeHorCursor,
            "r": Qt.CursorShape.SizeHorCursor,
            "t": Qt.CursorShape.SizeVerCursor,
            "b": Qt.CursorShape.SizeVerCursor,
        }.get(edge, Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edge = self._edge_at(event.position().toPoint())
        if edge:
            self._resize_edge = edge
            self._resize_start_geom = self.geometry()
            self._resize_start_pos = event.globalPosition().toPoint()
        else:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        gpos = event.globalPosition().toPoint()
        # Active resize
        if self._resize_edge and event.buttons() == Qt.MouseButton.LeftButton:
            self._do_resize(gpos)
            return
        # Active drag
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(gpos - self._drag_pos)
            return
        # No drag: just update cursor if hovering near edge
        edge = self._edge_at(event.position().toPoint())
        self.setCursor(QCursor(self._cursor_for(edge)))

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        self._resize_edge = None
        self._resize_start_geom = None
        self._resize_start_pos = None
        self.unsetCursor()

    def _do_resize(self, gpos):
        dx = gpos.x() - self._resize_start_pos.x()
        dy = gpos.y() - self._resize_start_pos.y()
        g = self._resize_start_geom
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        edge = self._resize_edge
        min_w, min_h = self.minimumWidth(), self.minimumHeight()

        if "l" in edge:
            new_w = max(min_w, w - dx)
            x = x + (w - new_w)
            w = new_w
        if "r" in edge:
            w = max(min_w, w + dx)
        if "t" in edge:
            new_h = max(min_h, h - dy)
            y = y + (h - new_h)
            h = new_h
        if "b" in edge:
            h = max(min_h, h + dy)

        self.setGeometry(x, y, w, h)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    device = None
    if len(sys.argv) > 1:
        try:
            device = int(sys.argv[1])
        except ValueError:
            print(f"Invalid device id: {sys.argv[1]}")
            sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow(device=device)
    window.show()
    sys.exit(app.exec())
