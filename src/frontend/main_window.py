import os
import threading
import time
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QAbstractAnimation
from PyQt6.QtGui import QFont, QCursor

from backend import audio
from backend import gemma_translate
from backend import local_translate as translate_module
from backend import scribe_batch as transcribe_module
from backend.history import load_history, save_history
from backend.workers import TranscribeWorker, TranslateWorker

from frontend.history_window import HistoryWindow

# Resize edge detection margin in pixels
RESIZE_MARGIN = 8


class MainWindow(QMainWindow):
    MIN_DISPLAY_MS = 3000  # keep each translation visible at least 3s

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

        self.german_label = QLabel("Starting")
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

        # --- Fade transition setup ---
        self._de_opacity = QGraphicsOpacityEffect(self.german_label)
        self.german_label.setGraphicsEffect(self._de_opacity)
        self._de_opacity.setOpacity(1.0)

        self._ar_opacity = QGraphicsOpacityEffect(self.arabic_label)
        self.arabic_label.setGraphicsEffect(self._ar_opacity)
        self._ar_opacity.setOpacity(1.0)

        self._live_anims = set()  # keep refs alive across both label swaps

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

        # --- Display pacing state ---
        self._display_unlock_at = 0
        self._pending_display = None
        self._display_timer = QTimer()
        self._display_timer.setSingleShot(True)
        self._display_timer.timeout.connect(self._show_pending)

        # --- History + new khutba session ---
        self.history = load_history()
        now_iso = datetime.now().isoformat(timespec="seconds")
        self.current_khutba = {
            "id": now_iso,
            "started_at": now_iso,
            "ended_at": None,
            "entries": [],
        }
        self.history["khutbas"].append(self.current_khutba)
        save_history(self.history)
        self.history_window = HistoryWindow(self.history)

        # --- Background loaders ---
        threading.Thread(target=translate_module.load, daemon=True).start()
        threading.Thread(target=gemma_translate.load, daemon=True).start()
        threading.Thread(target=transcribe_module.load, daemon=True).start()

        audio.start(device=device)

        # --- Workers ---
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

    # --- Slot: receive translation result ---
    def update_text(self, arabic, german):
        # Persist + history (always immediate, no pacing)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "ar": arabic,
            "de": german,
        }
        self.current_khutba["entries"].append(entry)
        save_history(self.history)
        self.history_window.refresh()

        # Display pacing: ensure shown >= MIN_DISPLAY_MS before swap
        now_ms = int(time.time() * 1000)
        if now_ms >= self._display_unlock_at:
            self._show_now(arabic, german)
        else:
            # Coalesce: keep newest pending; older pending discarded from display
            self._pending_display = (arabic, german)
            remaining = self._display_unlock_at - now_ms
            self._display_timer.start(remaining)

    FADE_MS = 220

    def _show_now(self, arabic, german):
        self._fade_swap(self.german_label, self._de_opacity, german)
        self._fade_swap(self.arabic_label, self._ar_opacity, arabic)
        self._display_unlock_at = int(time.time() * 1000) + self.MIN_DISPLAY_MS
        self._pending_display = None

    def _fade_swap(self, label, effect, new_text):
        """Fade out → set new text → fade in."""
        fade_out = QPropertyAnimation(effect, b"opacity", label)
        fade_out.setDuration(self.FADE_MS)
        fade_out.setStartValue(effect.opacity())
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        fade_in = QPropertyAnimation(effect, b"opacity", label)
        fade_in.setDuration(self.FADE_MS)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._live_anims.add(fade_out)
        self._live_anims.add(fade_in)

        def on_out_done():
            label.setText(new_text)
            fade_in.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        fade_out.finished.connect(on_out_done)
        fade_out.finished.connect(lambda: self._live_anims.discard(fade_out))
        fade_in.finished.connect(lambda: self._live_anims.discard(fade_in))
        fade_out.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _show_pending(self):
        if self._pending_display:
            ar, de = self._pending_display
            self._show_now(ar, de)

    def update_arabic(self, arabic):
        self.arabic_label.setText(arabic)

    # --- Settings panel auto-hide ---
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
        try:
            self.current_khutba["ended_at"] = datetime.now().isoformat(timespec="seconds")
            save_history(self.history)
        except Exception:
            pass
        try:
            self.history_window.close()
        except Exception:
            pass
        QApplication.quit()
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
        if self._resize_edge and event.buttons() == Qt.MouseButton.LeftButton:
            self._do_resize(gpos)
            return
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(gpos - self._drag_pos)
            return
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
