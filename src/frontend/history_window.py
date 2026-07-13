import html
import webbrowser
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from backend.history import save_history


class HistoryWindow(QMainWindow):
    def __init__(self, history):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(760, 560)
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
            QComboBox {
                background-color: #2A2A2A; color: #EEEEEE;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px 8px; min-height: 22px;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A; color: #EEEEEE;
                selection-background-color: #555;
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
        title = QLabel("Khutba History")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        export_btn = QPushButton("Export HTML")
        export_btn.clicked.connect(self.export_current)
        header.addWidget(export_btn)
        del_btn = QPushButton("Delete khutba")
        del_btn.clicked.connect(self.delete_current)
        header.addWidget(del_btn)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_all)
        header.addWidget(clear_btn)
        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(32)
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn)
        v.addLayout(header)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Khutba:"))
        self.khutba_picker = QComboBox()
        self.khutba_picker.currentIndexChanged.connect(self._on_khutba_selected)
        picker_row.addWidget(self.khutba_picker, stretch=1)
        v.addLayout(picker_row)

        self.list = QListWidget()
        v.addWidget(self.list, stretch=1)

        self.history = history
        self.refresh()

    def _fmt_dt(self, iso_str, fmt):
        try:
            return datetime.fromisoformat(iso_str).strftime(fmt)
        except Exception:
            return iso_str or "?"

    def _khutba_label(self, k):
        started = self._fmt_dt(k.get("started_at", ""), "%a %d %b %Y · %H:%M")
        ended_raw = k.get("ended_at")
        ended = self._fmt_dt(ended_raw, "%H:%M") if ended_raw else "active"
        n = len(k.get("entries", []))
        return f"{started}  →  {ended}   ({n} entries)"

    def refresh(self):
        prev_id = None
        idx = self.khutba_picker.currentIndex()
        if idx >= 0 and idx < len(self.history["khutbas"]):
            prev_id = list(reversed(self.history["khutbas"]))[idx].get("id")

        self.khutba_picker.blockSignals(True)
        self.khutba_picker.clear()
        khutbas_rev = list(reversed(self.history["khutbas"]))
        for k in khutbas_rev:
            self.khutba_picker.addItem(self._khutba_label(k))
        new_idx = 0
        if prev_id:
            for i, k in enumerate(khutbas_rev):
                if k.get("id") == prev_id:
                    new_idx = i
                    break
        self.khutba_picker.setCurrentIndex(new_idx if khutbas_rev else -1)
        self.khutba_picker.blockSignals(False)
        self._refresh_entries()

    def _on_khutba_selected(self, _idx):
        self._refresh_entries()

    def _current_khutba(self):
        idx = self.khutba_picker.currentIndex()
        if idx < 0:
            return None
        khutbas_rev = list(reversed(self.history["khutbas"]))
        if idx >= len(khutbas_rev):
            return None
        return khutbas_rev[idx]

    def _refresh_entries(self):
        self.list.clear()
        k = self._current_khutba()
        if not k:
            return

        date_header = self._fmt_dt(k.get("started_at", ""), "%A, %d %B %Y · started %H:%M")
        header_item = QListWidgetItem(f"📅  {date_header}")
        header_item.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.list.addItem(header_item)

        for e in reversed(k.get("entries", [])):
            t = self._fmt_dt(e.get("ts", ""), "%H:%M:%S")
            ar = e.get("ar", "")
            de = e.get("de", "")
            self.list.addItem(QListWidgetItem(f"🕐 {t}\nAR: {ar}\nDE: {de}"))

    def export_current(self):
        """Write the selected khutba as a standalone HTML review page and
        open it — Arabic and German side by side, one row per chunk."""
        k = self._current_khutba()
        if not k:
            return
        started = self._fmt_dt(k.get("started_at", ""), "%A, %d %B %Y %H:%M")
        rows = []
        for e in k.get("entries", []):
            t = self._fmt_dt(e.get("ts", ""), "%H:%M:%S")
            ar = html.escape(e.get("ar", ""))
            de = html.escape(e.get("de", ""))
            badge = ""
            if e.get("quran"):
                badge = f'<span class="badge">&#128214; {html.escape(e["quran"])}</span>'
            rows.append(
                f'<tr><td class="t">{t}</td>'
                f'<td class="ar" dir="rtl">{ar}</td>'
                f'<td class="de">{de} {badge}</td>'
                f'<td class="note" contenteditable="true"></td></tr>'
            )
        doc = f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<title>MinbarAI — {html.escape(started)}</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2rem auto; max-width: 1200px; color: #222; }}
 h1 {{ font-size: 1.3rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .55rem .7rem; vertical-align: top; text-align: left; }}
 th {{ background: #f4f4f4; position: sticky; top: 0; }}
 .t {{ white-space: nowrap; color: #888; font-size: .85rem; }}
 .ar {{ font-size: 1.15rem; width: 38%; }}
 .de {{ width: 42%; }}
 .note {{ width: 14%; background: #fffbe6; min-width: 8rem; }}
 .badge {{ background: #e5f3e8; color: #2c6e3f; border-radius: 4px; padding: 0 .4rem; font-size: .8rem; white-space: nowrap; }}
 p.hint {{ color: #777; font-size: .85rem; }}
</style></head><body>
<h1>&#128333; MinbarAI — Khutba vom {html.escape(started)}</h1>
<p class="hint">{len(rows)} Segmente · Notizen-Spalte ist direkt beschreibbar (Strg+S zum Sichern via Browser-Druck/PDF)</p>
<table><tr><th>Zeit</th><th>Arabisch</th><th>Deutsch</th><th>Notizen</th></tr>
{''.join(rows)}
</table></body></html>"""
        safe_id = (k.get("id") or "khutba").replace(":", "-")
        out = Path.cwd() / f"khutba_{safe_id}.html"
        out.write_text(doc, encoding="utf-8")
        webbrowser.open(out.as_uri())

    def delete_current(self):
        k = self._current_khutba()
        if not k:
            return
        self.history["khutbas"] = [x for x in self.history["khutbas"] if x is not k]
        save_history(self.history)
        self.refresh()

    def clear_all(self):
        self.history["khutbas"] = []
        save_history(self.history)
        self.refresh()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
