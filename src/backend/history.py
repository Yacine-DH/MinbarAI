"""Khutba history persistence (JSON file at project root)."""
import json
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "history.json"


def load_history():
    """Returns dict {'khutbas': [...]}. Auto-migrates legacy flat list format."""
    if not HISTORY_PATH.exists():
        return {"khutbas": []}
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if not data:
                return {"khutbas": []}
            return {
                "khutbas": [{
                    "id": "legacy",
                    "started_at": data[0].get("ts", ""),
                    "ended_at": data[-1].get("ts", ""),
                    "entries": data,
                }]
            }
        return data
    except Exception:
        return {"khutbas": []}


def save_history(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[history] save failed: {exc}", flush=True)
