"""MinbarAI overlay app — entry point.

Run:
    python src/ui.py            # default mic
    python src/ui.py 18         # specific device id
"""
import sys

from PyQt6.QtWidgets import QApplication

from frontend.main_window import MainWindow


def main():
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


if __name__ == "__main__":
    main()
