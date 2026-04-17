import sys
from PyQt6.QtWidgets import QApplication
from ui import MainWindow

def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    window = MainWindow(device=device)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
