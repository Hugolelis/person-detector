import sys

from main import parse_args
from PySide6.QtWidgets import QApplication
from src.config_loader import load_config
from src.errors.base import PersonDetectorError
from src.gui.main_window import MainWindow


def main():
    args = parse_args(video_required=False)
    try:
        config = load_config(args.config)
    except PersonDetectorError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow(args.video, config, args.config, args.output)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
