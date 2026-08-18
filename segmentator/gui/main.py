"""Entry point for ``segmentator-gui``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Imported before PyQt6 on purpose. The opencv-python wheel points
# QT_QPA_PLATFORM_PLUGIN_PATH at the Qt5 plugins it ships, and PyQt6 then cannot
# load its own xcb plugin. goncanalyser sidesteps this by depending on
# opencv-python-headless; segmentator cannot, because DisplaySink needs the full
# wheel to open a window in a batch run.
import cv2  # noqa: F401

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Edit a segmentator pipeline with a live preview.")
    parser.add_argument("config", type=Path, nargs="?", help="pipeline YAML to open")
    args = parser.parse_args()

    from PyQt6.QtWidgets import QApplication, QFileDialog

    from segmentator.gui.window import MainWindow

    app = QApplication(sys.argv)
    path = args.config
    if path is None:
        chosen, _ = QFileDialog.getOpenFileName(None, "Open config", "configs", "YAML (*.yaml *.yml)")
        if not chosen:
            return 0
        path = Path(chosen)
    window = MainWindow(path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
