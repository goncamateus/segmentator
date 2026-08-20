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

    from PyQt6.QtCore import QSettings
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from segmentator.gui import launcher, style
    from segmentator.gui.window import MainWindow

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(launcher.icon_path())))
    # The theme the window was last left in. The window writes it back on every
    # toggle; a first run has no key and gets the light one.
    style.apply(app, str(QSettings("segmentator", "gui").value("theme", "light")))
    path = args.config
    if path is None:
        # No config named, so ask which one — or offer to write a first one. Closing
        # the launcher is how you leave without opening anything.
        path = launcher.choose()
        if path is None:
            return 0
    window = MainWindow(path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
