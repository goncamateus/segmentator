"""Screenshot the real editor, offscreen, for the paper's editor figures.

The window is the shipped :class:`segmentator.gui.window.MainWindow` driven by
the offscreen Qt platform — not a mockup, and not a hand-drawn diagram of one.
The preview pane is filled by the same background worker the interactive editor
uses, so the capture waits on the event loop until a frame has actually arrived
rather than grabbing an empty pane.

    uv sync --extra gui
    uv run python docs/whitepaper/capture.py

Dark theme on purpose: the page is white, and a dark window keeps the screenshot
a distinct object on it rather than a pale rectangle bleeding into the margin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Before PyQt6, and for the same reason segmentator/gui/main.py does it: the
# opencv-python wheel repoints QT_QPA_PLATFORM_PLUGIN_PATH at its own Qt5
# plugins and PyQt6 then cannot load a platform plugin at all — offscreen
# included.
import cv2  # noqa: F401

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
CONFIG = HERE / "configs" / "motion.yaml"

WINDOW = (1360, 820)
SETTLE_MS = 6000  # how long the preview worker gets to produce a frame
FRAME = 400  # far enough in that the plume has developed and the metrics are non-zero
STAGE_ROW = 2  # farneback: the stage in this chain with a form worth showing


def main() -> int:
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication

    from segmentator.gui import style
    from segmentator.gui.window import MainWindow

    if not CONFIG.exists():
        raise SystemExit(f"{CONFIG} is missing — run bench.py first, it writes the configs")

    app = QApplication(sys.argv)
    style.apply(app, "dark")
    window = MainWindow(CONFIG)
    window.set_theme("dark")
    window.resize(*WINDOW)
    window.show()

    def settle(milliseconds: int) -> None:
        """Spin the event loop: the preview arrives from a worker thread through
        it, so a sleeping main thread grabs an empty pane."""
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    settle(SETTLE_MS)
    # A stage with real parameters, and a frame where the plume exists: an empty
    # form over a blank first frame illustrates neither the form nor the preview.
    window.stage_list.setCurrentRow(min(STAGE_ROW, window.stage_list.count() - 1))
    window.playback.seek(FRAME)
    settle(SETTLE_MS)

    FIGURES.mkdir(exist_ok=True)
    target = FIGURES / "editor-window.png"
    window.grab().save(str(target))

    # A second shot on a mid-chain tap rather than the final image: what the
    # preview tabs are for.
    if window.tabs.count() > 1:
        window.tabs.setCurrentIndex(window.tabs.count() - 1)
    settle(SETTLE_MS)
    window.grab().save(str(FIGURES / "editor-form.png"))

    print(f"wrote {target} and {FIGURES / 'editor-form.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
