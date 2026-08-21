"""Multi-channel colour histograms, drawn with OpenCV rather than a plot library.

Ported from goncanalyser ``features/color.py``. The histogram is a picture, and
the pipeline already has a fast path for turning a BGR ndarray into a displayed
or encoded frame; drawing the plot with ``cv2.polylines`` reuses it exactly and
adds no dependency for what is thirty lines of line drawing.
"""

from __future__ import annotations

import cv2
import numpy as np

# space -> (conversion from BGR, channel names, per-channel draw colour in BGR).
# Colours are the channel's own where that means something (R drawn red) and
# merely distinguishable where it does not (there is no colour for "hue").
SPACES: dict[str, tuple[int | None, tuple[str, ...], tuple[tuple[int, int, int], ...]]] = {
    "bgr": (None, ("B", "G", "R"), ((255, 80, 80), (80, 255, 80), (80, 80, 255))),
    "hsv": (cv2.COLOR_BGR2HSV, ("H", "S", "V"), ((255, 128, 0), (0, 200, 255), (230, 230, 230))),
    "lab": (cv2.COLOR_BGR2LAB, ("L", "a", "b"), ((230, 230, 230), (128, 128, 255), (255, 200, 0))),
}

PLOT_W, PLOT_H = 512, 256
GRID = (60, 60, 60)


def histograms(bgr: np.ndarray, space: str) -> np.ndarray:
    """(3, 256) counts for the chosen space, one row per channel."""
    if space not in SPACES:
        raise ValueError(f"unknown histogram space {space!r}; known: {sorted(SPACES)}")
    code, _, _ = SPACES[space]
    img = bgr if code is None else cv2.cvtColor(bgr, code)
    return np.stack([cv2.calcHist([img], [c], None, [256], [0, 256]).ravel() for c in range(3)])


def select_mask(
    bgr: np.ndarray,
    space: str,
    ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    """Boolean-AND band-pass mask across all three channels of ``space``, as 0/255 uint8."""
    if space not in SPACES:
        raise ValueError(f"unknown colour-select space {space!r}; known: {sorted(SPACES)}")
    code, _, _ = SPACES[space]
    img = bgr if code is None else cv2.cvtColor(bgr, code)
    keep = np.ones(img.shape[:2], dtype=bool)
    for channel, (lo, hi) in zip(cv2.split(img), ranges):
        keep &= (channel >= lo) & (channel <= hi)
    return (keep.astype(np.uint8)) * 255


def plot(counts: np.ndarray, space: str, w: int = PLOT_W, h: int = PLOT_H) -> np.ndarray:
    """The histograms as a BGR image: one polyline per channel, plus a legend."""
    _, names, colors = SPACES[space]
    canvas = np.zeros((h, w, 3), np.uint8)
    for i in range(1, 4):  # quarter gridlines, so the shape is readable
        cv2.line(canvas, (w * i // 4, 0), (w * i // 4, h), GRID, 1)
        cv2.line(canvas, (0, h * i // 4), (w, h * i // 4), GRID, 1)

    # One shared scale across the three channels: normalising each on its own
    # would make a flat channel look as busy as a peaked one.
    peak = counts.max() or 1.0
    xs = np.linspace(0, w - 1, 256).astype(np.int32)
    for row, color in zip(counts, colors):
        ys = (h - 1 - row / peak * (h - 1)).astype(np.int32)
        cv2.polylines(canvas, [np.column_stack((xs, ys))], False, color, 1, cv2.LINE_AA)

    for i, (name, color) in enumerate(zip(names, colors)):
        cv2.putText(
            canvas, name, (8 + i * 22, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return canvas
