"""Motion measurement helpers that are more than a single cv2 call.

Ported from goncanalyser ``features/motion.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from .common import odd_kernel

# Optical flow is measured in pixels per frame; everything else here is measured
# in grey levels. This is the conversion that lets one threshold serve both:
# 8 px/frame of flow reads as full scale. It is a calibration constant, not a
# derived one — a slow wide-angle plume may want it higher.
FLOW_GAIN = 32.0

# MOG2 and KNN label shadows 127 and real foreground 255. A shadow is the moving
# thing's effect on the background, not the moving thing.
FOREGROUND = 255


def sparse_flow(
    prev: np.ndarray,
    gray: np.ndarray,
    max_points: int = 200,
    win: int = 15,
    gain: float = FLOW_GAIN,
) -> np.ndarray:
    """Lucas-Kanade at tracked corners, painted into a dense float32 image.

    LK is sparse by construction — it answers "how fast did *this point* move",
    not "which pixels moved". Splatting each vector as a disc the size of its
    search window is what lets it feed the same mask and heat accumulator as the
    dense algorithms.

    ponytail: a disc is not a segmentation; use ``farneback`` when the shape of
    the moving thing matters.
    """
    heat = np.zeros(gray.shape, np.float32)
    points = cv2.goodFeaturesToTrack(
        prev, maxCorners=max(1, int(max_points)), qualityLevel=0.01, minDistance=7
    )
    if points is None:
        return heat

    size = max(3, odd_kernel(win))
    moved, found, _ = cv2.calcOpticalFlowPyrLK(
        prev, gray, points, None, winSize=(size, size), maxLevel=3
    )
    for (x0, y0), (x1, y1), ok in zip(
        points.reshape(-1, 2), moved.reshape(-1, 2), found.ravel()
    ):
        if not ok:
            continue
        value = min(255.0, float(np.hypot(x1 - x0, y1 - y0)) * gain)
        cv2.circle(heat, (int(x1), int(y1)), size // 2, value, cv2.FILLED)
    return heat


def blend(canvas: np.ndarray, heat: np.ndarray, opacity: float, threshold: float) -> None:
    """Paint a JET heatmap over ``canvas``, in place. Silently skips a shape mismatch.

    Per-pixel alpha rather than one opacity over the whole frame: JET maps zero to
    dark blue, so a flat blend would wash the entire image blue wherever nothing
    is happening. Weighting the blend by the heat itself leaves cold areas as the
    untouched frame, which is what makes the overlay readable.
    """
    if heat is None or canvas.shape[:2] != heat.shape[:2]:
        return
    level = np.clip(heat / 255.0, 0.0, 1.0)
    level[level < max(0.0, float(threshold))] = 0.0
    colored = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (level * float(opacity))[..., None]
    canvas[:] = (canvas * (1 - alpha) + colored * alpha).astype(np.uint8)


def speeds(centres, previous, limit: float) -> list[float]:
    """Pixels moved since the last frame, by nearest previous centroid.

    Greedy and identity-free: it answers "how fast is something moving here", not
    "where did object 7 go".

    ponytail: two blobs that cross swap speeds; add a real tracker (Hungarian, or
    cv2.TrackerCSRT) only if per-object identity is ever needed.
    """
    out = []
    for cx, cy in centres:
        best = min((np.hypot(cx - px, cy - py) for px, py in previous), default=None)
        out.append(round(float(best), 1) if best is not None and best <= limit else 0.0)
    return out
