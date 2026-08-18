"""Geometric detectors that are more than a single cv2 call.

Ported from goncanalyser ``features/structure.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

CONTOUR_MODES = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "tree": cv2.RETR_TREE,
}


def harris_corners(
    gray: np.ndarray, k: float = 0.04, quality: float = 0.01, max_corners: int = 200
) -> np.ndarray:
    """(N, 2) Harris corner coordinates, strongest first, capped at ``max_corners``."""
    response = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=k)
    peak = response.max()
    if peak <= 0:
        return np.empty((0, 2), np.float32)
    ys, xs = np.nonzero(response > quality * peak)
    points = np.column_stack((xs, ys)).astype(np.float32)
    if len(points) <= max_corners:
        return points
    # Harris marks every pixel over the threshold, so a strong corner comes back
    # as a blob of dozens. Keep the strongest, or the cap would just truncate
    # whichever ones happened to be scanned first (i.e. the top-left of the image).
    order = np.argsort(response[ys, xs])[::-1][: int(max_corners)]
    return points[order]


def shi_tomasi_corners(
    gray: np.ndarray, max_corners: int = 200, quality: float = 0.01, min_distance: int = 10
) -> np.ndarray:
    """(N, 2) Shi-Tomasi corner coordinates."""
    found = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(1, int(max_corners)),
        qualityLevel=max(1e-4, quality),
        minDistance=max(1, int(min_distance)),
    )
    return np.empty((0, 2), np.float32) if found is None else found.reshape(-1, 2)


def blob_detector(
    min_area: float = 50.0,
    max_area: float = 5000.0,
    circularity: float = 0.0,
    convexity: float = 0.0,
    dark: bool = True,
) -> cv2.SimpleBlobDetector:
    """SimpleBlobDetector with the filters worth exposing. ``0`` disables a filter."""
    params = cv2.SimpleBlobDetector.Params()
    params.filterByColor = True
    params.blobColor = 0 if dark else 255
    params.filterByArea = True
    params.minArea = float(max(1, min_area))
    params.maxArea = float(max(min_area + 1, max_area))
    params.filterByCircularity = circularity > 0
    params.minCircularity = max(1e-3, circularity)
    params.filterByConvexity = convexity > 0
    params.minConvexity = max(1e-3, convexity)
    params.filterByInertia = False
    return cv2.SimpleBlobDetector.create(params)


def find_contours(binary: np.ndarray, mode: str, min_area: float):
    """``[(contour, parent_index)]`` above ``min_area``, with the hierarchy intact.

    The hierarchy is indexed *before* filtering: OpenCV's parent/child links are
    positions in the original list, so dropping small contours first would make
    every remaining link point at the wrong shape.
    """
    if mode not in CONTOUR_MODES:
        raise ValueError(f"unknown contour mode {mode!r}; known: {sorted(CONTOUR_MODES)}")
    found, hierarchy = cv2.findContours(binary, CONTOUR_MODES[mode], cv2.CHAIN_APPROX_SIMPLE)
    parents = [int(h[3]) for h in hierarchy[0]] if hierarchy is not None else [-1] * len(found)
    return [(c, parents[i]) for i, c in enumerate(found) if cv2.contourArea(c) >= min_area]
