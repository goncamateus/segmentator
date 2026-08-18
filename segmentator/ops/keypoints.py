"""SIFT and ORB behind one normalised sensitivity knob.

Ported from goncanalyser ``features/keypoints.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

# detector -> (threshold at sensitivity 0, threshold at sensitivity 1).
# Both ranges run strict -> permissive, so the knob reads the same way for each
# even though the two detectors' native numbers are on unrelated scales.
SENSITIVITY: dict[str, tuple[float, float]] = {
    "sift": (0.16, 0.005),  # contrastThreshold
    "orb": (60.0, 3.0),  # fastThreshold
}


def threshold_for(detector: str, sensitivity: float) -> float:
    """The detector's native threshold, from the normalised 0..1 knob."""
    lo, hi = SENSITIVITY[detector]
    return lo + (hi - lo) * min(1.0, max(0.0, sensitivity))


def build(
    detector: str,
    max_keypoints: int = 500,
    sensitivity: float = 0.5,
    octaves: int = 3,
    edge: float = 10.0,
):
    """The configured detector."""
    if detector not in SENSITIVITY:
        raise ValueError(f"unknown detector {detector!r}; known: {sorted(SENSITIVITY)}")
    value = threshold_for(detector, sensitivity)
    octaves = max(1, int(octaves))
    if detector == "sift":
        return cv2.SIFT.create(
            nfeatures=max(0, int(max_keypoints)),
            nOctaveLayers=octaves,
            contrastThreshold=value,
            edgeThreshold=edge,
        )
    return cv2.ORB.create(
        nfeatures=max(1, int(max_keypoints)),
        nlevels=octaves + 5,
        edgeThreshold=max(1, int(edge)),
        fastThreshold=max(1, int(value)),
    )


def detect(detector, gray: np.ndarray, max_keypoints: int):
    """(keypoints, descriptors), strongest first and capped at ``max_keypoints``.

    SIFT's ``nfeatures`` cap is applied *before* its own scoring in some builds and
    ORB's is a target rather than a limit, so the cap is re-applied here.
    """
    found, described = detector.detectAndCompute(gray, None)
    if not found:
        return [], None
    if len(found) > max_keypoints:
        order = np.argsort([-kp.response for kp in found])[: int(max_keypoints)]
        found = [found[i] for i in order]
        described = None if described is None else described[order]
    return found, described
