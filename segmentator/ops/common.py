"""Small helpers shared across the operator modules: kernels, channels, gamma.

Ported from goncanalyser ``features/adjust.py``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


def odd_kernel(size: int) -> int:
    """Nearest valid kernel size: odd and at least 1.

    ``GaussianBlur`` and ``medianBlur`` both throw on an even or zero kernel, and
    a config can absolutely hold either, so this clamps rather than trusting the
    value it was given.
    """
    size = max(1, int(size))
    return size if size % 2 else size + 1


def to_gray(image: np.ndarray) -> np.ndarray:
    """Single-channel view of anything. Already-grey input passes straight through."""
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Three-channel view of anything, so callers never have to branch on ndim."""
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


@lru_cache(maxsize=64)
def gamma_lut(gamma: float) -> np.ndarray:
    """256-entry gamma curve. Cached — rebuilding it per frame is pure waste."""
    return np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255, 0, 255).astype(np.uint8)


def mask_condition(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """A boolean condition from a single-channel mask, broadcastable against ``image``.

    ``np.where`` aligns shapes from the trailing axis, so a ``(H, W)`` mask does
    not line up with a ``(H, W, 3)`` colour image on its own — this puts the
    channel axis back when ``image`` has one, so a mask fitted in grayscale
    still applies correctly to BGR, HSV or any other colour space.
    """
    condition = mask == 255
    return condition[..., None] if image.ndim == 3 else condition


def require_gray(image: np.ndarray, stage: str) -> None:
    """Fail with the fix named, rather than deep inside a cv2 assertion."""
    if image.ndim != 2:
        raise ValueError(
            f"stage {stage!r} needs a single-channel image, got shape {image.shape}; "
            "put a 'gray' stage earlier in the chain"
        )
