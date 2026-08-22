"""Small helpers shared across the operator modules: kernels, channels, gamma.

Ported from goncanalyser ``features/adjust.py``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np

# cv2 releases the GIL, so plain threads really do run the bands in parallel.
_BANDS = ThreadPoolExecutor(os.cpu_count() or 4)


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


def median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """``cv2.medianBlur``, bit-for-bit, but spread across cores.

    OpenCV's median filter is single-threaded for every kernel above 5 — one
    1080p BGR frame at ``ksize=7`` costs ~78 ms on a 12-core box, which is more
    than the rest of a typical chain put together, and it costs the same with
    ``setNumThreads(1)``.

    Splitting into horizontal bands that overlap by ``ksize // 2`` hands every
    band the real neighbouring rows its window needs and then crops them off, so
    each output row sees exactly the window it saw in the single-pass version —
    including ``BORDER_REPLICATE`` at the top and bottom of the frame, which stay
    real image edges. The result is identical, not approximate; see
    ``tests/test_stages.py::test_median_blur_bands_match_opencv``.
    """
    height = image.shape[0]
    bands = min(os.cpu_count() or 4, height // (4 * ksize))
    if ksize <= 3 or bands < 2:
        # ksize 3 has a sorting-network path fast enough (0.9 ms on a 1080p BGR
        # frame) that splitting it costs more than it saves. ksize 5 is already
        # worth banding: 3.4 ms -> 1.2 ms.
        return cv2.medianBlur(image, ksize)
    radius = ksize // 2
    cuts = np.linspace(0, height, bands + 1).astype(int)

    def band(index: int) -> np.ndarray:
        start, stop = int(cuts[index]), int(cuts[index + 1])
        top, bottom = max(0, start - radius), min(height, stop + radius)
        return cv2.medianBlur(image[top:bottom], ksize)[start - top : stop - top]

    return np.concatenate(list(_BANDS.map(band, range(bands))))


def masked(mask: np.ndarray, image: np.ndarray, fill: int = 0) -> np.ndarray:
    """``np.where(mask == 255, image, fill)`` — same answer, ~50x faster at 1080p.

    ``cv2.compare`` reproduces the ``== 255`` test exactly: ``copyTo`` on its own
    treats *any* non-zero mask value as a keep, which is a different filter for a
    mask carrying values other than 0 and 255.

    Anything that is not a pair of ``uint8`` arrays falls back to the numpy
    expression, which also casts the result — a float accumulator masked this way
    came out ``uint8`` before and still does.
    """
    if image.dtype != np.uint8 or mask.dtype != np.uint8:
        return np.where(mask_condition(mask, image), image, fill).astype(np.uint8)
    out = np.zeros_like(image) if fill == 0 else np.full_like(image, fill)
    cv2.copyTo(image, cv2.compare(mask, 255, cv2.CMP_EQ), out)
    return out


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
