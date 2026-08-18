"""Global texture descriptors: HOG and LBP, both from scikit-image.

Ported from goncanalyser ``features/texture.py``. OpenCV's ``HOGDescriptor`` is
built for the pedestrian detector — no visualisation, a rigid window contract —
and OpenCV has no LBP in its Python API at all, so this is the one place the
engine reaches outside cv2.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

LBP_METHODS = ("uniform", "default", "ror", "nri_uniform", "var")


def hog_of(
    gray: np.ndarray, orientations: int = 9, cell: int = 8, block: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """(descriptor vector, visualisation as uint8)."""
    vector, image = hog(
        gray,
        orientations=max(2, int(orientations)),
        pixels_per_cell=(max(2, int(cell)),) * 2,
        cells_per_block=(max(1, int(block)),) * 2,
        visualize=True,
        feature_vector=True,
    )
    return vector, cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def lbp_of(gray: np.ndarray, points: int = 8, radius: int = 1, method: str = "uniform"):
    """LBP codes as a float array, in whatever range the method produces."""
    if method not in LBP_METHODS:
        raise ValueError(f"unknown lbp method {method!r}; known: {sorted(LBP_METHODS)}")
    codes = local_binary_pattern(
        gray, P=max(1, int(points)), R=max(1, int(radius)), method=method
    )
    # "var" is NaN wherever the neighbourhood variance is undefined — on the
    # border, and on flat patches. Zero is the honest value there: no variation.
    return np.nan_to_num(codes)
