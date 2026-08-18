"""Texture description: Histograms of Oriented Gradients, Local Binary Patterns.

Ported from goncanalyser ``features/texture.py``. Both stages replace ``ctx.image``
with their visualisation and put the numbers that describe the texture into
``ctx.metrics``, so a ``json`` sink records them and a ``display`` sink shows them.

**HOG is the expensive operator here** — roughly 150-300 ms on a 640x512 frame,
which is slower than a video frame arrives. Fine for a batch, not for a preview.
"""

from __future__ import annotations

import cv2
import numpy as np

from segmentator.ops.common import to_gray
from segmentator.ops.texture import hog_of, lbp_of
from segmentator.pipeline import Ctx, register


@register("stage", "hog")
class Hog:
    """Histogram of Oriented Gradients. Emits the gradient visualisation."""

    def __init__(self, orientations: int = 9, cell: int = 8, block: int = 2):
        self.orientations, self.cell, self.block = orientations, cell, block

    def apply(self, ctx: Ctx) -> None:
        vector, image = hog_of(to_gray(ctx.image), self.orientations, self.cell, self.block)
        ctx.metrics["hog_dim"] = int(vector.size)
        ctx.metrics["hog_mean"] = round(float(vector.mean()), 4)
        ctx.store["hog"] = vector
        ctx.image = image


@register("stage", "lbp")
class Lbp:
    """Local Binary Patterns. Emits the code image, stretched for display.

    ``uniform`` tops out at ``points + 1`` (10 by default), which as raw grey
    levels is a black rectangle, hence the normalise.
    """

    def __init__(self, points: int = 8, radius: int = 1, method: str = "uniform"):
        self.points, self.radius, self.method = points, radius, method

    def apply(self, ctx: Ctx) -> None:
        codes = lbp_of(to_gray(ctx.image), self.points, self.radius, self.method)

        # One bin per distinct code. "var" is continuous, so cap the bin count
        # rather than trying to enumerate every float it produces.
        top = int(codes.max()) + 1
        counts = np.bincount(codes.astype(np.int64).ravel(), minlength=top)[: min(top, 256)]
        share = counts / (counts.sum() or 1)
        nonzero = share[share > 0]

        ctx.metrics["lbp_bins"] = int(len(counts))
        # Entropy is the one number that says whether the texture is varied or
        # dominated by a single pattern, which is what the descriptor is for.
        ctx.metrics["lbp_entropy"] = round(float(-(nonzero * np.log2(nonzero)).sum()), 3)
        ctx.rows["lbp"] = [{"code": i, "count": int(c)} for i, c in enumerate(counts)]
        ctx.image = cv2.normalize(codes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
