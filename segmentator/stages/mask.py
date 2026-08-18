"""Region-of-interest masking and the fixed background model.

These have no goncanalyser counterpart — they are the gas-plume recipe, where the
interesting region is a fixed dark window in an otherwise static scene, and the
background is the mean of the first N frames rather than an adaptive model.
"""

from __future__ import annotations

import cv2
import numpy as np

from segmentator.background_model import BackgroundModel
from segmentator.ops.common import require_gray
from segmentator.pipeline import Ctx, register


@register("stage", "static_mask")
class StaticMask:
    """Fit a fixed region-of-interest mask on the first frame it sees.

    Publishes the mask as ``ctx.store["mask"]`` on every frame; ``apply_mask`` and
    ``mean_background`` pick it up from there. Leaves ``ctx.image`` alone.

    Args:
        threshold: Grey level splitting region from background.
        invert: ``True`` (default) selects the *dark* region, matching the original
            gas-plume footage where the region of interest is darker than its frame.
    """

    def __init__(self, threshold: int = 127, invert: bool = True):
        self.threshold = threshold
        self.invert = invert
        self.mask: np.ndarray | None = None

    def apply(self, ctx: Ctx) -> None:
        if self.mask is None:
            require_gray(ctx.image, "static_mask")
            mode = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
            _, self.mask = cv2.threshold(ctx.image, self.threshold, 255, mode)
        ctx.store["mask"] = self.mask


@register("stage", "apply_mask")
class ApplyMask:
    """Zero (or fill) everything outside ``ctx.store["mask"]``."""

    def __init__(self, fill: int = 0):
        self.fill = fill

    def apply(self, ctx: Ctx) -> None:
        mask = ctx.store.get("mask")
        if mask is None:
            raise KeyError("'apply_mask' needs a mask; add a 'static_mask' stage before it")
        ctx.image = np.where(mask == 255, ctx.image, self.fill).astype(np.uint8)


@register("stage", "mean_background")
class MeanBackground:
    """Subtract the mean of the first ``n_frames`` — a fixed background estimate.

    Args:
        n_frames: Frames averaged to build the model.
        use_mask: Accumulate only the masked region (``ctx.store["mask"]``) while
            still subtracting from the full frame. This keeps activity outside the
            region of interest out of the background estimate.
    """

    def __init__(self, n_frames: int = 60, use_mask: bool = True):
        self.use_mask = use_mask
        self._model = BackgroundModel(n_frames)

    @property
    def ready(self) -> bool:
        return self._model.ready

    def apply(self, ctx: Ctx) -> None:
        frame = ctx.image
        mask = ctx.store.get("mask") if self.use_mask else None
        sample = frame if mask is None else np.where(mask == 255, frame, 0).astype(np.uint8)
        self._model.accumulate(sample)
        ctx.image = self._model.subtract(frame)
