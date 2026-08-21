"""Region-of-interest masking and the fixed background model.

These have no goncanalyser counterpart — they are the gas-plume recipe, where the
interesting region is a fixed dark window in an otherwise static scene, and the
background is the mean of the first N frames rather than an adaptive model.
"""

from __future__ import annotations

import cv2
import numpy as np

from segmentator.background_model import BackgroundModel
from segmentator.ops.common import mask_condition, require_gray
from segmentator.pipeline import Ctx, frame_for, register


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

    # Both parameters are stored as attributes, but the mask is fitted once and
    # cached — assigning them later would not refit it. The editor reads this to
    # know it must rebuild the stage rather than poke the live instance.
    RECONSTRUCT = ("threshold", "invert")

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
    """Zero (or fill) everything outside a binary mask, in a chosen image.

    Both ``input`` and ``mask`` resolve the same way as a sink's ``input:`` or
    ``select``'s — see :func:`~segmentator.pipeline.frame_for` — so either can
    name ``source``, a named tap, or a published artifact such as
    ``contour_mask``.

    Args:
        input: Which image gets masked. Defaults to ``"image"``, the current
            working frame.
        mask: Which image to use as the mask. Defaults to ``"mask"``,
            ``static_mask``'s published key, so existing configs are
            unaffected. Point it at a named tap to use a ``threshold`` or
            motion stage's output instead, e.g. after
            ``{type: mog2, name: fg}`` -> ``{type: threshold, mode: binary,
            name: fg_mask}`` -> ``{type: apply_mask, mask: fg_mask}``.
        fill: Value written outside the mask.
    """

    def __init__(self, input: str = "image", mask: str = "mask", fill: int = 0):
        self.input = input
        self.mask = mask
        self.fill = fill

    def apply(self, ctx: Ctx) -> None:
        image = frame_for(ctx, self.input)
        mask = frame_for(ctx, self.mask)
        ctx.image = np.where(mask_condition(mask, image), image, self.fill).astype(np.uint8)


_BUFFER_MODES = ("static", "circular")


@register("stage", "mean_background")
class MeanBackground:
    """Subtract the mean of the last ``n_frames`` — a background estimate.

    Args:
        n_frames: Frames averaged to build the model.
        use_mask: Accumulate only the masked region (``ctx.store["mask"]``) while
            still subtracting from the full frame. This keeps activity outside the
            region of interest out of the background estimate.
        buffer: ``static`` (default) freezes the mean after the first
            ``n_frames`` — a fixed background for a static scene. ``circular``
            keeps a rolling window instead, so the estimate follows slow
            lighting drift.
    """

    def __init__(self, n_frames: int = 60, use_mask: bool = True, buffer: str = "static"):
        if buffer not in _BUFFER_MODES:
            raise ValueError(f"unknown buffer mode {buffer!r}; known: {_BUFFER_MODES}")
        self.use_mask = use_mask
        self._model = BackgroundModel(n_frames, circular=buffer == "circular")

    @property
    def ready(self) -> bool:
        return self._model.ready

    def apply(self, ctx: Ctx) -> None:
        frame = ctx.image
        mask = ctx.store.get("mask") if self.use_mask else None
        sample = frame if mask is None else np.where(mask_condition(mask, frame), frame, 0).astype(np.uint8)
        self._model.accumulate(sample)
        ctx.image = self._model.subtract(frame)
