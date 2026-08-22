"""Colour statistics: per-channel histograms and their moments.

Ported from goncanalyser ``features/color.py``, where it was always on because
three ``calcHist`` calls over a 640x512 frame cost about a millisecond. Here it
is a stage like any other, because a chain that never looks at colour should not
pay for it.
"""

from __future__ import annotations

import numpy as np

from segmentator.ops.color import SPACES, histograms, plot, select_mask
from segmentator.ops.common import masked, to_bgr
from segmentator.pipeline import Ctx, register


@register("stage", "histogram")
class Histogram:
    """Per-channel histograms for ``bgr``, ``hsv`` or ``lab``.

    Leaves ``ctx.image`` alone by default and publishes the 512x256 plot as
    ``ctx.store["histogram"]``, so a sink reaches it with ``input: histogram``
    while the chain carries on with the actual frame. ``replace: true`` swaps the
    plot into ``ctx.image`` instead, for a chain that only wants the plot.

    Channel mean and standard deviation land in ``ctx.metrics``, computed from bin
    centres so they are mean *intensity* rather than mean count.
    """

    def __init__(self, space: str = "bgr", replace: bool = False):
        if space not in SPACES:
            raise ValueError(f"unknown histogram space {space!r}; known: {sorted(SPACES)}")
        self.space = space
        self.replace = replace

    def apply(self, ctx: Ctx) -> None:
        counts = histograms(to_bgr(ctx.image), self.space)
        _, names, _ = SPACES[self.space]

        bins = np.arange(256)
        for row, name in zip(counts, names):
            total = row.sum() or 1.0
            mean = float((row * bins).sum() / total)
            var = float((row * (bins - mean) ** 2).sum() / total)
            ctx.metrics[f"{name}_mean"] = round(mean, 1)
            ctx.metrics[f"{name}_sd"] = round(var**0.5, 1)

        ctx.rows["histogram"] = [
            {"bin": int(b), **{n: int(c) for n, c in zip(names, counts[:, b])}} for b in bins
        ]
        picture = plot(counts, self.space)
        ctx.store["histogram"] = picture
        if self.replace:
            ctx.image = picture


@register("stage", "color_select")
class ColorSelect:
    """Band-pass filter per channel in a chosen colour space.

    Converts to ``space``, keeps pixels whose value on every channel falls
    inside that channel's ``(lo, hi)`` range, and ANDs the three conditions
    together — narrowing one channel excludes a colour band (e.g. HSV hue
    ``ch0=(35, 130)`` keeps green-through-blue, drops red) while leaving a
    channel at ``(0, 255)`` passes it through unrestricted. OpenCV's 8-bit
    HSV hue tops out at 179, not 255.

    Filters ``ctx.image`` immediately — pixels outside the band become
    ``fill`` — and also publishes the mask as ``ctx.store["mask"]``, the same
    key ``static_mask`` uses, so a later stage that reads it (``apply_mask``
    on a different image, ``mean_background``'s ``use_mask``) picks it up
    with no extra wiring.

    Args:
        space: ``bgr``, ``hsv`` or ``lab``.
        ch0, ch1, ch2: Inclusive ``(lo, hi)`` range for each of ``space``'s
            three channels, in ``SPACES``' channel order. The editor labels
            these with the space's own channel letters (R/G/B, H/S/V, L/a/b).
        fill: Value written outside the band.
    """

    def __init__(
        self,
        space: str = "hsv",
        ch0: tuple[int, int] = (0, 255),
        ch1: tuple[int, int] = (0, 255),
        ch2: tuple[int, int] = (0, 255),
        fill: int = 0,
    ):
        self.space = space
        self.ch0 = ch0
        self.ch1 = ch1
        self.ch2 = ch2
        self.fill = fill

    def apply(self, ctx: Ctx) -> None:
        mask = select_mask(to_bgr(ctx.image), self.space, (self.ch0, self.ch1, self.ch2))
        ctx.store["mask"] = mask
        ctx.image = masked(mask, ctx.image, self.fill)
