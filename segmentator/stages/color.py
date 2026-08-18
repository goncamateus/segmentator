"""Colour statistics: per-channel histograms and their moments.

Ported from goncanalyser ``features/color.py``, where it was always on because
three ``calcHist`` calls over a 640x512 frame cost about a millisecond. Here it
is a stage like any other, because a chain that never looks at colour should not
pay for it.
"""

from __future__ import annotations

import numpy as np

from segmentator.ops.color import SPACES, histograms, plot
from segmentator.ops.common import to_bgr
from segmentator.pipeline import Ctx, register


@register("stage", "histogram")
class Histogram:
    """Per-channel histograms for ``rgb``, ``hsv`` or ``lab``.

    Leaves ``ctx.image`` alone by default and publishes the 512x256 plot as
    ``ctx.store["histogram"]``, so a sink reaches it with ``input: histogram``
    while the chain carries on with the actual frame. ``replace: true`` swaps the
    plot into ``ctx.image`` instead, for a chain that only wants the plot.

    Channel mean and standard deviation land in ``ctx.metrics``, computed from bin
    centres so they are mean *intensity* rather than mean count.
    """

    def __init__(self, space: str = "rgb", replace: bool = False):
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
