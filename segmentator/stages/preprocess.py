"""Pre-processing stages: the frame every detector downstream reads.

Adjustments (brightness, contrast, saturation, gamma), colour space, blur,
threshold and the region of interest. Ported from goncanalyser
``features/adjust.py``, where the same operators ran in one fixed order behind a
single ``Settings``; here each is its own stage and the config picks the order.
"""

from __future__ import annotations

import cv2
import numpy as np

from segmentator.ops.common import gamma_lut, odd_kernel, require_gray, to_bgr
from segmentator.pipeline import Ctx, frame_for, register

_THRESHOLD_MODES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
}

_MORPH_OPS = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}

_ADAPTIVE_METHODS = {
    "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


# Every cv2.COLOR_BGR2* conversion, keyed by its lower-cased short name.
_COLOR_CODES = {
    name[len("COLOR_BGR2") :].lower(): getattr(cv2, name)
    for name in dir(cv2)
    if name.startswith("COLOR_BGR2")
}


@register("stage", "gray")
class Gray:
    """Convert BGR to grayscale. No-op if the frame is already single-channel."""

    def apply(self, ctx: Ctx) -> None:
        if ctx.image.ndim == 3:
            ctx.image = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)


@register("stage", "colorspace")
class ColorSpace:
    """Convert the frame to another colour space, e.g. ``to: hsv`` or ``to: lab``.

    Args:
        to: Target space. Resolved against ``cv2.COLOR_BGR2*``, so anything
            OpenCV supports works (``hsv``, ``lab``, ``hls``, ``ycrcb``, ...).

    Matched case-insensitively, because OpenCV's own names are not consistently
    cased — ``COLOR_BGR2HSV`` but ``COLOR_BGR2YCrCb`` — and no config should have
    to know which is which.
    """

    def __init__(self, to: str):
        if to.lower() not in _COLOR_CODES:
            raise ValueError(
                f"unknown colour space {to!r}; known: {sorted(_COLOR_CODES)}"
            )
        self.to = to

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.cvtColor(ctx.image, _COLOR_CODES[self.to.lower()])


@register("stage", "brightness_contrast")
class BrightnessContrast:
    """Additive offset and multiplicative gain, in one pass.

    Args:
        brightness: Added to every channel, roughly -100..100.
        contrast: Gain, ``1.0`` is identity.
    """

    def __init__(self, brightness: int = 0, contrast: float = 1.0):
        self.brightness = brightness
        self.contrast = contrast

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.convertScaleAbs(ctx.image, alpha=self.contrast, beta=self.brightness)


@register("stage", "saturation")
class Saturation:
    """Scale the HSV saturation channel. ``gain: 1.0`` is identity. BGR only."""

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def apply(self, ctx: Ctx) -> None:
        if ctx.image.ndim != 3:
            raise ValueError("stage 'saturation' needs a 3-channel BGR image")
        # split/merge rather than an in-place slice: hsv[:, :, 1] is a
        # non-contiguous view and OpenCV refuses those.
        hue, sat, value = cv2.split(cv2.cvtColor(ctx.image, cv2.COLOR_BGR2HSV))
        ctx.image = cv2.cvtColor(
            cv2.merge((hue, cv2.convertScaleAbs(sat, alpha=self.gain), value)),
            cv2.COLOR_HSV2BGR,
        )


@register("stage", "gamma")
class Gamma:
    """Gamma correction through a cached 256-entry LUT. ``<1`` darkens, ``>1`` lifts."""

    def __init__(self, value: float = 1.0):
        if value <= 0:
            raise ValueError(f"gamma must be > 0, got {value}")
        self.value = value

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.LUT(ctx.image, gamma_lut(self.value))


@register("stage", "median_blur")
class MedianBlur:
    """Median smoothing. ``ksize`` is clamped to the nearest odd value."""

    def __init__(self, ksize: int = 7):
        self.ksize = odd_kernel(ksize)

    def apply(self, ctx: Ctx) -> None:
        if self.ksize > 1:
            ctx.image = cv2.medianBlur(ctx.image, self.ksize)


@register("stage", "gaussian_blur")
class GaussianBlur:
    """Gaussian smoothing. ``ksize`` is clamped odd; ``sigma=0`` derives it from ksize."""

    def __init__(self, ksize: int = 5, sigma: float = 0.0):
        self.ksize = odd_kernel(ksize)
        self.sigma = sigma

    def apply(self, ctx: Ctx) -> None:
        if self.ksize > 1:
            ctx.image = cv2.GaussianBlur(ctx.image, (self.ksize, self.ksize), self.sigma)


@register("stage", "clahe")
class Clahe:
    """Contrast-limited adaptive histogram equalisation on a grayscale frame."""

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "clahe")
        ctx.image = self._clahe.apply(ctx.image)


@register("stage", "morphology")
class Morphology:
    """Morphological op — ``open``, ``close``, ``erode``, ``dilate``, ``gradient``, ..."""

    def __init__(self, op: str = "open", ksize: int = 3, iterations: int = 1):
        if op not in _MORPH_OPS:
            raise ValueError(f"unknown morphology op {op!r}; known: {sorted(_MORPH_OPS)}")
        self.op = op
        self.iterations = iterations
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.morphologyEx(
            ctx.image, _MORPH_OPS[self.op], self._kernel, iterations=self.iterations
        )


@register("stage", "resize")
class Resize:
    """Resize to ``size: [width, height]``."""

    def __init__(self, size: tuple[int, int]):
        self.size = (int(size[0]), int(size[1]))

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.resize(ctx.image, self.size)


@register("stage", "threshold")
class Threshold:
    """Threshold the frame.

    Args:
        value: Threshold level, ignored when ``otsu`` is set.
        mode: One of ``binary``, ``binary_inv``, ``trunc``, ``tozero``, ``tozero_inv``.
            ``tozero`` keeps pixel magnitudes above ``value`` and zeroes the rest;
            ``tozero_inv`` at a high level clips saturated pixels away.
        otsu: Derive ``value`` automatically per frame (grayscale only).
    """

    def __init__(
        self, value: int = 127, mode: str = "binary", maxval: int = 255, otsu: bool = False
    ):
        if mode not in _THRESHOLD_MODES:
            raise ValueError(f"unknown threshold mode {mode!r}; known: {sorted(_THRESHOLD_MODES)}")
        self.value = value
        self.mode = mode
        self.maxval = maxval
        self.otsu = otsu

    def apply(self, ctx: Ctx) -> None:
        flags = _THRESHOLD_MODES[self.mode]
        if self.otsu:
            require_gray(ctx.image, "threshold(otsu)")
            flags |= cv2.THRESH_OTSU
        _, ctx.image = cv2.threshold(ctx.image, self.value, self.maxval, flags)


@register("stage", "adaptive_threshold")
class AdaptiveThreshold:
    """Per-neighbourhood threshold, for frames whose lighting is not uniform.

    Args:
        method: ``mean`` or ``gaussian`` — how the local level is computed.
        block: Neighbourhood size. Clamped odd and to at least 3, which OpenCV requires.
        c: Subtracted from the local mean; larger keeps less foreground.
        invert: Emit ``THRESH_BINARY_INV`` instead of ``THRESH_BINARY``.
    """

    def __init__(
        self, method: str = "gaussian", block: int = 11, c: float = 2.0, invert: bool = False
    ):
        if method not in _ADAPTIVE_METHODS:
            raise ValueError(
                f"unknown adaptive method {method!r}; known: {sorted(_ADAPTIVE_METHODS)}"
            )
        self.method = method
        self.block = max(3, odd_kernel(block))
        self.c = c
        self.invert = invert

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "adaptive_threshold")
        ctx.image = cv2.adaptiveThreshold(
            ctx.image,
            255,
            _ADAPTIVE_METHODS[self.method],
            cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY,
            self.block,
            self.c,
        )


# A region smaller than this is not analysable: SIFT and ORB build a scale
# pyramid and cv2.resize refuses a zero dimension, so a stray 1-px rectangle
# would take the run down. Sized in pixels, small enough never to get in the way.
MIN_ROI = 16

ROI_COLOR = (0, 255, 255)


def roi_rect(shape, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    """A rectangle fitted to this frame. Ported from goncanalyser ``adjust.roi_rect``.

    Fitted rather than trusted, in three ways, because all three happen in normal
    use: a rectangle written for a 1920x1080 clip has to mean something on a
    640x512 one; ``0`` for a dimension means "out to the edge", which keeps the
    frame size out of the config; and a stray rectangle is a few pixels wide.

    The size is clamped into ``MIN_ROI..frame`` and the origin is then pulled back
    far enough for it to fit — sliding the rectangle in beats shrinking it to a
    sliver. When the frame itself is under ``MIN_ROI``, the whole frame is the region.
    """
    height, width = shape[:2]
    cw = min(width, max(MIN_ROI, width if w <= 0 else int(w)))
    ch = min(height, max(MIN_ROI, height if h <= 0 else int(h)))
    return (min(max(0, int(x)), width - cw), min(max(0, int(y)), height - ch), cw, ch)


@register("stage", "roi")
class Roi:
    """Crop to a rectangle, so everything downstream measures only that region.

    The crop has to happen *before* the operators that read it, not after: Otsu
    takes its level from whatever histogram it is handed and a blur reads a
    kernel's worth of neighbours across the border, so analysing the full frame
    and cropping at the end would let pixels outside the rectangle change the
    numbers inside it.

    Publishes the fitted rectangle as ``ctx.store["roi"]`` for :class:`PasteRoi`.
    A copy, never a view — an overlay drawn on the region must not write through
    into the full frame kept behind it.
    """

    def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        self.rect = (x, y, w, h)

    def apply(self, ctx: Ctx) -> None:
        x, y, w, h = roi_rect(ctx.image.shape, *self.rect)
        ctx.store["roi"] = (x, y, w, h)
        ctx.image = ctx.image[y : y + h, x : x + w].copy()


# Row keys that hold a coordinate, so a region's results can be reported in frame
# space. Everything else a stage emits — ``bin``, ``code``, ``area``, ``response``
# — is not a position and must be left alone.
X_KEYS = ("x", "x1", "x2")
Y_KEYS = ("y", "y1", "y2")


@register("stage", "paste_roi")
class PasteRoi:
    """Put the analysed region back into the full frame, and fix up the rows.

    Compositing the region first and pasting after is what avoids translating
    every overlay: the stages already drew at region coordinates, and this moves
    all of them at once. ``ctx.rows`` is a different matter — a keypoint at
    (5, 5) of the region is not at (5, 5) of the frame, and nothing in an
    exported CSV would say so, so every coordinate key is shifted here.

    Args:
        border: Draw a one-pixel outline around the pasted region.
        draw_on: ``source`` (the untouched frame), ``image`` or a named tap to
            paste into — resolved the same way as any other stage's ``draw_on``.
    """

    def __init__(self, border: bool = True, draw_on: str = "source"):
        self.border = border
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        rect = ctx.store.get("roi")
        if rect is None:
            raise KeyError("'paste_roi' needs a region; add a 'roi' stage before it")
        x, y, w, h = rect

        for rows in ctx.rows.values():
            for row in rows:
                for key in X_KEYS:
                    if key in row:
                        row[key] += x
                for key in Y_KEYS:
                    if key in row:
                        row[key] += y

        region = ctx.image
        frame = canvas(ctx, self.draw_on)
        if region.shape[:2] == (h, w):
            frame[y : y + h, x : x + w] = to_bgr(region)
        if self.border:
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), ROI_COLOR, 1)
        ctx.image = frame


@register("stage", "select")
class Select:
    """Rebind ``ctx.image`` to an earlier image, so the chain can branch.

    A drawing stage consumes the working frame — ``hough_lines`` with
    ``draw_on: source`` leaves a colour overlay behind, and the edge map it read
    is gone. This puts one back:

        - {type: canny, name: edges}
        - {type: hough_lines, draw_on: source, name: lines}
        - {type: select, input: edges}      # back to the edge map
        - {type: contours, draw_on: image}

    ``input`` resolves exactly as a sink's does — see
    :func:`~segmentator.pipeline.frame_for` — so ``source``, a named tap and a
    published artifact such as ``contour_mask`` all work.

    This is what a linear chain has instead of a second input edge. It is enough
    for a tree (one producer, several consumers) and it is not enough for a merge
    (two producers into one stage), which the config format cannot express.
    """

    def __init__(self, input: str = "source"):
        self.input = input

    def apply(self, ctx: Ctx) -> None:
        ctx.image = frame_for(ctx, self.input)


def canvas(ctx: Ctx, draw_on: str) -> np.ndarray:
    """A fresh BGR array to draw overlays on, so nothing is mutated in place.

    ``draw_on`` resolves exactly as a sink's ``input:`` and ``select``'s do — see
    :func:`~segmentator.pipeline.frame_for` — so ``source`` (the original colour
    frame, the usual choice), ``image`` (the current working frame, which is how
    an overlay composes onto a mid-chain view such as an edge map) and a named
    tap all work. Anything single-channel is promoted to BGR on the way out.
    """
    return to_bgr(frame_for(ctx, draw_on)).copy()
