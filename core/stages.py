"""Built-in pipeline stages.

Importing this module registers every stage below under ``kind="stage"``, so
they become available to :meth:`core.pipeline.Pipeline.from_config`.

Stages fall into three loose groups — pre-processing, masking/detection and
post-processing — but they share one protocol and the chain order is entirely up
to the config. A stage that needs to be fitted on the first frame (``static_mask``)
does so lazily inside ``apply``.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from core.background_model import BackgroundModel
from core.pipeline import Ctx, register

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


def _require_gray(image: np.ndarray, stage: str) -> None:
    if image.ndim != 2:
        raise ValueError(
            f"stage {stage!r} needs a single-channel image, got shape {image.shape}; "
            "put a 'gray' stage earlier in the chain"
        )


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #


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
        to: Target space. Resolved as ``cv2.COLOR_BGR2<TO>``, so anything OpenCV
            supports works (``hsv``, ``lab``, ``ycrcb``, ``hls``, ...).
    """

    def __init__(self, to: str):
        attr = f"COLOR_BGR2{to.upper()}"
        code = getattr(cv2, attr, None)
        if code is None:
            raise ValueError(f"unknown colour space {to!r} (no cv2.{attr})")
        self.to = to
        self._code = code

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.cvtColor(ctx.image, self._code)


@register("stage", "median_blur")
class MedianBlur:
    """Median smoothing. ``ksize`` must be odd."""

    def __init__(self, ksize: int = 7):
        self.ksize = ksize

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.medianBlur(ctx.image, self.ksize)


@register("stage", "gaussian_blur")
class GaussianBlur:
    """Gaussian smoothing. ``ksize`` must be odd; ``sigma=0`` derives it from ksize."""

    def __init__(self, ksize: int = 5, sigma: float = 0.0):
        self.ksize = ksize
        self.sigma = sigma

    def apply(self, ctx: Ctx) -> None:
        ctx.image = cv2.GaussianBlur(ctx.image, (self.ksize, self.ksize), self.sigma)


@register("stage", "clahe")
class Clahe:
    """Contrast-limited adaptive histogram equalisation on a grayscale frame."""

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))

    def apply(self, ctx: Ctx) -> None:
        _require_gray(ctx.image, "clahe")
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


# --------------------------------------------------------------------------- #
# Masking and detection
# --------------------------------------------------------------------------- #


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
            _require_gray(ctx.image, "static_mask")
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


@register("stage", "mog2")
class Mog2:
    """OpenCV MOG2 adaptive background subtractor. Outputs a foreground mask."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        learning_rate: float = -1.0,
    ):
        self.learning_rate = learning_rate
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )

    def apply(self, ctx: Ctx) -> None:
        ctx.image = self._subtractor.apply(ctx.image, learningRate=self.learning_rate)


@register("stage", "knn")
class Knn:
    """OpenCV KNN adaptive background subtractor. Outputs a foreground mask."""

    def __init__(
        self,
        history: int = 500,
        dist2_threshold: float = 400.0,
        detect_shadows: bool = False,
        learning_rate: float = -1.0,
    ):
        self.learning_rate = learning_rate
        self._subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, dist2Threshold=dist2_threshold, detectShadows=detect_shadows
        )

    def apply(self, ctx: Ctx) -> None:
        ctx.image = self._subtractor.apply(ctx.image, learningRate=self.learning_rate)


@register("stage", "frame_diff")
class FrameDiff:
    """Absolute difference against the frame ``lag`` positions back.

    Emits a zero frame until enough history exists, so the output shape is stable
    from the very first frame.
    """

    def __init__(self, lag: int = 1):
        if lag < 1:
            raise ValueError(f"frame_diff lag must be >= 1, got {lag}")
        self._history: deque[np.ndarray] = deque(maxlen=lag)

    def apply(self, ctx: Ctx) -> None:
        current = ctx.image
        previous = self._history[0] if len(self._history) == self._history.maxlen else None
        self._history.append(current)
        ctx.image = np.zeros_like(current) if previous is None else cv2.absdiff(current, previous)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


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

    def __init__(self, value: int = 127, mode: str = "binary", maxval: int = 255, otsu: bool = False):
        if mode not in _THRESHOLD_MODES:
            raise ValueError(f"unknown threshold mode {mode!r}; known: {sorted(_THRESHOLD_MODES)}")
        self.value = value
        self.mode = mode
        self.maxval = maxval
        self.otsu = otsu

    def apply(self, ctx: Ctx) -> None:
        flags = _THRESHOLD_MODES[self.mode]
        if self.otsu:
            _require_gray(ctx.image, "threshold(otsu)")
            flags |= cv2.THRESH_OTSU
        _, ctx.image = cv2.threshold(ctx.image, self.value, self.maxval, flags)


@register("stage", "contours")
class Contours:
    """Draw external contours above ``min_area``.

    Args:
        min_area: Contours smaller than this are dropped.
        color: BGR triple for the outline.
        thickness: Outline thickness in pixels.
        draw_on: ``source`` overlays the original colour frame (the usual choice);
            ``image`` draws on the current working frame, promoted to BGR.
    """

    def __init__(
        self,
        min_area: float = 20.0,
        color: tuple[int, int, int] = (161, 177, 247),
        thickness: int = 2,
        draw_on: str = "source",
    ):
        if draw_on not in {"source", "image"}:
            raise ValueError(f"contours draw_on must be 'source' or 'image', got {draw_on!r}")
        self.min_area = min_area
        self.color = tuple(int(c) for c in color)
        self.thickness = thickness
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        _require_gray(ctx.image, "contours")
        _, binary = cv2.threshold(ctx.image, 0, 255, cv2.THRESH_BINARY)
        found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = [c for c in found if cv2.contourArea(c) >= self.min_area]
        canvas = _canvas(ctx, self.draw_on)
        cv2.drawContours(canvas, keep, -1, self.color, self.thickness)
        ctx.store["contours"] = keep
        ctx.image = canvas


@register("stage", "bounding_boxes")
class BoundingBoxes:
    """Draw axis-aligned boxes around external contours above ``min_area``."""

    def __init__(
        self,
        min_area: float = 20.0,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_on: str = "source",
    ):
        if draw_on not in {"source", "image"}:
            raise ValueError(f"bounding_boxes draw_on must be 'source' or 'image', got {draw_on!r}")
        self.min_area = min_area
        self.color = tuple(int(c) for c in color)
        self.thickness = thickness
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        found = ctx.store.get("contours")
        if found is None:
            _require_gray(ctx.image, "bounding_boxes")
            _, binary = cv2.threshold(ctx.image, 0, 255, cv2.THRESH_BINARY)
            found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = _canvas(ctx, self.draw_on)
        boxes = [
            cv2.boundingRect(c) for c in found if cv2.contourArea(c) >= self.min_area
        ]
        for x, y, w, h in boxes:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, self.thickness)
        ctx.store["boxes"] = boxes
        ctx.image = canvas


def _canvas(ctx: Ctx, draw_on: str) -> np.ndarray:
    """A fresh BGR array to draw overlays on, so nothing is mutated in place."""
    base = ctx.source if draw_on == "source" else ctx.image
    if base.ndim == 2:
        return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return base.copy()
