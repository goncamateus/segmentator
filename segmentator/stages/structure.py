"""Structural and geometric detectors: edges, lines, circles, corners, contours, blobs.

Ported from goncanalyser ``features/structure.py``. Where that module ran every
enabled detector over one shared frame and collected the drawing callables to
composite later, here each detector is its own stage: chain order *is* the
composition, so ``canny`` followed by ``harris(draw_on: image)`` puts the corners
on the edge map, and ``draw_on: source`` puts them on the original footage.

Each detector publishes its findings to ``ctx.store`` (so a later stage can reuse
them), counts to ``ctx.metrics`` and per-object detail to ``ctx.rows``.
"""

from __future__ import annotations

import cv2
import numpy as np

from segmentator.ops.common import odd_kernel, require_gray, to_gray
from segmentator.ops.structure import (
    blob_detector,
    find_contours,
    harris_corners,
    shi_tomasi_corners,
)
from segmentator.pipeline import Ctx, register
from segmentator.stages.preprocess import canvas

LINE_COLOR = (0, 255, 0)
CIRCLE_COLOR = (255, 255, 0)
CORNER_COLOR = (0, 0, 255)
CONTOUR_COLOR = (0, 255, 255)
BOX_COLOR = (255, 160, 0)
BLOB_COLOR = (255, 0, 255)


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #


@register("stage", "canny")
class Canny:
    """Canny edge detector. Emits a single-channel binary edge map."""

    def __init__(self, lo: int = 100, hi: int = 200):
        self.lo = lo
        self.hi = hi

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "canny")
        ctx.image = cv2.Canny(ctx.image, int(self.lo), int(self.hi))
        ctx.metrics["edge_px"] = int(np.count_nonzero(ctx.image))


@register("stage", "sobel")
class Sobel:
    """Sobel derivative magnitude, scaled back to uint8.

    ``ksize`` is clamped to 1/3/5/7 and ``dx``/``dy`` may not both be zero, both
    of which OpenCV throws on rather than defaulting.
    """

    def __init__(self, ksize: int = 3, dx: int = 1, dy: int = 1):
        self.ksize = min(7, odd_kernel(ksize))
        self.dx, self.dy = int(dx), int(dy)
        if not self.dx and not self.dy:
            self.dx = 1

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "sobel")
        ctx.image = cv2.convertScaleAbs(
            cv2.Sobel(ctx.image, cv2.CV_64F, self.dx, self.dy, ksize=self.ksize)
        )
        ctx.metrics["edge_px"] = int(np.count_nonzero(ctx.image))


@register("stage", "laplacian")
class Laplacian:
    """Laplacian second derivative, scaled back to uint8."""

    def __init__(self, ksize: int = 3):
        self.ksize = min(31, odd_kernel(ksize))

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "laplacian")
        ctx.image = cv2.convertScaleAbs(cv2.Laplacian(ctx.image, cv2.CV_64F, ksize=self.ksize))
        ctx.metrics["edge_px"] = int(np.count_nonzero(ctx.image))


# --------------------------------------------------------------------------- #
# Hough
# --------------------------------------------------------------------------- #


@register("stage", "hough_lines")
class HoughLines:
    """Probabilistic Hough line transform, drawn as segments.

    Always builds its own Canny from ``canny_lo``/``canny_hi`` rather than reading
    ``ctx.image`` directly: ``HoughLinesP`` needs a *binary* edge map, and a Sobel
    magnitude or a raw grey frame is not one.
    """

    def __init__(
        self,
        canny_lo: int = 100,
        canny_hi: int = 200,
        threshold: int = 120,
        min_len: int = 50,
        max_gap: int = 10,
        draw_on: str = "source",
    ):
        self.canny_lo, self.canny_hi = canny_lo, canny_hi
        self.threshold, self.min_len, self.max_gap = threshold, min_len, max_gap
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "hough_lines")
        binary = cv2.Canny(ctx.image, int(self.canny_lo), int(self.canny_hi))
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=max(1, int(self.threshold)),
            minLineLength=max(1, int(self.min_len)),
            maxLineGap=max(0, int(self.max_gap)),
        )
        # OpenCV 4 returns (N, 1, 4) here and OpenCV 5 returns (N, 4). Reshaping
        # rather than indexing `[:, 0]` reads the same on both.
        lines = np.empty((0, 4), np.int32) if lines is None else lines.reshape(-1, 4)

        ctx.store["lines"] = lines
        ctx.metrics["lines"] = len(lines)
        ctx.rows["lines"] = [
            {"x1": int(a), "y1": int(b), "x2": int(c), "y2": int(d)} for a, b, c, d in lines
        ]
        out = canvas(ctx, self.draw_on)
        for a, b, c, d in lines:
            cv2.line(out, (a, b), (c, d), LINE_COLOR, 1)
        ctx.image = out


@register("stage", "hough_circles")
class HoughCircles:
    """Hough circle transform, drawn as outlines."""

    def __init__(
        self,
        dp: float = 1.0,
        min_dist: int = 50,
        param1: int = 200,
        param2: int = 120,
        draw_on: str = "source",
    ):
        self.dp, self.min_dist = dp, min_dist
        self.param1, self.param2 = param1, param2
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "hough_circles")
        circles = cv2.HoughCircles(
            ctx.image,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=max(1, int(self.min_dist)),
            param1=max(1, int(self.param1)),
            param2=max(1, int(self.param2)),
        )
        found = (
            np.empty((0, 3), int)
            if circles is None
            else np.round(circles.reshape(-1, 3)).astype(int)
        )

        ctx.store["circles"] = found
        ctx.metrics["circles"] = len(found)
        ctx.rows["circles"] = [{"x": int(x), "y": int(y), "r": int(r)} for x, y, r in found]
        out = canvas(ctx, self.draw_on)
        for x, y, r in found:
            cv2.circle(out, (x, y), r, CIRCLE_COLOR, 1)
        ctx.image = out


# --------------------------------------------------------------------------- #
# Corners
# --------------------------------------------------------------------------- #


class _Corners:
    """Shared publish-and-draw half of the two corner stages."""

    draw_on = "source"

    def _publish(self, ctx: Ctx, points: np.ndarray) -> None:
        ctx.store["corners"] = points
        ctx.metrics["corners"] = len(points)
        ctx.rows["corners"] = [{"x": float(x), "y": float(y)} for x, y in points]
        out = canvas(ctx, self.draw_on)
        for x, y in points:
            cv2.circle(out, (int(x), int(y)), 3, CORNER_COLOR, 1)
        ctx.image = out


@register("stage", "harris")
class Harris(_Corners):
    """Harris corner response, thresholded and capped at the strongest ``max``."""

    def __init__(
        self, k: float = 0.04, quality: float = 0.01, max: int = 200, draw_on: str = "source"
    ):
        self.k, self.quality, self.max = k, quality, max
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "harris")
        self._publish(ctx, harris_corners(ctx.image, self.k, self.quality, self.max))


@register("stage", "shi_tomasi")
class ShiTomasi(_Corners):
    """Shi-Tomasi "good features to track" corners."""

    def __init__(
        self,
        max: int = 200,
        quality: float = 0.01,
        min_dist: int = 10,
        draw_on: str = "source",
    ):
        self.max, self.quality, self.min_dist = max, quality, min_dist
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "shi_tomasi")
        self._publish(ctx, shi_tomasi_corners(ctx.image, self.max, self.quality, self.min_dist))


# --------------------------------------------------------------------------- #
# Contours and blobs
# --------------------------------------------------------------------------- #


@register("stage", "contours")
class Contours:
    """Find contours above ``min_area`` and draw them.

    Args:
        min_area: Contours smaller than this are dropped.
        mode: ``external`` (top-level only), ``list`` (all, flat) or ``tree``
            (all, with the ``parent`` link in each row).
        color: BGR triple for the outline.
        thickness: Outline thickness in pixels.
        boxes: Also draw each contour's axis-aligned bounding box.
        draw_on: ``source`` overlays the original colour frame (the usual choice);
            ``image`` draws on the current working frame, promoted to BGR.

    Publishes the kept contours as ``ctx.store["contours"]`` and a filled binary
    mask of them as ``ctx.store["contour_mask"]``, which a sink can take with
    ``input: contour_mask``.
    """

    def __init__(
        self,
        min_area: float = 20.0,
        mode: str = "external",
        color: tuple[int, int, int] = (161, 177, 247),
        thickness: int = 2,
        boxes: bool = False,
        draw_on: str = "source",
    ):
        self.min_area = min_area
        self.mode = mode
        self.color = tuple(int(c) for c in color)
        self.thickness = thickness
        self.boxes = boxes
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "contours")
        _, binary = cv2.threshold(ctx.image, 0, 255, cv2.THRESH_BINARY)
        kept = find_contours(binary, self.mode, self.min_area)
        shapes = [c for c, _ in kept]

        ctx.store["contours"] = shapes
        ctx.metrics["contours"] = len(kept)
        ctx.metrics["contour_area"] = int(sum(cv2.contourArea(c) for c in shapes))
        ctx.rows["contours"] = [
            {
                "area": cv2.contourArea(c),
                "perimeter": cv2.arcLength(c, True),
                "x": int(rect[0]),
                "y": int(rect[1]),
                "w": int(rect[2]),
                "h": int(rect[3]),
                "parent": parent,
            }
            for c, parent in kept
            for rect in [cv2.boundingRect(c)]
        ]

        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, shapes, -1, 255, cv2.FILLED)
        ctx.store["contour_mask"] = mask

        out = canvas(ctx, self.draw_on)
        cv2.drawContours(out, shapes, -1, self.color, self.thickness)
        if self.boxes:
            for c in shapes:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 1)
        ctx.image = out


@register("stage", "bounding_boxes")
class BoundingBoxes:
    """Draw axis-aligned boxes around contours above ``min_area``.

    Reuses ``ctx.store["contours"]`` when a ``contours`` stage already ran, so the
    same shapes are not found twice; otherwise it finds them itself.
    """

    def __init__(
        self,
        min_area: float = 20.0,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_on: str = "source",
    ):
        self.min_area = min_area
        self.color = tuple(int(c) for c in color)
        self.thickness = thickness
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        found = ctx.store.get("contours")
        if found is None:
            require_gray(ctx.image, "bounding_boxes")
            _, binary = cv2.threshold(ctx.image, 0, 255, cv2.THRESH_BINARY)
            found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in found if cv2.contourArea(c) >= self.min_area]

        ctx.store["boxes"] = boxes
        ctx.metrics["boxes"] = len(boxes)
        ctx.rows["boxes"] = [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in boxes
        ]
        out = canvas(ctx, self.draw_on)
        for x, y, w, h in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), self.color, self.thickness)
        ctx.image = out


@register("stage", "blobs")
class Blobs:
    """SimpleBlobDetector with the area, circularity and convexity filters.

    ``0`` disables the circularity or convexity filter. ``dark`` looks for dark
    blobs on a light background, which is OpenCV's own default.
    """

    def __init__(
        self,
        min_area: float = 50.0,
        max_area: float = 5000.0,
        circularity: float = 0.0,
        convexity: float = 0.0,
        dark: bool = True,
        draw_on: str = "source",
    ):
        self._detector = blob_detector(min_area, max_area, circularity, convexity, dark)
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        found = self._detector.detect(to_gray(ctx.image))
        ctx.metrics["blobs"] = len(found)
        ctx.rows["blobs"] = [
            {"x": kp.pt[0], "y": kp.pt[1], "diameter": kp.size, "response": kp.response}
            for kp in found
        ]
        out = canvas(ctx, self.draw_on)
        cv2.drawKeypoints(
            out, found, out, BLOB_COLOR, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        ctx.image = out
