"""Motion: what changed between frames.

Ported from goncanalyser ``features/motion.py``. Six algorithms, and every one of
them reports its motion the same way — a single-channel 0..255 "heat" image in
``ctx.image`` — so a threshold means the same thing whichever one is selected and
switching algorithms does not mean relearning the parameters.

    mog2 / knn                statistical background model, per pixel
    farneback                 dense optical flow, magnitude per pixel
    lucas_kanade              sparse flow at corners, splatted into an image
    frame_diff / three_frame_diff   plain absolute difference

**Where this differs from goncanalyser, deliberately.** There, each algorithm had
threshold, morphology, contour finding, boxes and speeds welded onto the end of
it. Here every algorithm stops at the heat image and the rest is composed in the
config:

    - {type: farneback}
    - {type: threshold, value: 25, mode: binary}
    - {type: morphology, op: open, ksize: 3}
    - {type: motion_objects, min_area: 50}

which is four lines instead of six hidden parameters, and lets any of the
post-processing stages be swapped for something else.

State across frames lives on the stage instance, which is where the pipeline
contract puts it. goncanalyser's ``MotionState`` also had to defend against
re-analysing the same frame and against seeking backwards; neither can happen
here, because ``Pipeline.run`` is a single forward pass over a monotonic
``ctx.index``. Only the shape guard survives — a resized frame invalidates a
model — and it is the one rule that *does* still fire in a batch run.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from segmentator.ops.common import require_gray, to_gray
from segmentator.ops.motion import FOREGROUND, FLOW_GAIN, blend, sparse_flow, speeds
from segmentator.pipeline import Ctx, StageInfo, register
from segmentator.stages.preprocess import canvas

BOX_COLOR = (0, 200, 255)
LABEL_COLOR = (255, 255, 255)


class _Previous:
    """Two frames of history, reset when the frame shape changes underneath it."""

    def __init__(self) -> None:
        self.prev: np.ndarray | None = None
        self.prev2: np.ndarray | None = None

    def push(self, gray: np.ndarray) -> None:
        if self.prev is not None and self.prev.shape != gray.shape:
            self.prev = self.prev2 = None
        self.prev, self.prev2 = gray, self.prev


# --------------------------------------------------------------------------- #
# Background models
# --------------------------------------------------------------------------- #


@register("stage", "mog2")
class Mog2(StageInfo):
    """OpenCV MOG2 adaptive background subtractor. Outputs a foreground mask.

    Args:
        drop_shadows: Keep only pixels labelled 255. MOG2 labels shadows 127, and
            a shadow is the moving thing's *effect* on the background, not the
            moving thing. Only meaningful with ``detect_shadows: true``.
    """

    STATEFUL = True  # self._subtractor

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        learning_rate: float = -1.0,
        drop_shadows: bool = False,
    ):
        self.learning_rate = learning_rate
        self.drop_shadows = drop_shadows
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )

    def apply(self, ctx: Ctx) -> None:
        fg = self._subtractor.apply(ctx.image, learningRate=self.learning_rate)
        ctx.image = np.where(fg == FOREGROUND, 255, 0).astype(np.uint8) if self.drop_shadows else fg


@register("stage", "knn")
class Knn(StageInfo):
    """OpenCV KNN adaptive background subtractor. Outputs a foreground mask."""

    STATEFUL = True  # self._subtractor

    def __init__(
        self,
        history: int = 500,
        dist2_threshold: float = 400.0,
        detect_shadows: bool = False,
        learning_rate: float = -1.0,
        drop_shadows: bool = False,
    ):
        self.learning_rate = learning_rate
        self.drop_shadows = drop_shadows
        self._subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, dist2Threshold=dist2_threshold, detectShadows=detect_shadows
        )

    def apply(self, ctx: Ctx) -> None:
        fg = self._subtractor.apply(ctx.image, learningRate=self.learning_rate)
        ctx.image = np.where(fg == FOREGROUND, 255, 0).astype(np.uint8) if self.drop_shadows else fg


# --------------------------------------------------------------------------- #
# Differencing
# --------------------------------------------------------------------------- #


@register("stage", "frame_diff")
class FrameDiff(StageInfo):
    """Absolute difference against the frame ``lag`` positions back.

    Emits a zero frame until enough history exists, so the output shape is stable
    from the very first frame.
    """

    STATEFUL = True  # self._history deque

    def __init__(self, lag: int = 1):
        if lag < 1:
            raise ValueError(f"frame_diff lag must be >= 1, got {lag}")
        self._history: deque[np.ndarray] = deque(maxlen=lag)

    def apply(self, ctx: Ctx) -> None:
        current = ctx.image
        previous = self._history[0] if len(self._history) == self._history.maxlen else None
        if previous is not None and previous.shape != current.shape:
            self._history.clear()
            previous = None
        self._history.append(current)
        ctx.image = np.zeros_like(current) if previous is None else cv2.absdiff(current, previous)


@register("stage", "three_frame_diff")
class ThreeFrameDiff(StageInfo):
    """Three-frame differencing: changed *and* still changing.

    A pixel counts only where it differs from the previous frame and the previous
    frame differed from the one before, which is what removes the ghost a plain
    two-frame difference leaves behind a moving object.

    The AND is a per-pixel minimum, not ``bitwise_and`` — these are grey levels,
    not flags, and ANDing the bits of 128 and 64 gives 0 for two large changes.
    """

    STATEFUL = True  # self._history

    def __init__(self) -> None:
        self._history = _Previous()

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "three_frame_diff")
        gray = ctx.image
        prev, prev2 = self._history.prev, self._history.prev2
        self._history.push(gray)
        if prev is None or prev2 is None or prev.shape != gray.shape:
            ctx.image = np.zeros_like(gray)
            return
        ctx.image = cv2.min(cv2.absdiff(prev2, prev), cv2.absdiff(prev, gray))


# --------------------------------------------------------------------------- #
# Optical flow
# --------------------------------------------------------------------------- #


@register("stage", "farneback")
class Farneback(StageInfo):
    """Dense Farneback optical flow, as a 0..255 magnitude image.

    Args:
        gain: Grey levels per pixel-of-flow-per-frame. This is what lets one
            threshold serve both flow and differencing: at the default 32, 8
            px/frame reads as full scale. It is a **calibration constant, not a
            derived one** — a slow plume drifting under 1 px/frame peaks around
            30 at this gain and all but disappears after a morphological open, so
            raise it until the motion you care about uses the range.
    """

    STATEFUL = True  # self._history

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        gain: float = FLOW_GAIN,
    ):
        self.pyr_scale = min(0.9, max(0.1, float(pyr_scale)))
        self.levels = max(1, int(levels))
        self.winsize = max(3, int(winsize))
        self.iterations = max(1, int(iterations))
        self.gain = float(gain)
        self._history = _Previous()

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "farneback")
        gray = ctx.image
        prev = self._history.prev
        self._history.push(gray)
        if prev is None or prev.shape != gray.shape:
            ctx.image = np.zeros_like(gray)
            return
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, self.pyr_scale, self.levels, self.winsize, self.iterations, 5, 1.2, 0
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1]) * self.gain
        ctx.image = np.clip(magnitude, 0, 255).astype(np.uint8)


@register("stage", "lucas_kanade")
class LucasKanade(StageInfo):
    """Sparse Lucas-Kanade flow at tracked corners, splatted into a dense image.

    ``gain`` is the same calibration constant :class:`Farneback` documents.
    """

    STATEFUL = True  # self._history

    def __init__(self, max_points: int = 200, win: int = 15, gain: float = FLOW_GAIN):
        self.max_points = max_points
        self.win = win
        self.gain = float(gain)
        self._history = _Previous()

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "lucas_kanade")
        gray = ctx.image
        prev = self._history.prev
        self._history.push(gray)
        if prev is None or prev.shape != gray.shape:
            ctx.image = np.zeros_like(gray)
            return
        heat = sparse_flow(prev, gray, self.max_points, self.win, self.gain)
        ctx.image = np.clip(heat, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Heat and objects
# --------------------------------------------------------------------------- #


@register("stage", "motion_heat")
class MotionHeat(StageInfo):
    """Accumulate the current heat image over a rolling window.

    An exponential average, not a true N-frame window: one ``accumulateWeighted``
    call, no ring buffer, and the difference is a soft tail rather than a hard
    cutoff.

    ponytail: swap for a deque of frames if a hard cutoff is ever needed.

    Leaves ``ctx.image`` alone and publishes the accumulator as
    ``ctx.store["heat"]`` — a float32 array, so it is not a displayable frame;
    :class:`Heatmap` is what turns it into one.
    """

    STATEFUL = True  # self._heat accumulator
    PUBLISHES = ("heat",)
    WRITES = ("store:heat",)

    def __init__(self, window: int = 20):
        self.window = max(1, int(window))
        self._heat: np.ndarray | None = None

    def apply(self, ctx: Ctx) -> None:
        require_gray(ctx.image, "motion_heat")
        frame = ctx.image.astype(np.float32)
        if self._heat is None or self._heat.shape != frame.shape:
            self._heat = np.zeros(frame.shape, np.float32)
        cv2.accumulateWeighted(frame, self._heat, 1.0 / self.window)
        ctx.store["heat"] = self._heat


@register("stage", "heatmap")
class Heatmap(StageInfo):
    """Paint the accumulated heat over a frame as a JET overlay.

    Reads ``ctx.store["heat"]`` when a ``motion_heat`` stage ran, and falls back to
    the current image, so a chain that wants the instantaneous heat rather than
    the rolling average simply leaves ``motion_heat`` out.
    """

    READS = ("store:heat",)

    def __init__(self, opacity: float = 0.5, threshold: float = 0.05, draw_on: str = "source"):
        self.opacity = opacity
        self.threshold = threshold
        self.draw_on = draw_on

    def apply(self, ctx: Ctx) -> None:
        heat = ctx.store.get("heat")
        if heat is None:
            require_gray(ctx.image, "heatmap")
            heat = ctx.image.astype(np.float32)
        out = canvas(ctx, self.draw_on)
        blend(out, heat, self.opacity, self.threshold)
        ctx.image = out


@register("stage", "motion_objects")
class MotionObjects(StageInfo):
    """Turn a motion mask into objects: boxes, areas and per-frame speeds.

    Expects a binary mask — i.e. a ``threshold`` (and usually a ``morphology``)
    stage between the motion algorithm and this one.

    Args:
        min_area: Blobs smaller than this are noise.
        max_travel: Pixels beyond which two blobs in consecutive frames are not
            taken to be the same thing. A calibration constant, not a derived one.
        boxes: Draw the boxes.
        labels: Write area and speed above each box.
        draw_on: ``source`` or ``image``.
    """

    STATEFUL = True  # self._previous centroids
    WRITES = (
        "store:boxes",
        "metrics:motion_px",
        "metrics:motion_frac",
        "metrics:motion_objects",
        "metrics:motion_speed",
        "rows:motion",
    )

    def __init__(
        self,
        min_area: float = 50.0,
        max_travel: float = 60.0,
        boxes: bool = True,
        labels: bool = False,
        draw_on: str = "source",
    ):
        self.min_area = min_area
        self.max_travel = max_travel
        self.boxes = boxes
        self.labels = labels
        self.draw_on = draw_on
        self._previous: list[tuple[float, float]] = []

    def apply(self, ctx: Ctx) -> None:
        mask = to_gray(ctx.image)
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = [c for c in found if cv2.contourArea(c) >= self.min_area]
        boxes = [cv2.boundingRect(c) for c in kept]
        centres = [(x + w / 2, y + h / 2) for x, y, w, h in boxes]
        moved = speeds(centres, self._previous, self.max_travel)
        self._previous = centres

        moving = int(np.count_nonzero(mask))
        ctx.store["boxes"] = boxes
        ctx.metrics["motion_px"] = moving
        ctx.metrics["motion_frac"] = round(moving / mask.size, 4)
        ctx.metrics["motion_objects"] = len(kept)
        ctx.metrics["motion_speed"] = max(moved, default=0.0)
        ctx.rows["motion"] = [
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": float(cv2.contourArea(c)),
                "speed": speed,
            }
            for c, (x, y, w, h), speed in zip(kept, boxes, moved)
        ]

        if not self.boxes:
            return
        out = canvas(ctx, self.draw_on)
        for (x, y, w, h), speed in zip(boxes, moved):
            cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 1)
            if self.labels:
                cv2.putText(
                    out,
                    f"{w * h}px {speed:g}/f",
                    (x, max(10, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    LABEL_COLOR,
                    1,
                    cv2.LINE_AA,
                )
        ctx.image = out
