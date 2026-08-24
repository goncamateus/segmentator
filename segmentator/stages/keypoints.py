"""Local feature detectors: SIFT and ORB.

Ported from goncanalyser ``features/keypoints.py``. One stage rather than two,
because the two detectors share every parameter — including ``sensitivity``,
which is normalised 0..1 and mapped onto each detector's own native threshold by
:data:`segmentator.ops.keypoints.SENSITIVITY`, so the knob reads the same way for
both even though SIFT wants ~0.04 where ORB wants ~20.
"""

from __future__ import annotations

import cv2

from segmentator.ops.common import to_gray
from segmentator.ops.keypoints import build, detect
from segmentator.pipeline import Ctx, StageInfo, register
from segmentator.stages.preprocess import canvas

KP_COLOR = (0, 200, 255)


@register("stage", "keypoints")
class Keypoints(StageInfo):
    """Detect and draw keypoints.

    Args:
        detector: ``sift`` or ``orb``.
        max: Cap on keypoints kept, strongest first.
        sensitivity: 0 (strict) .. 1 (permissive), mapped per detector.
        octaves: Scale-pyramid depth.
        edge: Edge-response rejection threshold.
        rich: Draw the scale circle and orientation ray rather than bare dots —
            the whole point of looking at SIFT output.
        draw_on: ``source`` or ``image``.

    Publishes the descriptor matrix as ``ctx.store["descriptors"]``.
    """

    WRITES = ("store:keypoints", "store:descriptors", "metrics:*", "rows:keypoints")

    def __init__(
        self,
        detector: str = "sift",
        max: int = 500,
        sensitivity: float = 0.5,
        octaves: int = 3,
        edge: float = 10.0,
        rich: bool = True,
        draw_on: str = "source",
    ):
        self.max = max
        self.rich = rich
        self.draw_on = draw_on
        self._detector = build(detector, max, sensitivity, octaves, edge)

    def apply(self, ctx: Ctx) -> None:
        found, described = detect(self._detector, to_gray(ctx.image), self.max)

        ctx.store["keypoints"] = found
        ctx.store["descriptors"] = described
        ctx.metrics["keypoints"] = len(found)
        if described is not None:
            ctx.metrics["descriptor"] = f"{described.shape[1]}d"
        ctx.rows["keypoints"] = [
            {
                "x": kp.pt[0],
                "y": kp.pt[1],
                "size": kp.size,
                "angle": kp.angle,
                "response": kp.response,
                "octave": kp.octave,
            }
            for kp in found
        ]

        flags = (
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            if self.rich
            else cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        out = canvas(ctx, self.draw_on)
        cv2.drawKeypoints(out, found, out, KP_COLOR, flags)
        ctx.image = out
