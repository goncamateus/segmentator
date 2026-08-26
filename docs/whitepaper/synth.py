"""Synthetic media for the white paper: a plume-like clip and a geometry still.

Every figure and every measurement in the paper is produced from these two
files, so nothing in it depends on footage that cannot be published. The clip
is a *stand-in with plume-like statistics* — advecting puffs, turbulent jitter,
sensor noise over a static scene — not a physical model of gas dispersion.

    uv run python docs/whitepaper/synth.py            # 320x240, 900 frames
    uv run python docs/whitepaper/synth.py --width 1280 --height 720

Geometry is defined in normalised coordinates, so the same scene renders at any
resolution: the resolution sweep in bench.py measures the pipeline, not a
different picture at each size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SEED = 20260826
MEDIA = Path(__file__).parent / "media"

# Fixed scene furniture, in fractions of frame width/height: (x0, y0, x1, y1, level).
# A vessel, two pipe runs and a skid — enough hard geometry that the structure
# family has edges and lines to find, and enough static clutter that a
# background model has something to learn.
#
# The column and the pipe runs sit *above* 127 on purpose: `static_mask` selects
# the dark region and excludes them, so the region of interest here has the same
# shape as the one in the field footage the project started from — hot fixed
# plant in an otherwise cool static scene.
BOXES = (
    (0.05, 0.55, 0.30, 0.95, 96.0),
    (0.62, 0.10, 0.72, 0.95, 168.0),
    (0.36, 0.78, 0.58, 0.95, 96.0),
)
PIPES = ((0.30, 0.72, 0.95, 0.72, 150.0), (0.67, 0.30, 0.95, 0.30, 150.0))
NOZZLE = (0.34, 0.70)  # where the plume is emitted from


def _background(width: int, height: int) -> np.ndarray:
    """The static scene: a vertical thermal gradient plus fixed furniture."""
    ramp = np.linspace(70, 40, height, dtype=np.float32)
    scene = np.repeat(ramp[:, None], width, axis=1)
    for x0, y0, x1, y1, level in BOXES:
        box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
        cv2.rectangle(scene, box[:2], box[2:], level, -1)
    for x0, y0, x1, y1, level in PIPES:
        start = (int(x0 * width), int(y0 * height))
        end = (int(x1 * width), int(y1 * height))
        cv2.line(scene, start, end, level, max(2, height // 60))
    return scene


def _render_puff(scene: np.ndarray, x: float, y: float, sigma: float, amp: float) -> None:
    """Add one Gaussian puff, evaluated only inside its own +/-3 sigma box.

    The bounding box is what keeps the 1280x720 sweep cheap: without it every
    puff costs a full-frame exponential, and the clip takes minutes per size.
    """
    height, width = scene.shape
    reach = int(3 * sigma) + 1
    x0, x1 = max(0, int(x) - reach), min(width, int(x) + reach)
    y0, y1 = max(0, int(y) - reach), min(height, int(y) + reach)
    if x0 >= x1 or y0 >= y1:
        return
    dx = np.arange(x0, x1, dtype=np.float32) - x
    dy = np.arange(y0, y1, dtype=np.float32) - y
    falloff = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2 * sigma**2))
    scene[y0:y1, x0:x1] += amp * falloff


def plume_clip(path: Path, width: int, height: int, frames: int, fps: float) -> Path:
    """Write the plume clip: puffs emitted at the nozzle, drifting up and right."""
    rng = np.random.default_rng(SEED)
    scale = height / 240.0  # puff sizes and speeds are authored at 320x240
    background = _background(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MPEG-4 part 2, as the field footage is
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    puffs: list[list[float]] = []  # [x, y, vx, vy, sigma, amplitude]
    for index in range(frames):
        if index % 3 == 0:
            # Each puff leaves the nozzle with its own velocity. Perturbing the
            # velocity rather than the position is what makes the plume wander:
            # per-frame position noise averages out into a straight beam.
            puffs.append([
                NOZZLE[0] * width,
                NOZZLE[1] * height,
                (1.05 + rng.normal(0, 0.30)) * scale,
                (-0.85 + rng.normal(0, 0.28)) * scale,
                2.5 * scale,
                34.0,
            ])
        for puff in puffs:
            puff[2] += rng.normal(0, 0.07) * scale  # turbulent velocity random walk
            puff[3] += rng.normal(0, 0.07) * scale
            puff[0] += puff[2]
            puff[1] += puff[3]
            puff[4] += 0.30 * scale  # entrainment: puffs grow as they travel
            puff[5] *= 0.977  # and cool
        puffs = [p for p in puffs if p[5] > 1.0]

        frame = background.copy()
        for x, y, _vx, _vy, sigma, amp in puffs:
            _render_puff(frame, x, y, sigma, amp)
        frame += rng.normal(0, 2.5, frame.shape).astype(np.float32)  # sensor noise
        gray = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def geometry_still(path: Path, width: int, height: int) -> Path:
    """Write the still: hard geometry and texture patches, no plume.

    Edges, lines, corners and contours need straight boundaries; HOG and LBP
    need texture with structure in it. The plume clip has neither, which is why
    the structure and texture families get their own scene.
    """
    rng = np.random.default_rng(SEED)
    scene = _background(width, height)

    cv2.circle(scene, (int(0.80 * width), int(0.62 * height)), int(0.09 * height), 130.0, -1)
    cv2.circle(scene, (int(0.16 * width), int(0.22 * height)), int(0.07 * height), 150.0, 2)
    pts = np.array(
        [[0.40, 0.20], [0.55, 0.40], [0.40, 0.58], [0.26, 0.40]], dtype=np.float32
    ) * (width, height)
    cv2.polylines(scene, [pts.astype(np.int32)], True, 140.0, 2)

    # Three texture patches: stripes, checkerboard, and structureless noise —
    # LBP and HOG separate these, a threshold does not.
    yy, xx = np.mgrid[0 : int(0.16 * height), 0 : int(0.16 * width)]
    patches = (
        (0.05, 0.05, 40 * ((xx // 3) % 2)),
        (0.05, 0.24, 40 * (((xx // 4) + (yy // 4)) % 2)),
        (0.05, 0.43, rng.integers(0, 40, xx.shape)),
    )
    for top, left, texture in patches:
        y0, x0 = int(top * height), int(left * width)
        block = scene[y0 : y0 + texture.shape[0], x0 : x0 + texture.shape[1]]
        block += texture[: block.shape[0], : block.shape[1]].astype(np.float32)

    gray = np.clip(scene, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=MEDIA)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.width}x{args.height}"
    clip = plume_clip(args.out / f"plume-{tag}.mp4", args.width, args.height, args.frames, args.fps)
    still = geometry_still(args.out / f"geometry-{tag}.png", args.width, args.height)
    print(f"{clip} ({args.frames} frames @ {args.fps} fps)\n{still}")


if __name__ == "__main__":
    main()
