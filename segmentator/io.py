"""Pipeline endpoints: frame sources, and the display / file / report sinks.

Importing this module registers everything below under ``kind="source"`` and
``kind="sink"``.

Three sources, because "process any input image" has to mean a still and a folder
of stills as well as a video. Extension sets and the sequential fast path are
ported from goncanalyser ``core/source.py``.
"""

from __future__ import annotations

import csv as csv_module
import json as json_module
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from segmentator.pipeline import Ctx, frame_for, register
from segmentator.video_writer import FfmpegWriter

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}


@register("source", "video")
class VideoSource:
    """Frames from a video file, plus the ``fps`` and ``size`` sinks need.

    Holds a single :class:`cv2.VideoCapture` — reading the metadata does not
    require opening the file a second time. Single-pass: iterate once, then close.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"Could not open video: {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self._cap.release()


@register("source", "image")
class ImageSource:
    """A single still image, as a one-frame sequence.

    ``fps`` and ``size`` exist so the same sinks work unchanged; the frame rate is
    meaningless for one frame and is only there to keep ``FfmpegSink`` happy if
    somebody points one at a still.
    """

    def __init__(self, path: str | Path, fps: float = 1.0):
        self.path = Path(path)
        frame = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Could not read image: {self.path}")
        self._frame = frame
        self.fps = fps
        self.size = (frame.shape[1], frame.shape[0])

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self._frame

    def close(self) -> None:
        pass


@register("source", "folder")
class FolderSource:
    """Every image in a directory, in sorted order, as a frame sequence.

    A folder of stills is a video you cannot scrub, so it is the same protocol.
    Sorted rather than in directory order, because ``frame_0001.png`` ..
    ``frame_0100.png`` only means a sequence if it is read as one.

    Args:
        path: Directory to read.
        pattern: Glob applied inside it. The default takes every readable image
            extension; narrow it to pick one series out of a mixed folder.
        fps: Rate handed to a video sink.
    """

    def __init__(self, path: str | Path, pattern: str = "*", fps: float = 30.0):
        self.path = Path(path)
        if not self.path.is_dir():
            raise OSError(f"Not a directory: {self.path}")
        self.frames = sorted(
            p for p in self.path.glob(pattern) if p.suffix.lower() in IMAGE_EXT
        )
        if not self.frames:
            raise OSError(f"No images matching {pattern!r} in {self.path}")
        first = cv2.imread(str(self.frames[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise OSError(f"Could not read image: {self.frames[0]}")
        self.fps = fps
        self.size = (first.shape[1], first.shape[0])

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self.frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"Could not read image: {path}")
            yield frame

    def close(self) -> None:
        pass


@register("sink", "display")
class DisplaySink:
    """Show frames in an OpenCV window; returns ``False`` when the quit key is hit.

    Args:
        input: Which image to show — ``image`` (the chain's output), ``source``, or a
            named stage. See :func:`~segmentator.pipeline.frame_for`.
        window: Window title. Defaults to ``input``, so several display sinks in one
            config open separate windows instead of overwriting each other.
        size: Optional ``[width, height]`` to scale the preview to.
        delay: ``cv2.waitKey`` delay in ms.
        quit_key: Key that stops the run.
    """

    def __init__(
        self,
        input: str = "image",
        window: str | None = None,
        size: tuple[int, int] | None = None,
        delay: int = 1,
        quit_key: str = "q",
    ):
        self.input = input
        self.window = window if window is not None else input
        self.size = None if size is None else (int(size[0]), int(size[1]))
        self.delay = delay
        self.quit_key = quit_key
        self._shown = False

    def write(self, ctx: Ctx) -> bool:
        frame = frame_for(ctx, self.input)
        if self.size is not None:
            frame = cv2.resize(frame, self.size)
        cv2.imshow(self.window, frame)
        self._shown = True
        return cv2.waitKey(self.delay) & 0xFF != ord(self.quit_key)

    def close(self) -> None:
        # Only if a window was actually opened: destroying one that was never
        # created is a null-pointer error out of highgui, and a run that ends
        # before its first frame — an empty source, `--max-frames 0`, a config
        # built and closed without running — is not an error worth crashing on.
        if self._shown:
            cv2.destroyWindow(self.window)
            self._shown = False


@register("sink", "ffmpeg")
class FfmpegSink:
    """Encode frames to an mp4 via :class:`~segmentator.video_writer.FfmpegWriter`.

    The writer is opened on the first frame, because stages such as ``resize`` or
    ``contours`` can change the shape and channel count of what actually gets
    written. Single-channel frames are promoted to BGR.

    Args:
        path: Output file. Parent directories are created.
        input: Which image to encode — ``image`` (the chain's output), ``source``, or a
            named stage. See :func:`~segmentator.pipeline.frame_for`.
        fps: Frame rate. Defaults to the source's rate when the pipeline binds one.
    """

    def __init__(self, path: str | Path, input: str = "image", fps: float | None = None):
        self.path = Path(path)
        self.input = input
        self.fps = fps
        self._writer: FfmpegWriter | None = None

    def bind_source(self, source: object) -> None:
        """Adopt the source frame rate unless the config pinned one."""
        if self.fps is None:
            self.fps = float(getattr(source, "fps", 30.0))

    def write(self, ctx: Ctx) -> bool:
        frame = frame_for(ctx, self.input)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self._writer is None:
            height, width = frame.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = FfmpegWriter(self.path, self.fps or 30.0, width, height)
        self._writer.write(np.ascontiguousarray(frame))
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


@register("sink", "image")
class ImageSink:
    """Write one image file per frame.

    Args:
        path: Output template. ``{index}`` is substituted with the frame number,
            so ``out/frame_{index:05d}.png`` numbers a sequence. A path without
            ``{index}`` is overwritten every frame, which is what you want for a
            single-frame run and almost never otherwise.
        input: Which image to write. See :func:`~segmentator.pipeline.frame_for`.
    """

    def __init__(self, path: str | Path, input: str = "image"):
        self.path = str(path)
        self.input = input

    def write(self, ctx: Ctx) -> bool:
        frame = frame_for(ctx, self.input)
        out = Path(self.path.format(index=ctx.index))
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), frame):
            raise OSError(f"Could not write image: {out}")
        return True

    def close(self) -> None:
        pass


@register("sink", "csv")
class CsvSink:
    """Write ``ctx.rows`` to one CSV per kind, with a ``frame`` column prepended.

    The header is taken from the first row seen for a kind, so a stage that emits
    a different shape mid-run raises rather than writing a ragged file. Files are
    opened lazily — a kind that never produces a row never produces a file.

    Args:
        dir: Directory for the CSVs; created if missing.
        kinds: Restrict to these row kinds. Empty means every kind emitted.
    """

    def __init__(self, dir: str | Path, kinds: tuple[str, ...] = ()):
        self.dir = Path(dir)
        self.kinds = tuple(kinds)
        self._writers: dict[str, tuple] = {}

    def write(self, ctx: Ctx) -> bool:
        for kind, rows in ctx.rows.items():
            if (self.kinds and kind not in self.kinds) or not rows:
                continue
            if kind not in self._writers:
                self.dir.mkdir(parents=True, exist_ok=True)
                handle = open(self.dir / f"{kind}.csv", "w", newline="", encoding="utf-8")
                writer = csv_module.DictWriter(handle, ["frame", *rows[0]])
                writer.writeheader()
                self._writers[kind] = (handle, writer)
            _, writer = self._writers[kind]
            for row in rows:
                writer.writerow({"frame": ctx.index, **row})
        return True

    def close(self) -> None:
        for handle, _ in self._writers.values():
            handle.close()
        self._writers.clear()


@register("sink", "json")
class JsonSink:
    """Write one JSON object of ``ctx.metrics`` per frame, newline-delimited.

    JSON Lines rather than one array: the file is valid and readable after a run
    that was interrupted, which for a long batch is the normal way it ends.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def write(self, ctx: Ctx) -> bool:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "w", encoding="utf-8")
        self._handle.write(json_module.dumps({"frame": ctx.index, **ctx.metrics}) + "\n")
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@register("sink", "crops")
class CropsSink:
    """Cut every row's ``x/y/w/h`` rectangle out of a frame and save it.

    The dataset half of the report: what the CSV describes, this hands you as
    pixels. Rows without a full rectangle are skipped rather than guessed at, so
    pointing this at ``keypoints`` (which has ``x``/``y`` but no size) writes
    nothing instead of one-pixel files.

    Args:
        dir: Directory for the crops; created if missing.
        kind: Which ``ctx.rows`` kind to cut out.
        input: Which image to cut from — usually ``source``.
        pad: Pixels added around each rectangle, clipped to the frame.
    """

    def __init__(
        self, dir: str | Path, kind: str = "contours", input: str = "source", pad: int = 0
    ):
        self.dir = Path(dir)
        self.kind = kind
        self.input = input
        self.pad = int(pad)
        self.count = 0

    def write(self, ctx: Ctx) -> bool:
        rows = ctx.rows.get(self.kind)
        if not rows:
            return True
        frame = frame_for(ctx, self.input)
        height, width = frame.shape[:2]
        for number, row in enumerate(rows):
            if not {"x", "y", "w", "h"} <= row.keys():
                continue
            x0 = max(0, int(row["x"]) - self.pad)
            y0 = max(0, int(row["y"]) - self.pad)
            x1 = min(width, int(row["x"]) + int(row["w"]) + self.pad)
            y1 = min(height, int(row["y"]) + int(row["h"]) + self.pad)
            if x1 <= x0 or y1 <= y0:
                continue
            self.dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.dir / f"{ctx.index:06d}_{number:03d}.png"), frame[y0:y1, x0:x1])
            self.count += 1
        return True

    def close(self) -> None:
        pass
