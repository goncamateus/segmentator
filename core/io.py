"""Pipeline endpoints: the video source and the display / file sinks.

Importing this module registers everything below under ``kind="source"`` and
``kind="sink"``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.pipeline import Ctx, frame_for, register
from core.video_writer import FfmpegWriter


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


@register("sink", "display")
class DisplaySink:
    """Show frames in an OpenCV window; returns ``False`` when the quit key is hit.

    Args:
        input: Which image to show — ``image`` (the chain's output), ``source``, or a
            named stage. See :func:`~core.pipeline.frame_for`.
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

    def write(self, ctx: Ctx) -> bool:
        frame = frame_for(ctx, self.input)
        if self.size is not None:
            frame = cv2.resize(frame, self.size)
        cv2.imshow(self.window, frame)
        return cv2.waitKey(self.delay) & 0xFF != ord(self.quit_key)

    def close(self) -> None:
        cv2.destroyWindow(self.window)


@register("sink", "ffmpeg")
class FfmpegSink:
    """Encode frames to an mp4 via :class:`~core.video_writer.FfmpegWriter`.

    The writer is opened on the first frame, because stages such as ``resize`` or
    ``contours`` can change the shape and channel count of what actually gets
    written. Single-channel frames are promoted to BGR.

    Args:
        path: Output file. Parent directories are created.
        input: Which image to encode — ``image`` (the chain's output), ``source``, or a
            named stage. See :func:`~core.pipeline.frame_for`.
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
