import subprocess
from pathlib import Path

import numpy as np


class FfmpegWriter:
    """Pipes raw BGR frames to ffmpeg's libx264 encoder for a widely-playable mp4.

    cv2.VideoWriter's H.264 fourccs (avc1/h264/X264) resolve to whatever encoder
    ffmpeg registers first for that codec id — on this build that's the hardware
    v4l2m2m encoder, which fails without a device. Shelling out to ffmpeg lets us
    pick libx264 explicitly.
    """

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()
