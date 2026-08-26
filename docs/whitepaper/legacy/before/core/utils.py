import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.config import Config


def read_frames(video_path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            yield frame
    finally:
        cap.release()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", nargs="?", default="inputs/gasvid.mp4", type=Path
    )
    parser.add_argument(
        "--output", nargs="?", default="outputs/output.mp4", type=Path
    )
    parser.add_argument(
        "--headless", action="store_true", help="run without a display window"
    )
    args = parser.parse_args()
    return Config(
        video_path=args.video, output_path=args.output, headless=args.headless
    )
