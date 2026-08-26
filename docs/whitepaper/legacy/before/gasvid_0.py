"""Gas leak segmentation from video: static-region mask + background subtraction."""

import cv2
import numpy as np

from core.background_model import BackgroundModel
from core.config import Config
from core.preprocessing import preprocess, segment, static_mask
from core.utils import parse_args, read_frames
from core.video_writer import FfmpegWriter


def run(cfg: Config) -> None:
    frames = read_frames(cfg.video_path)
    first_frame = next(frames, None)
    if first_frame is None:
        raise OSError(f"Video has no frames: {cfg.video_path}")

    smoothed = preprocess(first_frame, cfg.median_ksize)
    mask = static_mask(smoothed, cfg.mask_threshold)
    background = BackgroundModel(cfg.background_frames)

    writer = None
    if cfg.headless:
        height, width = first_frame.shape[:2]
        probe = cv2.VideoCapture(str(cfg.video_path))
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        probe.release()
        writer = FfmpegWriter(cfg.output_path, fps, width, height)

    for frame in (first_frame, *frames):
        smoothed = preprocess(frame, cfg.median_ksize)
        masked = np.where(mask == 255, smoothed, 0).astype(np.uint8)
        background.accumulate(masked)

        result = segment(background.subtract(smoothed), mask, cfg)

        if cfg.headless:
            writer.write(result)
        else:
            display = cv2.resize(result, cfg.display_size)
            cv2.imshow("Segmented", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if cfg.headless:
        writer.release()
    else:
        cv2.destroyAllWindows()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
