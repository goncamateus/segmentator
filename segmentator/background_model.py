import numpy as np


class BackgroundModel:
    """Mean of the first N masked frames, used as a fixed background to subtract."""

    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self._frames: list[np.ndarray] = []
        self.mean: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def accumulate(self, frame: np.ndarray) -> None:
        if self.ready or len(self._frames) >= self.n_frames:
            return
        self._frames.append(frame)
        if len(self._frames) == self.n_frames:
            self.mean = np.mean(self._frames, axis=0).astype(np.uint8)

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        if not self.ready:
            return frame
        # signed diff avoids uint8 wraparound before clipping
        diff = frame.astype(np.int16) - self.mean.astype(np.int16)
        return np.clip(diff, 0, 255).astype(np.uint8)
