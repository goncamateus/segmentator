from collections import deque

import numpy as np


class BackgroundModel:
    """Mean of the last N masked frames, used as a background to subtract.

    Args:
        n_frames: Window size.
        circular: ``False`` (default) freezes the mean from the first N frames
            forever — a fixed background for a static scene. ``True`` keeps a
            rolling window of the last N frames instead, so the estimate
            follows slow lighting drift.
    """

    def __init__(self, n_frames: int, circular: bool = False):
        self.n_frames = n_frames
        self.circular = circular
        self._frames: deque[np.ndarray] = deque(maxlen=n_frames if circular else None)
        self.mean: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def accumulate(self, frame: np.ndarray) -> None:
        if not self.circular and self.ready:
            return
        self._frames.append(frame)
        if len(self._frames) >= self.n_frames:
            self.mean = np.mean(self._frames, axis=0).astype(np.uint8)
        # ponytail: recomputes the full mean every frame once the circular
        # window is full. Fine at n_frames ~60-90; swap for an incremental
        # running sum (+new/n, -evicted/n) if profiling ever says otherwise.

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        if not self.ready:
            return frame
        # signed diff avoids uint8 wraparound before clipping
        diff = frame.astype(np.int16) - self.mean.astype(np.int16)
        return np.clip(diff, 0, 255).astype(np.uint8)
