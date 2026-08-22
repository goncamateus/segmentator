"""Runs the chain optimizer off the GUI thread.

Short-lived, unlike :class:`~segmentator.gui.worker.PreviewWorker`: it starts
when Optimize is picked, emits its findings, and ends. The same lock-free
handover the preview uses applies — the GUI writes plain attributes, the worker
answers with Qt signals.

It opens its **own** source rather than borrowing the preview's. Two readers on
one :class:`cv2.VideoCapture` fight over the capture cursor, and every miss costs
a ``CAP_PROP_POS_FRAMES`` seek — 87 ms against 2 ms for reading the next frame on
a 1080p H.264 file. Decoding the sample twice is far cheaper than that.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from segmentator.optimize import SAMPLE_DEFAULT, Finding, analyse, needs_contiguous, sample_frames
from segmentator.pipeline import build


def plain(cfg: dict[str, Any]) -> dict[str, Any]:
    """A detached copy of the parts the optimizer reads.

    The editor's config is a live ruamel document the GUI thread keeps editing;
    the worker must not read it while that happens, and must not hand ruamel's
    comment-carrying maps to code that will copy them around.
    """
    return {
        "source": dict(cfg["source"]),
        "stages": [dict(entry) for entry in cfg.get("stages", [])],
        "sinks": [dict(entry) for entry in cfg.get("sinks", [])],
    }


class OptimizeWorker(QThread):
    """Sample the source, analyse the chain, report what could come out."""

    progress = pyqtSignal(int, int, str)  # done, total, what is being tried
    done = pyqtSignal(list)  # list[Finding]
    failed = pyqtSignal(str)

    def __init__(self, cfg: dict[str, Any], samples: int = SAMPLE_DEFAULT) -> None:
        super().__init__()
        self.cfg = plain(cfg)
        self.samples = samples
        # Kept, not emitted: the caller needs them to re-check whichever subset of
        # the findings the user actually ticks, and they are ~6 MB of 1080p each.
        self.frames: list[tuple[int, np.ndarray]] = []
        self.findings: list[Finding] = []
        self._running = True

    def cancel(self) -> None:
        """Ask the search to stop at the next candidate. Safe from the GUI thread."""
        self._running = False

    def release(self) -> None:
        """Drop the sampled frames once the caller is finished with them."""
        self.frames = []

    def _tick(self, done: int, total: int, label: str) -> bool:
        self.progress.emit(done, total, label)
        return self._running

    def run(self) -> None:
        stages = self.cfg["stages"]
        if not stages:
            self.done.emit([])
            return
        contiguous = needs_contiguous(stages)
        self.progress.emit(0, 0, "sampling frames…")
        try:
            source = build("source", dict(self.cfg["source"]))
        except Exception as exc:
            self.failed.emit(f"source: {exc}")
            return
        try:
            self.frames = sample_frames(source, self.samples, contiguous=contiguous)
            if not self.frames:
                self.failed.emit("could not read any frames from the source")
                return
            self.findings = analyse(self.cfg, self.frames, progress=self._tick)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        finally:
            source.close()
        self.done.emit(self.findings)
