"""Turning worker output into paintable state: converted pixmaps and the last
measurement snapshot.

Independent of where a frame comes from — the worker, a test, anything that
hands it a dict of ``QImage`` — and independent of
:class:`~segmentator.gui.playback_controller.PlaybackController`. Scaling a
pixmap into a ``QLabel`` and filling a ``QTableWidget`` from a measurement
stay in :class:`~segmentator.gui.window.MainWindow`, which is the only place
that owns those widgets.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QImage, QPixmap


class PaintController:
    """Owns the converted preview images and the last frame's measurement."""

    def __init__(self) -> None:
        self.images: dict[str, QPixmap] = {}
        self.measured: tuple[int, dict[str, Any], dict[str, list]] = (0, {}, {})

    def receive_images(self, images: dict[str, QImage]) -> None:
        """Replace the held images wholesale — a new frame supersedes the last one entirely."""
        self.images = {key: QPixmap.fromImage(image) for key, image in images.items()}

    def pixmap_for(self, key: str) -> QPixmap | None:
        return self.images.get(key)

    def receive_measurement(self, index: int, metrics: dict[str, Any], rows: dict[str, list]) -> None:
        self.measured = (index, metrics, rows)

    def metric_rows(self) -> list[tuple[str, Any]]:
        """Metrics, then one ``<kind> rows`` count per rows kind — what the metrics table shows."""
        _, metrics, rows = self.measured
        return list(metrics.items()) + [(f"{kind} rows", len(r)) for kind, r in rows.items()]
