"""Tests for :class:`~segmentator.gui.paint_controller.PaintController`.

Independent of a worker or a window: everything here hands the controller a
synthetic frame or a bare measurement tuple directly.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from segmentator.gui.paint_controller import PaintController  # noqa: E402
from segmentator.gui.worker import to_qimage  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def test_receive_images_converts_every_key_to_a_pixmap(app):
    controller = PaintController()
    frame = np.zeros((10, 12, 3), np.uint8)

    controller.receive_images({"image": to_qimage(frame), "source": to_qimage(frame)})

    assert controller.pixmap_for("image") is not None
    assert not controller.pixmap_for("image").isNull()
    assert controller.pixmap_for("source") is not None


def test_pixmap_for_an_unknown_key_is_none(app):
    controller = PaintController()
    assert controller.pixmap_for("nothing") is None


def test_receive_images_replaces_rather_than_merges(app):
    controller = PaintController()
    frame = np.zeros((5, 5, 3), np.uint8)
    controller.receive_images({"image": to_qimage(frame)})

    controller.receive_images({"source": to_qimage(frame)})

    assert controller.pixmap_for("image") is None
    assert controller.pixmap_for("source") is not None


def test_receive_measurement_stores_the_snapshot(app):
    controller = PaintController()

    controller.receive_measurement(7, {"edge_px": 120}, {"contours": [{}, {}]})

    assert controller.measured == (7, {"edge_px": 120}, {"contours": [{}, {}]})


def test_metric_rows_combines_metrics_and_row_counts(app):
    controller = PaintController()
    controller.receive_measurement(0, {"edge_px": 120, "lbp_entropy": 3.5}, {"contours": [{}, {}, {}]})

    assert controller.metric_rows() == [
        ("edge_px", 120),
        ("lbp_entropy", 3.5),
        ("contours rows", 3),
    ]


def test_metric_rows_starts_empty(app):
    controller = PaintController()
    assert controller.metric_rows() == []
