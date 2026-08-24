"""Tests for :class:`~segmentator.gui.optimize_controller.OptimizeController`.

Mirrors ``tests/test_playback_controller.py``: the worker's thread is never
started for real — ``run()`` is driven synchronously, as
``tests/test_gui.py``'s own worker tests do — so these exercise everything the
controller does around an already-``run()``-through worker (``recheck``,
``apply``, ``release``), plus ``build`` itself.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402
from ruamel.yaml.comments import CommentedMap, CommentedSeq  # noqa: E402

from segmentator.gui.document_controller import DocumentController  # noqa: E402
from segmentator.gui.optimize_controller import OptimizeController  # noqa: E402
from segmentator.optimize import Finding  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def footage(tmp_path):
    """Eight frames of a square moving right, as a folder source."""
    for index in range(8):
        frame = np.zeros((32, 48, 3), np.uint8)
        cv2.rectangle(frame, (index * 2, 8), (index * 2 + 6, 16), (255, 255, 255), -1)
        cv2.imwrite(str(tmp_path / f"f{index:03d}.png"), frame)
    return {"type": "folder", "path": str(tmp_path)}


def test_build_constructs_a_fresh_worker_over_the_document(app, footage):
    document = DocumentController(CommentedMap(source=footage, stages=[{"type": "gray"}], sinks=[]))
    controller = OptimizeController(document)

    worker = controller.build(samples=8)

    assert controller.worker is worker
    assert worker.cfg["stages"] == [{"type": "gray"}]


def test_release_drops_the_worker_and_its_frames(app, footage):
    document = DocumentController(CommentedMap(source=footage, stages=[{"type": "gray"}], sinks=[]))
    controller = OptimizeController(document)
    worker = controller.build(samples=8)
    worker.run()
    assert worker.frames

    controller.release()

    assert controller.worker is None
    assert worker.frames == []


def test_release_without_a_worker_is_a_no_op(app):
    document = DocumentController(CommentedMap(source={"type": "folder", "path": "."}, stages=[]))
    controller = OptimizeController(document)

    controller.release()  # must not raise
    assert controller.worker is None


def test_recheck_confirms_an_identity_finding(app, footage):
    document = DocumentController(
        CommentedMap(
            source=footage,
            stages=[{"type": "gaussian_blur", "ksize": 1}, {"type": "gray"}],
            sinks=[{"type": "ffmpeg", "path": "out.mp4"}],
        )
    )
    controller = OptimizeController(document)
    controller.build(samples=8)
    controller.worker.run()

    assert controller.recheck([Finding((0,), (), "drop gaussian_blur", "identity")])


def test_recheck_rejects_a_finding_that_changes_the_output(app, footage):
    document = DocumentController(
        CommentedMap(
            source=footage,
            stages=[{"type": "gaussian_blur", "ksize": 1}, {"type": "gray"}],
            sinks=[{"type": "ffmpeg", "path": "out.mp4"}],
        )
    )
    controller = OptimizeController(document)
    controller.build(samples=8)
    controller.worker.run()

    assert not controller.recheck([Finding((1,), (), "drop gray", "sampled")])


def test_apply_deletes_and_returns_the_row_that_slid_up(app):
    stages = CommentedSeq([{"type": "gaussian_blur", "ksize": 1}, {"type": "gray"}, {"type": "threshold"}])
    document = DocumentController(CommentedMap(stages=stages))
    controller = OptimizeController(document)

    row = controller.apply([Finding((0,), (), "drop gaussian_blur", "identity")])

    assert [e["type"] for e in document.specs("stage")] == ["gray", "threshold"]
    assert row == 0


def test_apply_a_fusion_replaces_the_run_in_place(app):
    stages = CommentedSeq([{"type": "gaussian_blur"}, {"type": "gray"}, {"type": "threshold"}])
    document = DocumentController(CommentedMap(stages=stages))
    controller = OptimizeController(document)
    table = list(range(256))

    row = controller.apply([Finding((0, 1), ({"type": "lut", "table": table},), "fuse", "exhaustive")])

    assert [e["type"] for e in document.specs("stage")] == ["lut", "threshold"]
    assert row == 0
