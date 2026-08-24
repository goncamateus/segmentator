"""Tests for :class:`~segmentator.gui.playback_controller.PlaybackController`.

Built the way ``tests/test_gui.py``'s own worker tests are: the worker's
thread is never started for real here — spinning a real ``QThread`` has no
place in a fast unit test — so these exercise everything the controller does
to an already-``build()``-ed worker (``push``, ``toggle_play``, ``jump``,
``seek``, ``stop``), plus the raising path of ``build`` itself, which never
reaches the thread at all.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from segmentator.gui.document_controller import DocumentController  # noqa: E402
from segmentator.gui.playback_controller import PlaybackController  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def footage(tmp_path):
    for index in range(5):
        cv2.imwrite(str(tmp_path / f"f{index:03d}.png"), np.zeros((32, 48, 3), np.uint8))
    return {"type": "folder", "path": str(tmp_path)}


def test_build_raises_and_leaves_no_worker_on_a_bad_source(app):
    document = DocumentController({"source": {"type": "folder", "path": "/no/such/directory"}, "stages": []})
    controller = PlaybackController(document)

    with pytest.raises(OSError):
        controller.build()
    assert controller.worker is None


def test_build_stops_whatever_worker_was_running_first(app, footage):
    document = DocumentController({"source": footage, "stages": []})
    controller = PlaybackController(document)
    first = controller.build()

    second = controller.build()

    assert second is not first
    assert controller.worker is second


def test_push_hands_the_worker_a_fresh_copy_of_the_specs(app, footage):
    document = DocumentController({"source": footage, "stages": [{"type": "gray"}]})
    controller = PlaybackController(document)
    controller.build()

    document.specs("stage").append({"type": "canny"})
    controller.push()

    assert controller.worker.specs == ({"type": "gray"}, {"type": "canny"})


def test_push_without_a_worker_is_a_no_op(app):
    document = DocumentController({"source": {"type": "folder", "path": "."}, "stages": []})
    controller = PlaybackController(document)

    controller.push()  # must not raise


def test_toggle_play_flips_and_returns_the_new_state(app, footage):
    document = DocumentController({"source": footage, "stages": []})
    controller = PlaybackController(document)
    controller.build()

    assert controller.toggle_play() is True
    assert controller.worker.playing is True
    assert controller.toggle_play() is False
    assert controller.worker.playing is False


def test_toggle_play_without_a_worker_returns_false(app):
    document = DocumentController({"source": {"type": "folder", "path": "."}, "stages": []})
    controller = PlaybackController(document)

    assert controller.toggle_play() is False


def test_jump_steps_the_worker_from_its_current_index(app, footage):
    document = DocumentController({"source": footage, "stages": []})
    controller = PlaybackController(document)
    controller.build()
    controller.worker._index = 2

    controller.jump(-1)

    assert controller.worker._seek == 1


def test_seek_pauses_and_moves_to_an_absolute_index(app, footage):
    document = DocumentController({"source": footage, "stages": []})
    controller = PlaybackController(document)
    controller.build()
    controller.worker.playing = True

    controller.seek(3)

    assert controller.worker.playing is False
    assert controller.worker._seek == 3


def test_set_wanted_tells_the_worker_which_images_to_convert(app, footage):
    document = DocumentController({"source": footage, "stages": []})
    controller = PlaybackController(document)
    controller.build()

    controller.set_wanted(("image", "source"))

    assert controller.worker.wanted == ("image", "source")


def test_set_wanted_without_a_worker_is_a_no_op(app):
    document = DocumentController({"source": {"type": "folder", "path": "."}, "stages": []})
    controller = PlaybackController(document)

    controller.set_wanted(("image",))  # must not raise


def test_stop_clears_the_worker_and_is_safe_to_call_twice(app, footage):
    document = DocumentController({"source": footage, "stages": []})
    controller = PlaybackController(document)
    controller.build()

    controller.stop()
    assert controller.worker is None
    controller.stop()  # a second stop must not raise
