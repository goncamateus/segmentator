"""Tests for :class:`~segmentator.gui.edit_ops_controller.EditOpsController`.

Qt-free: the controller operates on a :class:`DocumentController` alone, no
``QListWidget`` involved — a caller (:class:`~segmentator.gui.window.MainWindow`)
supplies whatever row is currently selected on the widget.
"""

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from segmentator.gui.document_controller import DocumentController
from segmentator.gui.edit_ops_controller import EditOpsController


def test_add_inserts_after_the_selected_row():
    document = DocumentController(CommentedMap(stages=[{"type": "gray"}, {"type": "canny"}]))
    controller = EditOpsController(document)

    at = controller.add("stage", "median_blur", 0)

    assert at == 1
    assert [e["type"] for e in document.specs("stage")] == ["gray", "median_blur", "canny"]


def test_add_with_nothing_selected_appends_at_the_end():
    document = DocumentController(CommentedMap(stages=[{"type": "gray"}]))
    controller = EditOpsController(document)

    at = controller.add("stage", "median_blur", -1)

    assert at == 1
    assert [e["type"] for e in document.specs("stage")] == ["gray", "median_blur"]


def test_remove_deletes_and_returns_the_row_that_slid_into_place():
    document = DocumentController(
        CommentedMap(stages=[{"type": "gray"}, {"type": "median_blur"}, {"type": "threshold"}])
    )
    controller = EditOpsController(document)

    next_row = controller.remove("stage", 1)

    assert [e["type"] for e in document.specs("stage")] == ["gray", "threshold"]
    assert next_row == 1  # threshold, which slid up into row 1


def test_remove_the_last_row_selects_the_new_last_row():
    document = DocumentController(CommentedMap(stages=[{"type": "gray"}, {"type": "median_blur"}]))
    controller = EditOpsController(document)

    assert controller.remove("stage", 1) == 0


def test_remove_the_only_row_selects_nothing():
    document = DocumentController(CommentedMap(stages=[{"type": "gray"}]))
    controller = EditOpsController(document)

    next_row = controller.remove("stage", 0)

    assert next_row == -1
    assert document.specs("stage") == []


def test_shift_moves_an_entry_and_returns_its_new_position():
    # A real ruamel sequence, as `move` (comment-preserving) requires — see
    # tests/test_document_controller.py::test_move_reorders_the_sequence.
    stages = CommentedSeq([{"type": "gray"}, {"type": "canny"}])
    document = DocumentController(CommentedMap(stages=stages))
    controller = EditOpsController(document)

    target = controller.shift("stage", 0, 1)

    assert target == 1
    assert [e["type"] for e in document.specs("stage")] == ["canny", "gray"]


def test_shift_out_of_range_is_a_no_op():
    document = DocumentController(CommentedMap(stages=[{"type": "gray"}, {"type": "canny"}]))
    controller = EditOpsController(document)

    assert controller.shift("stage", 0, -1) is None  # would go negative
    assert controller.shift("stage", 1, 1) is None  # would run past the end
    assert controller.shift("stage", -1, 1) is None  # nothing selected
    assert [e["type"] for e in document.specs("stage")] == ["gray", "canny"]


def test_move_reorders_for_a_drag_and_drop():
    stages = CommentedSeq([{"type": "gray"}, {"type": "canny"}, {"type": "threshold"}])
    document = DocumentController(CommentedMap(stages=stages))
    controller = EditOpsController(document)

    controller.move("stage", 0, 2)

    assert [e["type"] for e in document.specs("stage")] == ["canny", "threshold", "gray"]


def test_reorder_drag_translates_qts_pre_removal_row():
    # Qt's rowsMoved reports the insert point *before* the source row is
    # removed, so a drop below the source lands one index higher than where
    # the entry actually ends up: dragging row 0 to "row 2" really means the
    # entry ends up at index 1, after canny slides up to fill the gap.
    stages = CommentedSeq([{"type": "gray"}, {"type": "canny"}, {"type": "threshold"}])
    document = DocumentController(CommentedMap(stages=stages))
    controller = EditOpsController(document)

    controller.reorder_drag("stage", 0, 2)

    assert [e["type"] for e in document.specs("stage")] == ["canny", "gray", "threshold"]


def test_reorder_drag_above_the_source_needs_no_adjustment():
    stages = CommentedSeq([{"type": "gray"}, {"type": "canny"}, {"type": "threshold"}])
    document = DocumentController(CommentedMap(stages=stages))
    controller = EditOpsController(document)

    controller.reorder_drag("stage", 2, 0)

    assert [e["type"] for e in document.specs("stage")] == ["threshold", "gray", "canny"]


def test_operates_on_sinks_too():
    document = DocumentController(CommentedMap(sinks=[{"type": "display"}]))
    controller = EditOpsController(document)

    controller.add("sink", "csv", 0)

    assert [e["type"] for e in document.specs("sink")] == ["display", "csv"]
