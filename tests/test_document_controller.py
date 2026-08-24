"""Tests for :class:`~segmentator.gui.document_controller.DocumentController`.

Qt-free on purpose: the controller is constructible from nothing but a loaded
config, so every test here builds one directly — no ``QApplication``, no
``MainWindow`` — the same split ``tests/test_gui.py`` already draws between
Qt-free spec logic and Qt-dependent worker/window logic.
"""

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from segmentator.gui import spec as spec_module
from segmentator.gui.document_controller import DocumentController

# --------------------------------------------------------------------------- #
# spec-list CRUD
# --------------------------------------------------------------------------- #


def test_specs_creates_an_empty_list_for_a_document_with_neither_key():
    document = DocumentController(CommentedMap())

    assert document.specs("stage") == []
    assert document.specs("sink") == []
    assert document.cfg["stages"] == []
    assert document.cfg["sinks"] == []


def test_specs_returns_the_live_sequence_already_on_the_document():
    cfg = CommentedMap(stages=[{"type": "gray"}], sinks=[{"type": "display"}])
    document = DocumentController(cfg)

    assert document.specs("stage") == [{"type": "gray"}]
    assert document.specs("sink") == [{"type": "display"}]


def test_insert_adds_an_entry_at_the_given_position():
    document = DocumentController(CommentedMap(stages=[{"type": "gray"}]))

    document.insert("stage", 0, {"type": "median_blur"})

    assert [e["type"] for e in document.specs("stage")] == ["median_blur", "gray"]


def test_delete_removes_the_entry_at_the_given_position():
    document = DocumentController(
        CommentedMap(stages=[{"type": "gray"}, {"type": "median_blur"}, {"type": "threshold"}])
    )

    document.delete("stage", 1)

    assert [e["type"] for e in document.specs("stage")] == ["gray", "threshold"]


def test_move_reorders_the_sequence():
    # A real ruamel sequence (as `spec_module.load` produces), not a plain
    # list: `move` carries a per-item comment table (`.ca`) that only a
    # `CommentedSeq` has.
    stages = CommentedSeq([{"type": "gray"}, {"type": "farneback"}, {"type": "threshold"}])
    document = DocumentController(CommentedMap(stages=stages))

    document.move("stage", 1, 2)

    assert [e["type"] for e in document.specs("stage")] == ["gray", "threshold", "farneback"]


def test_move_keeps_a_standalone_comment_attached_to_the_item_it_precedes(tmp_path):
    """Ported directly from what used to be MainWindow's ``shift`` behaviour."""
    path = tmp_path / "commented.yaml"
    path.write_text(
        "source:\n"
        "  type: folder\n"
        "  path: .\n"
        "\n"
        "stages:\n"
        "  - {type: gray}\n"
        "\n"
        "  # gain is a calibration constant, not a derived one.\n"
        "  - {type: farneback, gain: 160}\n"
        "  - {type: threshold, value: 40}\n",
        encoding="utf-8",
    )
    document = DocumentController(spec_module.load(path))

    document.move("stage", 1, 2)  # farneback, and its comment, move down one

    out = tmp_path / "out.yaml"
    spec_module.save(out, document.cfg)
    lines = [line.strip() for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines.index("# gain is a calibration constant, not a derived one.") == lines.index(
        "- {type: farneback, gain: 160}"
    ) - 1


# --------------------------------------------------------------------------- #
# derived, read-only lookups
# --------------------------------------------------------------------------- #


def test_label_numbers_a_stage_by_its_position():
    document = DocumentController(CommentedMap())

    assert document.label("stage", 0, {"type": "gray"}) == "1.  gray"
    assert document.label("stage", 4, {"type": "canny"}) == "5.  canny"


def test_label_shows_a_sinks_explicit_input():
    document = DocumentController(CommentedMap())

    assert document.label("sink", 0, {"type": "ffmpeg", "input": "mask"}) == "ffmpeg  ← mask"


def test_label_falls_back_to_the_sink_default_when_input_is_missing():
    document = DocumentController(CommentedMap())

    assert document.label("sink", 0, {"type": "csv"}) == "csv  ← rows"
    assert document.label("sink", 0, {"type": "display"}) == "display  ← image"


def test_marks_reports_kind_type_name_and_stateful():
    document = DocumentController(CommentedMap())

    assert document.marks("stage", {"type": "farneback"}) == {
        "kind": "stage",
        "type": "farneback",
        "name": None,
        "stateful": True,  # farneback is in STATEFUL
    }
    assert document.marks("stage", {"type": "threshold", "name": "mask"}) == {
        "kind": "stage",
        "type": "threshold",
        "name": "mask",
        "stateful": False,
    }
    # A sink's `name:` (if any) is never painted as a tap pill.
    assert document.marks("sink", {"type": "ffmpeg", "name": "irrelevant"})["name"] is None


def test_sink_default_matches_each_sink_kinds_own_convention():
    document = DocumentController(CommentedMap())

    assert document.sink_default("csv") == "rows"
    assert document.sink_default("json") == "metrics"
    assert document.sink_default("crops") == "source"
    assert document.sink_default("display") == "image"
    assert document.sink_default("ffmpeg") == "image"


def test_image_keys_includes_names_and_published_keys_above_the_position():
    document = DocumentController(
        CommentedMap(
            stages=[
                {"type": "static_mask", "name": "m"},  # publishes "mask"
                {"type": "gray"},
            ]
        )
    )

    assert document.image_keys(0) == ("image", "source")
    assert document.image_keys(1) == ("image", "source", "m", "mask")
    assert document.image_keys() == ("image", "source", "m", "mask")
