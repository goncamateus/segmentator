"""Tests for the config editor.

Two halves, and the split is deliberate: :mod:`segmentator.gui.spec` has no Qt in
it and is tested as plain Python, while the worker needs a ``QApplication`` and
runs under the offscreen platform. Nothing here opens a window, and nothing here
reads ``inputs/`` — a fresh clone has neither a display nor the footage.

    QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

from segmentator.gui import spec as spec_module
from segmentator.gui.spec import STATEFUL, params, rebuild_params
from segmentator.pipeline import registered

PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from segmentator.gui.worker import WARMUP, PreviewWorker, preview_key, to_qimage  # noqa: E402


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    from PyQt6.QtCore import QSettings

    from segmentator.gui import style

    # The window remembers its theme in QSettings; a test run must not rewrite
    # the theme the person running it left the editor in. Both formats, because
    # the default on Unix is NativeFormat and redirecting IniFormat alone still
    # writes to the real ~/.config.
    root = str(tmp_path_factory.mktemp("settings"))
    for fmt in (QSettings.Format.IniFormat, QSettings.Format.NativeFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, root)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    made = QApplication.instance() or QApplication([])
    style.apply(made)  # the styled path is the only path; test what ships
    return made


@pytest.fixture
def footage(tmp_path):
    """Twenty frames of a square moving right, as a folder source."""
    for index in range(20):
        frame = np.zeros((64, 96, 3), np.uint8)
        cv2.rectangle(frame, (index * 3, 20), (index * 3 + 12, 40), (255, 255, 255), -1)
        cv2.imwrite(str(tmp_path / f"f{index:03d}.png"), frame)
    return {"type": "folder", "path": str(tmp_path)}


def config(footage, stages):
    return {"source": dict(footage), "stages": list(stages)}


def worker_at(app, footage, stages, index=5):
    """A worker parked on one frame with its chain built, without a running loop."""
    worker = PreviewWorker(config(footage, stages))
    worker._index = index
    worker._seek = None
    worker._sync(worker.specs, same_frame=False, reset=True)
    worker._render(0)
    return worker


def edit(worker, position, **changes):
    """Apply an edit the way the window does — a fresh snapshot, never in place."""
    specs = [dict(entry) for entry in worker.specs]
    specs[position].update(changes)
    worker.specs = tuple(specs)
    same_frame = worker._cached_index == worker._index
    return worker._sync(worker.specs, same_frame=same_frame, reset=False)


# --------------------------------------------------------------------------- #
# spec: the form, the classification, the YAML
# --------------------------------------------------------------------------- #


def test_spec_demo():
    spec_module._demo()


def test_every_stateful_name_is_a_real_stage():
    assert STATEFUL <= set(registered("stage"))


def test_a_stage_without_an_init_has_no_parameters():
    assert params("stage", "gray") == []


@pytest.mark.parametrize(
    "type_name,expected",
    [
        ("farneback", set()),  # every knob is a live attribute
        ("mog2", {"history", "var_threshold", "detect_shadows"}),  # baked into the subtractor
        ("frame_diff", {"lag"}),  # baked into the deque's maxlen
        ("static_mask", {"threshold", "invert"}),  # RECONSTRUCT: the mask is cached
        ("threshold", set()),
    ],
)
def test_rebuild_parameters_are_the_ones_not_on_the_instance(type_name, expected):
    assert rebuild_params("stage", {"type": type_name}) == expected


def test_a_half_typed_spec_does_not_raise():
    """The editor is being typed into; half a config is not an error worth a dialog."""
    assert rebuild_params("stage", {"type": "nope"}) == frozenset()
    # colorspace validates `to` in its constructor, so this one genuinely fails to
    # build and every parameter is reported as needing a rebuild.
    assert rebuild_params("stage", {"type": "colorspace", "to": "nope"}) == {"to"}


def test_saving_a_commented_config_changes_nothing(tmp_path):
    for config_path in sorted(__import__("pathlib").Path("configs").glob("*.yaml")):
        out = tmp_path / config_path.name
        spec_module.save(out, spec_module.load(config_path))
        assert out.read_text() == config_path.read_text(), config_path


# --------------------------------------------------------------------------- #
# worker: the prefix cache and the three rules
# --------------------------------------------------------------------------- #


def test_every_stage_is_previewable_named_or_not(app, footage):
    stages = [{"type": "gray"}, {"type": "canny", "name": "edges"}]
    worker = worker_at(app, footage, stages)
    ctx = worker._render(0)
    assert preview_key(0) in ctx.taps and preview_key(1) in ctx.taps
    assert "edges" in ctx.taps
    assert to_qimage(ctx.taps["edges"]).width() == 96


def test_the_cache_holds_one_context_per_stage(app, footage):
    worker = worker_at(app, footage, [{"type": "gray"}, {"type": "canny"}, {"type": "gray"}])
    assert len(worker._cache) == 3


def test_an_edit_only_re_runs_the_stages_below_it(app, footage):
    stages = [{"type": "gray"}, {"type": "farneback"}, {"type": "threshold", "value": 20}]
    worker = worker_at(app, footage, stages)
    before = list(worker._stages)

    start = edit(worker, 2, value=40)

    assert start == 2, "the threshold edit must not invalidate the flow above it"
    assert worker._stages[0] is before[0] and worker._stages[1] is before[1]
    assert len(worker._cache) == 2, "only the stages above the edit stay cached"


def test_a_live_knob_is_assigned_not_rebuilt(app, footage):
    worker = worker_at(app, footage, [{"type": "gray"}, {"type": "threshold", "value": 20}])
    before = worker._stages[1]

    edit(worker, 1, value=40)

    assert worker._stages[1] is before, "value is an attribute; nothing needs rebuilding"
    assert before.value == 40


def test_playing_forward_tunes_a_model_without_resetting_it(app, footage):
    """The price of rule 1 is only paid while paused. Playing, a knob stays live."""
    worker = worker_at(app, footage, [{"type": "gray"}, {"type": "motion_heat"}])
    heat = worker._stages[1]
    accumulated = heat._heat.copy()

    worker._index += 1  # what the run loop does when playing
    worker._cached_index = None
    edit(worker, 1, window=5)

    assert worker._stages[1] is heat, "a new frame is not a frame the stage has seen"
    assert heat.window == 5
    assert np.array_equal(heat._heat, accumulated), "the accumulator survived the edit"


def test_tuning_a_model_while_paused_costs_its_history(app, footage):
    """Rule 1 has a price and it is this one: the accumulator starts over."""
    worker = worker_at(app, footage, [{"type": "gray"}, {"type": "motion_heat"}])
    before = worker._stages[1]

    edit(worker, 1, window=5)

    assert worker._stages[1] is not before
    assert worker._stages[1].window == 5


def test_a_construction_knob_rebuilds_the_stage(app, footage):
    worker = worker_at(app, footage, [{"type": "gray"}, {"type": "mog2"}])
    before = worker._stages[1]

    edit(worker, 1, history=100)

    assert worker._stages[1] is not before, "history is baked into the subtractor"


def test_a_stateful_stage_is_never_shown_the_same_frame_twice(app, footage):
    """Rule 1: re-rendering a paused frame must not advance a model."""
    stages = [{"type": "gray"}, {"type": "farneback", "gain": 32.0}]
    worker = worker_at(app, footage, stages)
    before = worker._stages[1]

    edit(worker, 1, gain=160.0)  # a live knob, but on a stage that remembers frames

    assert worker._stages[1] is not before, "rebuilt rather than fed the frame again"
    assert worker._stages[1].gain == 160.0


def test_a_rebuilt_flow_stage_is_warmed_up_rather_than_left_blank(app, footage):
    """The warm-up replay: a reset stage still has a previous frame to work from."""
    stages = [{"type": "gray"}, {"type": "farneback", "gain": 160.0}]
    worker = worker_at(app, footage, stages)

    edit(worker, 1, gain=200.0)
    ctx = worker._render(0)

    assert worker._stages[1]._history.prev is not None
    assert ctx.image.max() > 0, "a warmed flow stage sees the square move"


def test_a_seek_resets_what_a_jump_invalidates(app, footage):
    """Rule 2: the previous frame is no longer the previous frame."""
    stages = [{"type": "gray"}, {"type": "frame_diff"}]
    worker = worker_at(app, footage, stages, index=10)
    before = worker._stages[1]

    worker._index = 2
    worker._cached_index = None
    worker._sync(worker.specs, same_frame=False, reset=True)

    assert worker._stages[1] is not before


def test_the_warm_up_stops_at_the_last_stage_that_remembers(app, footage):
    """Replaying a HOG below the model would be pure cost."""
    stages = [{"type": "gray"}, {"type": "farneback"}, {"type": "threshold", "value": 10}]
    worker = worker_at(app, footage, stages)
    counted = []
    original = worker._stages[2].apply
    worker._stages[2].apply = lambda ctx: (counted.append(1), original(ctx))[1]

    edit(worker, 1, gain=200.0)

    assert counted == [], "the threshold is a pure function; it has nothing to warm"
    assert WARMUP > 0


def test_a_broken_chain_reports_instead_of_dying(app, footage):
    """farneback on a colour frame raises; the editor has to survive being typed into."""
    worker = PreviewWorker(config(footage, [{"type": "farneback"}]))
    worker._index = 3
    worker._sync(worker.specs, same_frame=False, reset=True)
    with pytest.raises(ValueError):
        worker._render(0)
    worker.source.close()


def test_sink_inputs_resolve_exactly_as_they_do_in_a_batch_run(app, footage):
    from segmentator.pipeline import frame_for

    stages = [{"type": "gray"}, {"type": "static_mask", "threshold": 100}, {"type": "canny"}]
    worker = worker_at(app, footage, stages)
    ctx = worker._render(0)

    assert frame_for(ctx, "mask").shape == (64, 96)
    assert frame_for(ctx, "source").ndim == 3
    with pytest.raises(KeyError):
        frame_for(ctx, "not_a_tap")


# --------------------------------------------------------------------------- #
# window: the operations the editor offers
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(footage, tmp_path):
    """A commented config, the shape the repo's own ones have."""
    path = tmp_path / "edited.yaml"
    path.write_text(
        "# A commented config, which a save has to give back unchanged.\n"
        "name: test\n"
        "\n"
        "source:\n"
        "  type: folder\n"
        f"  path: {footage['path']}\n"
        "\n"
        "stages:\n"
        "  - {type: gray}\n"
        "\n"
        "  # gain is a calibration constant, not a derived one.\n"
        "  - {type: farneback, gain: 160}\n"
        "  - {type: threshold, value: 40, name: mask}\n"
        "\n"
        "sinks:\n"
        "  - {type: ffmpeg, path: out.mp4, input: mask}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def window(app, project):
    from segmentator.gui.window import MainWindow

    made = MainWindow(project)
    yield made
    made.close()


def marks(item):
    from PyQt6.QtCore import Qt

    return item.data(Qt.ItemDataRole.UserRole)


def test_the_window_lists_what_the_config_holds(window):
    assert window.stage_list.count() == 3
    assert window.sink_list.count() == 1
    assert window.stage_list.item(1).text() == "2.  farneback"
    assert window.sink_list.item(0).text() == "ffmpeg  ← mask"


def test_a_row_carries_its_decorations_as_data_not_as_glyphs(window):
    """The tap pill and the state dot are painted by RowDelegate, not typed."""
    assert marks(window.stage_list.item(1)) == {
        "kind": "stage",
        "type": "farneback",
        "name": None,
        "stateful": True,
    }
    assert marks(window.stage_list.item(2)) == {
        "kind": "stage",
        "type": "threshold",
        "name": "mask",
        "stateful": False,
    }
    assert "⟨" not in window.stage_list.item(2).text()


def test_the_two_themes_define_the_same_roles(app):
    from segmentator.gui import style

    style._demo()


def test_toggling_the_theme_repaints_what_was_already_drawn(app, window):
    from segmentator.gui.style import DARK, LIGHT, PALETTE

    window.set_theme("light")  # explicit: the theme is remembered between runs
    window.on_measured(7, {"motion_px": 12}, {})
    assert window.theme_button.text() == "☀"
    assert window.metrics.item(0, 1).foreground().color().name() == LIGHT["ink"]

    window.toggle_theme()

    assert window.theme == "dark"
    assert window.theme_button.text() == "☾"
    assert PALETTE["panel"] == DARK["panel"], "the live palette the delegate paints from"
    assert window.metrics.item(0, 1).foreground().color().name() == DARK["ink"], (
        "cells written before the switch have to be repainted, not left behind"
    )

    window.toggle_theme()
    assert window.theme == "light" and PALETTE["ink"] == LIGHT["ink"]


def test_an_unknown_theme_says_what_it_knows(app):
    from segmentator.gui import style

    with pytest.raises(KeyError, match="dracula"):
        style.apply(app, "dracula")


def test_a_construction_parameter_tints_its_field(window):
    """The amber row in the figure: label and field carry the same warning."""
    window.stage_list.setCurrentRow(1)  # farneback — every knob is live
    fields = [
        window.form._layout.itemAt(row, window.form._layout.ItemRole.FieldRole)
        for row in range(window.form._layout.rowCount())
    ]
    live = [f.widget() for f in fields if f is not None]
    assert not any(widget.property("rebuild") for widget in live)

    window.specs("stage").insert(1, spec_module.new_spec("stage", "mog2"))
    window.reload_lists()
    window.select("stage", 1)
    tinted = [
        window.form._layout.itemAt(row, window.form._layout.ItemRole.LabelRole).widget().text()
        for row in range(window.form._layout.rowCount())
        if (item := window.form._layout.itemAt(row, window.form._layout.ItemRole.FieldRole))
        and item.widget().property("rebuild")
    ]
    assert tinted == ["history", "var_threshold", "detect_shadows"]


def test_selecting_a_stage_generates_its_form(window):
    window.stage_list.setCurrentRow(1)
    labels = [
        window.form._layout.itemAt(row, window.form._layout.ItemRole.LabelRole)
        for row in range(window.form._layout.rowCount())
    ]
    shown = [item.widget().text() for item in labels if item is not None]
    assert shown == ["pyr_scale", "levels", "winsize", "iterations", "gain", "name (tap)"]


def test_a_form_edit_lands_in_the_document_and_not_a_copy(window):
    window.stage_list.setCurrentRow(2)
    window.form._write("value", 90)
    assert window.cfg["stages"][2]["value"] == 90


def test_a_default_is_removed_rather_than_written_out(window):
    window.stage_list.setCurrentRow(2)
    window.form._write("value", 127)  # the constructor default
    assert "value" not in window.cfg["stages"][2]


def test_reordering_moves_the_comment_with_the_stage(window, project):
    window.stage_list.setCurrentRow(1)
    window.shift(1)  # farneback, and its calibration comment, move down one
    window.save()
    saved = project.read_text(encoding="utf-8")
    lines = [line.strip() for line in saved.splitlines() if line.strip()]
    assert lines.index("# gain is a calibration constant, not a derived one.") == lines.index(
        "- {type: farneback, gain: 160}"
    ) - 1


def test_saving_gives_back_every_comment(window, project):
    before = project.read_text(encoding="utf-8")
    window.save()
    assert project.read_text(encoding="utf-8") == before


def test_adding_a_stage_puts_it_after_the_selection(window):
    window.stage_list.setCurrentRow(0)
    window.specs("stage").insert(1, spec_module.new_spec("stage", "median_blur"))
    window.reload_lists()
    assert [entry["type"] for entry in window.cfg["stages"]] == [
        "gray",
        "median_blur",
        "farneback",
        "threshold",
    ]


def test_the_preview_tabs_are_the_selection_the_source_and_every_image_sink(window):
    window.stage_list.setCurrentRow(1)
    labels = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    keys = [window.tabs.tabData(index) for index in range(window.tabs.count())]
    assert labels == ["selected: farneback", "source", "ffmpeg ← mask"]
    assert keys == [preview_key(1), "source", "mask"]


def test_the_run_command_is_offered_rather_than_run(window, project):
    window.copy_command()
    assert QApplication.clipboard().text() == f"uv run segmentator {project}"
