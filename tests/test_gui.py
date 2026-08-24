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
from segmentator.gui.spec import channel_params, is_stateful, params, publishes, rebuild_params
from segmentator.pipeline import registered

PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

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


def test_is_stateful_reads_the_registry():
    """STATEFUL is gone (issue #15) — is_stateful reads StageInfo off the
    registered class instead of a hand-maintained set here."""
    from segmentator.pipeline import component

    for name in registered("stage"):
        if name.startswith("_"):
            continue
        assert is_stateful(name) == component("stage", name).STATEFUL, name
    assert is_stateful("nonsense") is False


def test_publishes_and_channel_params_read_the_registry():
    """PUBLISHES/CHANNEL_PARAMS are gone too — same deal as is_stateful."""
    from segmentator.pipeline import component

    for name in registered("stage"):
        if name.startswith("_"):
            continue
        assert publishes(name) == component("stage", name).PUBLISHES, name
        assert channel_params(name) == component("stage", name).CHANNEL_PARAMS, name
    assert publishes("nonsense") == ()
    assert channel_params("nonsense") == ()


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


def test_a_spec_that_fails_to_build_does_not_kill_the_loop(app, footage):
    """A stage that raises out of its own constructor — `resize` on the freshly
    added placeholder `size: 0`, the way `new_spec` used to seed it — is a
    *construction*-time failure, unlike `farneback` above which fails to
    *apply*. Before `run()` wrapped its `_sync` call in a `try/except`, this
    exited the loop for good: the thread died silently and the editor looked
    alive but never rendered or accepted an edit again. `run()` is called
    directly here (never `.start()`), so the loop is driven by hand: two
    `msleep` calls — one per failed retry — are enough to prove the `except`
    reports and loops back instead of letting the exception escape.
    """
    worker = PreviewWorker(config(footage, [{"type": "resize", "size": 0}]))
    statuses = []
    worker.status.connect(statuses.append)
    calls = 0

    def fake_sleep(_ms):
        nonlocal calls
        calls += 1
        if calls >= 2:
            worker._running = False

    worker.msleep = fake_sleep
    worker.run()  # exits once fake_sleep stops it — no real thread involved
    worker.source.close()

    assert calls == 2, "the loop must not exit via the exception itself"
    assert statuses and all("TypeError" in s for s in statuses)


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


def test_resetting_a_live_param_to_its_default_does_not_raise(app, footage):
    """_write pops a key when it's set back to its default; _sync must cope with
    that instead of indexing the now-missing key (regression for the crash at
    the original worker.py:201)."""
    stages = [{"type": "gray"}, {"type": "threshold", "value": 90, "name": "mask"}]
    worker = worker_at(app, footage, stages)

    specs = [dict(entry) for entry in worker.specs]
    del specs[1]["value"]  # what _write does when value is reset to the default (127)
    worker.specs = tuple(specs)
    worker._sync(worker.specs, same_frame=False, reset=False)

    assert worker._stages[1].value == 127


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


# --------------------------------------------------------------------------- #
# optimize
# --------------------------------------------------------------------------- #


@pytest.fixture
def cluttered(footage, tmp_path):
    """A config carrying the two things a hand-tuned chain accumulates.

    A `gaussian_blur ksize: 1` that does nothing, and a second `gray` on an image
    that is already single-channel. Commented, so the round trip is testable too.
    """
    path = tmp_path / "cluttered.yaml"
    path.write_text(
        "# Keep me.\n"
        "name: cluttered\n"
        "\n"
        "source:\n"
        "  type: folder\n"
        f"  path: {footage['path']}\n"
        "\n"
        "stages:\n"
        "  - {type: gaussian_blur, ksize: 1}\n"
        "  - {type: gray}\n"
        "  - {type: gray}\n"
        "\n"
        "  # value tuned by eye.\n"
        "  - {type: threshold, value: 40}\n"
        "\n"
        "sinks:\n"
        "  - {type: ffmpeg, path: out.mp4}\n",
        encoding="utf-8",
    )
    return path


def test_the_optimize_worker_finds_the_clutter(app, footage):
    """Driven synchronously — `run()` is a plain method, no event loop needed."""
    from segmentator.gui.optimize_worker import OptimizeWorker

    worker = OptimizeWorker(
        {
            "source": dict(footage),
            "stages": [
                {"type": "gaussian_blur", "ksize": 1},
                {"type": "gray"},
                {"type": "gray"},
                {"type": "threshold", "value": 40},
            ],
            "sinks": [{"type": "ffmpeg", "path": "out.mp4"}],
        },
        samples=6,
    )
    worker.run()

    assert worker.findings, "the identity blur alone should have been found"
    assert any(f.basis == "identity" and f.positions == (0,) for f in worker.findings)
    assert worker.frames, "the frames are kept for the caller's combination check"
    worker.release()
    assert worker.frames == []


def test_the_optimize_worker_reports_a_bad_source_instead_of_raising(app):
    from segmentator.gui.optimize_worker import OptimizeWorker

    worker = OptimizeWorker({"source": {"type": "video", "path": "/no/such.mp4"}, "stages": [{"type": "gray"}]})
    seen = []
    worker.failed.connect(seen.append)
    worker.run()

    assert seen and "source" in seen[0]


def test_optimize_does_nothing_with_an_empty_chain(window):
    del window.specs("stage")[:]

    window.optimize()

    assert window.optimizer.worker is None


def test_the_dialog_ticks_deletions_but_not_a_fusion(app):
    """A fusion trades two readable stages for a 256-entry table — the user's call."""
    from segmentator.gui.window import OptimizeDialog
    from segmentator.optimize import Finding

    findings = [
        Finding((0,), (), "drop gaussian_blur", "identity", 1.0, "ksize: 1 is a passthrough"),
        Finding((2, 3), ({"type": "lut"},), "fuse into one lut", "exhaustive", 4.0, "all 256 values"),
        Finding((5,), (), "drop saturation", "sampled", 9.0, "no counterexample in 16 frames"),
    ]
    dialog = OptimizeDialog(None, findings)

    assert [f.label for f in dialog.selected()] == ["drop gaussian_blur", "drop saturation"]
    dialog._boxes[1][0].setChecked(True)
    assert len(dialog.selected()) == 3


def test_applying_a_finding_edits_the_document_and_keeps_the_comments(app, cluttered):
    """The whole point of editing the ruamel sequence row by row."""
    from segmentator.gui.window import MainWindow
    from segmentator.optimize import Finding

    window = MainWindow(cluttered)
    try:
        window.apply_optimizations([Finding((0,), (), "drop gaussian_blur", "identity")])

        assert [entry["type"] for entry in window.cfg["stages"]] == ["gray", "gray", "threshold"]
        window.save()
        text = cluttered.read_text(encoding="utf-8")
        assert "# Keep me." in text
        assert "# value tuned by eye." in text
        assert "gaussian_blur" not in text
    finally:
        window.close()


def test_applying_a_fusion_writes_one_flow_style_line(app, cluttered):
    from segmentator.gui.window import MainWindow
    from segmentator.optimize import Finding

    window = MainWindow(cluttered)
    try:
        table = list(range(256))
        window.apply_optimizations(
            [Finding((1, 2), ({"type": "lut", "table": table},), "fuse", "exhaustive")]
        )

        assert [entry["type"] for entry in window.cfg["stages"]] == ["gaussian_blur", "lut", "threshold"]
        window.save()
        lines = cluttered.read_text(encoding="utf-8").splitlines()
        lut_lines = [line for line in lines if "type: lut" in line]
        assert len(lut_lines) == 1 and lut_lines[0].lstrip().startswith("- {type: lut")
    finally:
        window.close()


# --------------------------------------------------------------------------- #
# AddDialog: the sectioned palette that replaced the one-combo picker
# --------------------------------------------------------------------------- #


def test_the_dialog_is_half_the_main_windows_default_width(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    assert dialog.width() == 640  # half of MainWindow's resize(1280, 760)


def test_every_family_is_a_card_and_covers_every_stage(app, window):
    # spec._demo() is what checks FAMILIES against the live registry (and does
    # so past registrations a shared test run can leave behind); this test only
    # has to check that the dialog renders FAMILIES faithfully.
    from segmentator.gui.spec import FAMILIES
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    assert len(dialog._cards) == len(FAMILIES)
    shown = {
        dialog._lists[i].item(row).text()
        for i, (_, members) in enumerate(FAMILIES)
        for row in range(len(members))
    }
    expected = {name for _, members in FAMILIES for name in members}
    assert shown == expected


def test_sinks_get_one_unsectioned_card(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "sink")
    assert len(dialog._cards) == 1
    assert dialog._lists[0].count() == len(registered("sink"))


def test_a_stateful_stage_carries_its_amber_dot_into_the_dialog(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    for listw in dialog._lists:
        for row in range(listw.count()):
            item = listw.item(row)
            assert marks(item)["stateful"] == is_stateful(item.text())


def test_each_family_wears_its_catalogue_accent(app, window):
    """The four colour groups `docs/assets/stage-families.svg` draws."""
    from segmentator.gui.style import FAMILY_ACCENT, PALETTE
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    for card, (title, _) in zip(dialog._cards, spec_module.FAMILIES):
        accent = PALETTE[FAMILY_ACCENT[title]]
        assert accent in card.styleSheet(), title
        heading = card.layout().itemAt(0).widget()
        assert accent in heading.styleSheet(), title

    # Not all one colour: the grouping is the point.
    assert len({FAMILY_ACCENT[title] for title, _ in spec_module.FAMILIES}) == 4


def test_the_card_carries_the_border_so_its_list_does_not(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    sheet = dialog._cards[0].styleSheet()
    assert "QFrame#card QListWidget { border: 0" in sheet


def test_stage_names_are_monospace(app, window):
    """They are `type:` values to be typed into a config, not prose."""
    from PyQt6.QtGui import QFontInfo
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    assert QFontInfo(dialog._lists[0].font()).fixedPitch()


def test_the_dot_is_explained(app, window):
    from segmentator.gui.window import AddDialog

    from PyQt6.QtWidgets import QLabel

    stages = AddDialog(window, "stage")
    legends = [
        w for w in stages.findChildren(QLabel) if "carries state" in w.text()
    ]
    assert len(legends) == 1

    # Sinks carry no state and get no dots, so the legend would explain nothing.
    sinks = AddDialog(window, "sink")
    assert not [w for w in sinks.findChildren(QLabel) if "carries state" in w.text()]


def test_every_section_header_is_pinned_to_the_same_height(app, window):
    """Otherwise a QGridLayout equalises a row to its tallest card and the
    heading — the one child in a card not already held to a fixed size — is
    what stretches to fill the difference."""
    from segmentator.gui.window import ROW_HEIGHT, AddDialog

    dialog = AddDialog(window, "stage")
    heading_heights = {card.layout().itemAt(0).widget().height() for card in dialog._cards}
    assert heading_heights == {ROW_HEIGHT}


def test_a_card_never_grows_past_three_rows(app, window):
    from segmentator.gui.window import CARD_ROWS, ROW_HEIGHT, AddDialog

    dialog = AddDialog(window, "stage")
    for listw in dialog._lists:
        expected = min(listw.count(), CARD_ROWS) * ROW_HEIGHT + 4
        assert listw.height() == expected


def test_a_family_over_the_cap_scrolls_instead_of_hiding_rows(app, window):
    """The rest of a >3-member family is reachable, not simply cut off."""
    from segmentator.gui.spec import FAMILIES
    from segmentator.gui.window import CARD_ROWS, AddDialog

    dialog = AddDialog(window, "stage")
    index = next(i for i, (_, members) in enumerate(FAMILIES) if len(members) > CARD_ROWS)
    listw = dialog._lists[index]
    assert listw.verticalScrollBar().maximum() > 0
    assert not any(listw.isRowHidden(row) for row in range(listw.count()))


def test_a_family_at_or_under_the_cap_does_not_scroll(app, window):
    from segmentator.gui.spec import FAMILIES
    from segmentator.gui.window import CARD_ROWS, AddDialog

    dialog = AddDialog(window, "stage")
    index = next(i for i, (_, members) in enumerate(FAMILIES) if len(members) <= CARD_ROWS)
    assert dialog._lists[index].verticalScrollBar().maximum() == 0


def test_selecting_one_row_clears_every_other_cards_selection(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    first, second = dialog._lists[0], dialog._lists[1]
    first.setCurrentRow(0)
    dialog._choose(first, first.item(0).text())

    second.setCurrentRow(0)
    dialog._choose(second, second.item(0).text())

    assert first.currentRow() == -1
    assert dialog.chosen == second.item(0).text()
    assert dialog.ok_button.isEnabled()


def test_the_ok_button_starts_disabled_until_something_is_chosen(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    assert not dialog.ok_button.isEnabled()


def test_double_clicking_a_row_accepts_immediately(app, window):
    from PyQt6.QtWidgets import QDialog

    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    dialog._accept_choice("canny")
    assert dialog.chosen == "canny"
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_the_filter_narrows_rows_and_hides_empty_cards(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    dialog.filter.setText("farneback")

    # isVisible() reflects on-screen visibility and this dialog is never shown,
    # so isHidden() is the one that reads the explicit setVisible() call made by
    # _apply_filter.
    visible_cards = [card for card in dialog._cards if not card.isHidden()]
    assert len(visible_cards) == 1
    visible_rows = [
        listw.item(row).text()
        for listw in dialog._lists
        for row in range(listw.count())
        if not listw.isRowHidden(row)
    ]
    assert visible_rows == ["farneback"]


def test_clearing_the_filter_shows_every_card_again(app, window):
    from segmentator.gui.window import AddDialog

    dialog = AddDialog(window, "stage")
    dialog.filter.setText("farneback")
    dialog.filter.setText("")

    assert all(not card.isHidden() for card in dialog._cards)
    assert all(
        not listw.isRowHidden(row)
        for listw in dialog._lists
        for row in range(listw.count())
    )


def test_add_opens_the_dialog_and_inserts_after_the_selection(monkeypatch, window):
    from segmentator.gui.window import AddDialog

    monkeypatch.setattr(AddDialog, "get_type", staticmethod(lambda parent, kind: "median_blur"))
    window.stage_list.setCurrentRow(0)

    window.add("stage")

    assert [entry["type"] for entry in window.cfg["stages"]] == [
        "gray",
        "median_blur",
        "farneback",
        "threshold",
    ]


def test_add_does_nothing_on_cancel(monkeypatch, window):
    from segmentator.gui.window import AddDialog

    monkeypatch.setattr(AddDialog, "get_type", staticmethod(lambda parent, kind: None))
    before = [dict(entry) for entry in window.cfg["stages"]]

    window.add("stage")

    assert list(window.cfg["stages"]) == before


# --------------------------------------------------------------------------- #
# launcher: the window that replaced the startup file picker
# --------------------------------------------------------------------------- #


@pytest.fixture
def clip(tmp_path):
    """A real mp4, because a new project is derived from a video and only a video."""
    path = tmp_path / "a clip: derived.mp4"  # the space and colon are the point
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    if not writer.isOpened():
        pytest.skip("this OpenCV build has no mp4v encoder")
    for index in range(20):
        frame = np.zeros((64, 96, 3), np.uint8)
        cv2.rectangle(frame, (index * 3, 20), (index * 3 + 12, 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_a_new_project_writes_a_config_beside_its_video(clip, monkeypatch, tmp_path):
    from segmentator.gui import launcher

    monkeypatch.chdir(tmp_path)  # no `configs/` here, so it falls back to the video
    written = launcher.starter_config(clip)

    assert written == clip.with_suffix(".yaml")
    cfg = spec_module.load(written)
    # The quoting has to survive a path with a space and a colon in it.
    assert cfg["source"] == {"type": "video", "path": str(clip)}
    assert cfg["name"] == clip.stem


def test_a_new_project_prefers_a_configs_directory_when_there_is_one(clip, monkeypatch, tmp_path):
    from segmentator.gui import launcher

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()

    assert launcher.starter_config(clip).parent == tmp_path / "configs"


def test_a_new_project_never_overwrites_an_existing_config(clip, monkeypatch, tmp_path):
    from segmentator.gui import launcher

    monkeypatch.chdir(tmp_path)
    first = launcher.starter_config(clip)
    second = launcher.starter_config(clip)

    assert first != second
    assert second.name == f"{clip.stem}-2.yaml"
    assert first.exists()


def test_a_new_config_previews_without_an_edit(app, clip, monkeypatch, tmp_path):
    """The whole claim of the starter template: it runs as written."""
    from segmentator.gui import launcher

    monkeypatch.chdir(tmp_path)
    cfg = spec_module.load(launcher.starter_config(clip))

    worker = PreviewWorker(cfg)
    try:
        worker._index = 0
        worker._seek = None
        worker._sync(worker.specs, same_frame=False, reset=True)
        assert worker._render(0) is not None
    finally:
        worker.stop()


def test_the_launcher_offers_new_and_open(app):
    from segmentator.gui.launcher import LauncherWindow

    launcher_window = LauncherWindow()
    try:
        buttons = {b.objectName(): b.text() for b in launcher_window.findChildren(QPushButton)}
        assert buttons == {"new": "New Project", "open": "Open Project"}
    finally:
        launcher_window.close()


def test_the_launcher_shows_the_packaged_version(app):
    from segmentator.gui.launcher import LauncherWindow, app_version

    launcher_window = LauncherWindow()
    try:
        assert launcher_window.version_label.text() == f"version: {app_version()}"
        assert app_version()  # the tests run against an installed package
    finally:
        launcher_window.close()


def drive(monkeypatch, presses, files):
    """Run `choose` against a scripted launcher and a scripted file dialog.

    `presses` is what the user clicks each time round, `files` what they pick.
    Both are popped, so running out of either is the test's own bug, not a hang.
    """
    from PyQt6.QtWidgets import QDialog, QFileDialog

    from segmentator.gui import launcher

    presses, files = list(presses), list(files)

    def exec_(self):
        choice = presses.pop(0)
        if choice is None:
            return QDialog.DialogCode.Rejected
        self.picked = choice
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(launcher.LauncherWindow, "exec", exec_)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (files.pop(0), "")),
    )
    return launcher.choose(), presses, files


def test_choosing_new_returns_a_config_written_from_the_video(app, clip, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    chosen, _, _ = drive(monkeypatch, presses=["new"], files=[str(clip)])

    assert chosen == clip.with_suffix(".yaml")
    assert spec_module.load(chosen)["source"]["path"] == str(clip)


def test_cancelling_the_video_dialog_comes_back_to_the_launcher(app, clip, monkeypatch, tmp_path):
    """Backing out is "I did not mean that", not "quit the application"."""
    monkeypatch.chdir(tmp_path)

    # Press New, cancel the picker, press New again, then pick.
    chosen, presses, files = drive(monkeypatch, presses=["new", "new"], files=["", str(clip)])

    assert chosen == clip.with_suffix(".yaml")
    assert not presses and not files  # both rounds were actually run


def test_closing_the_launcher_opens_nothing(app, monkeypatch):
    chosen, _, _ = drive(monkeypatch, presses=[None], files=[])

    assert chosen is None


def test_the_icon_ships_with_the_package():
    """It moved out of `packaging/` so the launcher could read it at runtime."""
    from PyQt6.QtGui import QPixmap

    from segmentator.gui.launcher import icon_path

    assert icon_path().is_file()
    assert not QPixmap(str(icon_path())).isNull()
