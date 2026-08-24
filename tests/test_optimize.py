"""Tests for the chain optimizer.

The interesting assertions here are the negative ones. An optimizer that finds
simplifications is easy; one that does not propose a stage which is quietly
load-bearing is the whole job, so most of what follows pins down what must
*not* be reported.
"""

import cv2
import numpy as np
import pytest

from segmentator import optimize
from segmentator.optimize import (
    Finding,
    analyse,
    apply_findings,
    check,
    dead_stages,
    identity_stages,
    observable,
    point_op_runs,
    redundant_grays,
    run_chain,
    structural,
)
from segmentator.pipeline import Ctx, _build_stages


@pytest.fixture
def frames():
    """Eight frames of a square drifting across noise.

    Noise rather than flat colour so a dropped stage has something to change,
    and drifting so a stage that matters on one frame and not another shows up.
    """
    rng = np.random.default_rng(0)
    out = []
    for index in range(8):
        frame = rng.integers(0, 90, (120, 160, 3), dtype=np.uint8)
        left = 10 + index * 12
        cv2.rectangle(frame, (left, 40), (left + 40, 80), (230, 230, 230), -1)
        out.append((index, frame))
    return out


# --------------------------------------------------------------------------- #
# The observable set
# --------------------------------------------------------------------------- #


def test_observable_covers_rows_and_metrics_not_just_the_image():
    """A csv or json sink watches things no image comparison would catch."""
    assert observable([{"type": "display"}]) == {"image:image"}
    assert observable([{"type": "ffmpeg", "path": "x", "input": "mask"}]) == {"image:mask"}
    assert observable([{"type": "json", "path": "x"}]) == {"metrics:*"}
    assert observable([{"type": "csv", "dir": "x"}]) == {"rows:*"}
    assert observable([{"type": "csv", "dir": "x", "kinds": ("contours",)}]) == {"rows:contours"}
    # No sinks at all — the editor previewing — still observes the final image.
    assert observable([]) == {"image:image"}


# --------------------------------------------------------------------------- #
# Structural findings
# --------------------------------------------------------------------------- #


def test_identity_parameters_are_reported():
    specs = [
        {"type": "gaussian_blur", "ksize": 1},
        {"type": "gamma", "value": 1.0},
        {"type": "morphology", "op": "open", "iterations": 0},
        {"type": "gaussian_blur", "ksize": 5},  # does something: must not appear
    ]
    assert [f.positions[0] for f in identity_stages(specs)] == [0, 1, 2]


def test_a_named_identity_stage_is_left_alone():
    """The tap is a name a sink's `input:` may resolve, passthrough or not."""
    assert identity_stages([{"type": "gaussian_blur", "ksize": 1, "name": "keep"}]) == []


def test_apply_mask_from_source_orphans_the_stages_above_it():
    """The edit that makes a chain grow dead weight, and the reason for the walk.

    `apply_mask` pointed at `source` rebuilds the working image from scratch, so
    `canny` above it computes a frame nobody ever looks at. The `threshold` is
    still alive: its tap is what the mask resolves to.
    """
    specs = [
        {"type": "gray"},
        {"type": "threshold", "value": 100, "name": "m"},
        {"type": "canny"},  # nothing reads this
        {"type": "apply_mask", "input": "source", "mask": "m"},
    ]
    assert [f.positions[0] for f in dead_stages(specs, {"image:image"})] == [2]


def test_a_stage_between_is_still_read_and_so_not_dead():
    """`static_mask` fits itself from the working image, so it consumes it."""
    specs = [
        {"type": "gray"},
        {"type": "canny"},
        {"type": "static_mask", "name": "m"},
        {"type": "apply_mask", "input": "source", "mask": "m"},
    ]
    assert dead_stages(specs, {"image:image"}) == []


def test_a_stage_feeding_only_a_json_metric_is_not_dead():
    """`canny` writes nothing but `edge_px`; with a json sink that is output."""
    specs = [
        {"type": "gray"},
        {"type": "canny"},
        {"type": "static_mask", "name": "m"},
        {"type": "apply_mask", "input": "source", "mask": "m"},
    ]
    assert dead_stages(specs, {"metrics:*", "image:image"}) == []


def test_a_stage_whose_tap_is_read_downstream_is_not_dead():
    specs = [
        {"type": "gray"},
        {"type": "threshold", "value": 10, "name": "t"},
        {"type": "apply_mask", "input": "source", "mask": "t"},
    ]
    assert dead_stages(specs, {"image:image"}) == []


def test_a_store_key_read_by_a_later_stage_keeps_it_alive():
    """`contours` publishes `contour_mask`; `apply_mask` names it."""
    specs = [
        {"type": "gray"},
        {"type": "contours"},
        {"type": "apply_mask", "input": "source", "mask": "contour_mask"},
    ]
    assert dead_stages(specs, {"image:image"}) == []


def test_redundant_gray_is_clarity_only():
    found = redundant_grays([{"type": "gray"}, {"type": "gray"}])
    assert [f.positions[0] for f in found] == [1]
    assert found[0].saved_ms == 0.0
    # The first gray on a colour frame is doing real work.
    assert redundant_grays([{"type": "gray"}]) == []


# --------------------------------------------------------------------------- #
# Point-op fusion — the one finding proved over the whole input domain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "run",
    [
        [{"type": "brightness_contrast", "brightness": -79, "contrast": 1.31}, {"type": "gamma", "value": 0.1}],
        # A negative brightness is the case algebra gets wrong: convertScaleAbs is
        # |src*alpha + beta|, so the curve reflects at zero instead of clamping.
        [{"type": "brightness_contrast", "brightness": -120}, {"type": "brightness_contrast", "contrast": 2.0}],
        [{"type": "gamma", "value": 0.4}, {"type": "threshold", "value": 90, "mode": "tozero"}],
    ],
)
def test_a_fused_lut_matches_the_run_it_replaces_on_every_value(run):
    """Exhaustive: all 256 inputs, both channel layouts. This is a proof, not a sample."""
    found = point_op_runs(run)
    assert len(found) == 1 and found[0].positions == (0, 1)

    for probe in (
        np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2),
        np.arange(256, dtype=np.uint8)[None, :].copy(),
    ):
        before, after = Ctx(image=probe, source=probe, index=0), Ctx(image=probe, source=probe, index=0)
        for stage in _build_stages(run):
            stage.apply(before)
        for stage in _build_stages(list(found[0].replacement)):
            stage.apply(after)
        assert np.array_equal(before.image, after.image)


def test_otsu_threshold_is_not_a_point_op():
    """Its level comes from the frame histogram, so no fixed table can stand in."""
    run = [{"type": "gray"}, {"type": "gamma", "value": 0.5}, {"type": "threshold", "otsu": True}]
    assert point_op_runs(run) == []


def test_a_lone_point_op_is_not_worth_fusing():
    assert point_op_runs([{"type": "gamma", "value": 0.5}]) == []


def test_a_named_point_op_breaks_the_run():
    """Fusing would delete a tap something may resolve."""
    run = [{"type": "gamma", "value": 0.5, "name": "keep"}, {"type": "brightness_contrast", "brightness": 5}]
    assert point_op_runs(run) == []


def test_lut_rejects_a_table_that_is_not_256_entries():
    with pytest.raises(ValueError, match="256-entry"):
        _build_stages([{"type": "lut", "table": [0, 1, 2]}])


# --------------------------------------------------------------------------- #
# The sampled oracle
# --------------------------------------------------------------------------- #


def test_check_rejects_a_candidate_that_changes_the_output(frames):
    specs = [{"type": "gray"}, {"type": "gaussian_blur", "ksize": 7}, {"type": "threshold", "value": 100}]
    drop_blur = Finding((1,), (), "drop gaussian_blur", "sampled")
    assert not check(specs, [drop_blur], frames, {"image:image"})


def test_check_rejects_a_candidate_that_will_not_even_build(frames):
    """Deleting a named stage leaves a dangling `mask:`; that is a rejection, not a crash."""
    specs = [
        {"type": "gray"},
        {"type": "threshold", "value": 100, "name": "t"},
        {"type": "apply_mask", "input": "source", "mask": "t"},
    ]
    assert not check(specs, [Finding((1,), (), "drop threshold", "sampled")], frames, {"image:image"})


def test_a_difference_the_sinks_cannot_see_is_not_a_difference(frames):
    """The oracle watches the sinks, not the intermediate taps — that is the point.

    `equalize` changes the image `canny` is handed, but this chain reports only
    `edge_px` to a json sink, and on these frames that count does not move.
    """
    specs = [{"type": "gray"}, {"type": "median_blur", "ksize": 3}, {"type": "canny", "lo": 10, "hi": 250}]
    reference = run_chain(specs, frames, {"metrics:*"})
    assert len(reference) == len(frames)
    # Same chain, same answer — the harness itself must be deterministic.
    assert check(specs, [], frames, {"metrics:*"})


def test_sampling_one_frame_would_have_lied():
    """The false positive that shaped this design, in miniature.

    The first frame is featureless, so the blur changes nothing that reaches the
    sink and dropping it looks free. Every later frame has a noisy square whose
    contour the blur is what holds together. One frame says drop it; eight do not.
    """
    rng = np.random.default_rng(3)
    frames = [(0, np.zeros((120, 160, 3), np.uint8))]
    for index in range(1, 8):
        frame = rng.integers(0, 200, (120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (110, 90), (250, 250, 250), -1)
        frames.append((index, frame))

    specs = [
        {"type": "gray"},
        {"type": "gaussian_blur", "ksize": 9},
        {"type": "threshold", "value": 200},
        {"type": "contours", "min_area": 1.0},
    ]
    drop_blur = Finding((1,), (), "drop gaussian_blur", "sampled")
    assert check(specs, [drop_blur], frames[:1], {"rows:contours"})
    assert not check(specs, [drop_blur], frames, {"rows:contours"})


def test_findings_that_each_hold_alone_can_break_together(frames):
    """Why a caller must re-check the selection instead of trusting the list.

    Both greys can go — individually. The first because the second still converts,
    the second because the first already did. Taking both leaves `threshold`
    looking at a colour frame.
    """
    specs = [{"type": "gray"}, {"type": "gray"}, {"type": "threshold", "value": 120}]
    drops = [Finding((0,), (), "drop gray", "sampled"), Finding((1,), (), "drop gray", "dataflow")]

    assert all(check(specs, [one], frames, {"image:image"}) for one in drops)
    assert not check(specs, drops, frames, {"image:image"})


def test_apply_findings_edits_from_the_bottom_up():
    specs = [{"type": "gray"}, {"type": "gamma"}, {"type": "median_blur"}, {"type": "canny"}]
    findings = [
        Finding((0,), (), "", "sampled"),
        Finding((2, 3), ({"type": "lut"},), "", "exhaustive"),
    ]
    assert [s["type"] for s in apply_findings(specs, findings)] == ["gamma", "lut"]
    assert [s["type"] for s in specs] == ["gray", "gamma", "median_blur", "canny"]  # untouched


def test_analyse_without_frames_returns_only_provable_findings():
    cfg = {
        "stages": [
            {"type": "brightness_contrast", "brightness": 10},
            {"type": "gamma", "value": 0.5},
            {"type": "gaussian_blur", "ksize": 1},
        ],
        "sinks": [{"type": "display"}],
    }
    found = analyse(cfg)
    assert found and all(f.proven for f in found)
    assert {f.basis for f in found} <= {"identity", "dataflow", "exhaustive"}


def test_analyse_with_frames_prices_and_confirms(frames):
    cfg = {
        "stages": [
            {"type": "gaussian_blur", "ksize": 1},  # identity
            {"type": "gray"},
            {"type": "gray"},  # redundant
            {"type": "threshold", "value": 120},
        ],
        "sinks": [{"type": "display"}],
    }
    found = analyse(cfg, frames)
    assert [f.positions[0] for f in found if f.basis == "identity"] == [0]
    assert all(f.saved_ms is not None for f in found)
    # Every finding stands on its own against the frames it was judged on.
    for finding in found:
        assert check(cfg["stages"], [finding], frames, observable(cfg["sinks"]))
    # And a selection of one identity plus one redundant gray really does apply.
    keep = [f for f in found if f.positions[0] in (0, 2)]
    assert check(cfg["stages"], keep, frames, observable(cfg["sinks"]))
    assert [s["type"] for s in apply_findings(cfg["stages"], keep)] == ["gray", "threshold"]


def test_channel_walk_tables_only_name_stages_that_exist():
    """``_MAKES_GRAY``/``_MAKES_BGR`` have no StageInfo equivalent — a stage's
    output channel count isn't part of its self-description — so they stay
    hand-maintained here; this just guards against a typo'd stage name."""
    from segmentator.pipeline import registered

    known = set(registered("stage"))
    assert optimize._MAKES_GRAY <= known and optimize._MAKES_BGR <= known


def test_every_stage_exposes_self_description():
    """Every registered stage carries the class-level metadata from StageInfo.

    Since issue #15, this is the only place these attributes are validated:
    optimize.py and gui/spec.py no longer carry hand-maintained tables of
    their own — both read STATEFUL/PUBLISHES/CHANNEL_PARAMS/READS/WRITES/
    POINT_OP/IDENTITY_PARAMS straight off the registered class. Skips
    leading-underscore registrations the same way gui/spec.py's _demo() does:
    those exist only for a test's own process (tests/test_pipeline.py's
    "_test_tag") and were never meant to describe themselves for the
    registry's consumers.
    """
    from segmentator.pipeline import StageInfo, component, registered

    shipped = [name for name in registered("stage") if not name.startswith("_")]
    assert shipped, "sanity: the registry should not be empty"

    for name in shipped:
        cls = component("stage", name)
        assert issubclass(cls, StageInfo), f"{name}: does not inherit StageInfo"
        assert isinstance(cls.STATEFUL, bool), f"{name}: STATEFUL must be a bool"
        assert isinstance(cls.POINT_OP, bool), f"{name}: POINT_OP must be a bool"
        for attr in ("PUBLISHES", "CHANNEL_PARAMS", "READS", "WRITES"):
            value = getattr(cls, attr)
            assert isinstance(value, tuple), f"{name}: {attr} must be a tuple"
            assert all(isinstance(v, str) for v in value), f"{name}: {attr} entries must be str"
        assert isinstance(cls.IDENTITY_PARAMS, dict), f"{name}: IDENTITY_PARAMS must be a dict"

        # The cross-table invariant the parent issue calls out: a published
        # (image-valued) store key must also be a key this stage says it
        # writes to the store — checked directly on the class, which is the
        # only place either fact is recorded now.
        written_store_keys = {k.removeprefix("store:") for k in cls.WRITES if k.startswith("store:")}
        missing = set(cls.PUBLISHES) - written_store_keys
        assert not missing, f"{name}: {missing} in PUBLISHES but not in WRITES"


def test_stage_info_values_for_representative_stages():
    """Literal spot-checks on a handful of stages covering every kind of
    attribute (stateful, reads, writes, point op, identity) — the sort of
    copy-paste slip that would land a table entry on the wrong stage.
    """
    from segmentator.pipeline import component

    def cls(name):
        return component("stage", name)

    assert cls("mog2").STATEFUL is True
    assert cls("gray").STATEFUL is False

    assert cls("mean_background").READS == ("store:mask",)
    assert cls("paste_roi").READS == ("store:roi",)

    assert cls("contours").WRITES == (
        "store:contours",
        "store:contour_mask",
        "metrics:contours",
        "metrics:contour_area",
        "rows:contours",
    )
    assert cls("harris").WRITES == ("store:corners", "metrics:corners", "rows:corners")
    assert cls("shi_tomasi").WRITES == ("store:corners", "metrics:corners", "rows:corners")

    assert cls("brightness_contrast").POINT_OP is True
    assert cls("gamma").POINT_OP is True
    assert cls("threshold").POINT_OP is True
    assert cls("lut").POINT_OP is False

    assert cls("morphology").IDENTITY_PARAMS == {"iterations": (0,)}
    assert cls("gaussian_blur").IDENTITY_PARAMS == {"ksize": (0, 1)}


def test_optimize_module_has_no_gui_import():
    """Acceptance criterion of issue #15: the optimizer works with only the base
    (non-GUI) install, so nothing in it may import from segmentator.gui — it used
    to reach into gui/spec.py's STATEFUL set for `needs_contiguous`.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(optimize.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("segmentator.gui"), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("segmentator.gui"), alias.name


def test_needs_contiguous_reads_stateful_off_the_registry():
    assert optimize.needs_contiguous([{"type": "mog2"}]) is True
    assert optimize.needs_contiguous([{"type": "gray"}, {"type": "canny"}]) is False
    assert optimize.needs_contiguous([]) is False
