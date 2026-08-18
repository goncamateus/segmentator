"""Tests for the pipeline core. Synthetic arrays only — no video file needed.

    uv run pytest
"""

import numpy as np
import pytest
import yaml

from segmentator.pipeline import Ctx, Pipeline, build, frame_for, register
from segmentator.stages.mask import MeanBackground, StaticMask
from segmentator.stages.preprocess import Threshold


@register("source", "_test_array")
class ArraySource:
    """Stand-in for VideoSource. Buildable from YAML, so `count` generates the frames."""

    def __init__(self, frames=None, count=1, size=4):
        self.frames = frames if frames is not None else [
            np.zeros((size, size), np.uint8) for _ in range(count)
        ]
        self.fps = 30.0
        self.size = (self.frames[0].shape[1], self.frames[0].shape[0])
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        self.closed = True


class RecordingSink:
    """Records every frame it sees; optionally stops the run at frame `stop_at`."""

    def __init__(self, stop_at=None, input="image"):
        self.seen = []
        self.stop_at = stop_at
        self.input = input
        self.closed = False

    def write(self, ctx):
        self.seen.append(frame_for(ctx, self.input).copy())
        return self.stop_at is None or ctx.index < self.stop_at

    def close(self):
        self.closed = True


@register("stage", "_test_tag")
class Tag:
    """Appends its label to ctx.store['order'] so stage ordering is observable."""

    def __init__(self, label):
        self.label = label

    def apply(self, ctx):
        ctx.store.setdefault("order", []).append(self.label)
        ctx.image = ctx.image + 1


@pytest.fixture
def frames():
    """Ten 4x4 frames, each filled with its own index."""
    return [np.full((4, 4), i, np.uint8) for i in range(10)]


def test_build_configures_the_instance():
    """A {'type': ..., **params} mapping becomes a configured instance."""
    stage = build("stage", {"type": "threshold", "value": 15, "mode": "tozero"})
    assert isinstance(stage, Threshold)
    assert (stage.value, stage.mode) == (15, "tozero")


def test_build_rejects_unknown_type_and_lists_known_ones():
    with pytest.raises(KeyError, match="unknown stage 'nope'") as excinfo:
        build("stage", {"type": "nope"})
    assert "threshold" in str(excinfo.value), "the error should list what is available"


def test_build_rejects_spec_without_type():
    with pytest.raises(KeyError, match="missing a 'type' key"):
        build("stage", {"value": 15})


def test_yaml_builds_a_pipeline_that_runs_stages_in_order():
    """A YAML document builds a pipeline whose stages run top to bottom."""
    cfg = yaml.safe_load(
        """
        source: {type: _test_array, count: 1}
        stages:
          - {type: _test_tag, label: first}
          - {type: _test_tag, label: second}
        sinks: []
        """
    )
    pipeline = Pipeline.from_config(cfg)
    recorder = RecordingSink()
    pipeline.sinks = [recorder]
    pipeline.run()

    frame = np.zeros((4, 4), np.uint8)
    ctx = Ctx(image=frame, source=frame)
    for stage in pipeline.stages:
        stage.apply(ctx)
    assert ctx.store["order"] == ["first", "second"]
    assert recorder.seen[0].max() == 2, "both stages should have incremented the frame"


def test_mean_background_cancels_a_static_scene():
    """On an unchanging scene the mean model subtracts the frame down to zero."""
    scene = np.full((8, 8), 100, np.uint8)
    stage = MeanBackground(n_frames=5, use_mask=False)
    for index in range(6):
        ctx = Ctx(image=scene.copy(), source=scene, index=index)
        stage.apply(ctx)
    assert stage.ready
    assert ctx.image.max() == 0


def test_static_mask_is_fitted_once():
    """The mask comes from the first frame and is reused, not refitted."""
    dark = np.zeros((4, 4), np.uint8)
    bright = np.full((4, 4), 255, np.uint8)

    stage = StaticMask(threshold=127, invert=True)
    stage.apply(Ctx(image=dark, source=dark, index=0))
    fitted = stage.mask.copy()
    assert fitted.max() == 255, "invert=True selects the dark region"

    ctx = Ctx(image=bright, source=bright, index=1)
    stage.apply(ctx)
    assert np.array_equal(stage.mask, fitted), "mask must not be refitted"
    assert ctx.store["mask"] is stage.mask


@pytest.mark.parametrize("stage_name", ["clahe", "static_mask", "contours"])
def test_gray_only_stages_reject_a_colour_frame(stage_name):
    """Feeding BGR to a single-channel stage names the fix instead of failing deep in cv2."""
    stage = build("stage", {"type": stage_name})
    colour = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError, match="single-channel"):
        stage.apply(Ctx(image=colour, source=colour))


def test_apply_mask_without_a_mask_names_the_missing_stage():
    frame = np.zeros((4, 4), np.uint8)
    with pytest.raises(KeyError, match="static_mask"):
        build("stage", {"type": "apply_mask"}).apply(Ctx(image=frame, source=frame))


def test_stop_reaches_every_sink_then_closes(frames):
    """A sink returning False stops the run only after all sinks saw that frame."""
    source = ArraySource(frames[:5])
    stopper = RecordingSink(stop_at=2)
    other = RecordingSink()

    with Pipeline(source, [], [stopper, other]) as pipeline:
        processed = pipeline.run()

    assert processed == 3
    assert len(stopper.seen) == len(other.seen) == 3, "the later sink must see the last frame too"
    assert stopper.closed and other.closed and source.closed


def test_max_frames_caps_the_run(frames):
    sink = RecordingSink()
    with Pipeline(ArraySource(frames), [], [sink]) as pipeline:
        assert pipeline.run(max_frames=4) == 4
    assert len(sink.seen) == 4


# --------------------------------------------------------------------------- #
# Named stages and sink input selection
# --------------------------------------------------------------------------- #

TAPPED_CONFIG = """
source: {type: _test_array, count: 1}
stages:
  - {type: _test_tag, label: first, name: early}
  - {type: _test_tag, label: second}
sinks: []
"""


def _run_with_sinks(config_yaml, sinks):
    pipeline = Pipeline.from_config(yaml.safe_load(config_yaml))
    pipeline.sinks = list(sinks)
    pipeline.run()
    return sinks


def test_named_stage_is_tapped_with_its_own_output():
    """A tap holds what the named stage produced, not what later stages did to it."""
    early, final = RecordingSink(input="early"), RecordingSink()
    _run_with_sinks(TAPPED_CONFIG, [early, final])
    assert early.seen[0].max() == 1, "one increment had run when 'early' was tapped"
    assert final.seen[0].max() == 2, "the default input is still the chain's last image"


def test_unnamed_stages_are_not_tapped():
    sink = RecordingSink()
    pipeline = Pipeline.from_config(yaml.safe_load(TAPPED_CONFIG))
    pipeline.sinks = [sink]
    pipeline.run()
    ctx = Ctx(image=np.zeros((4, 4), np.uint8), source=np.zeros((4, 4), np.uint8))
    for stage in pipeline.stages:
        stage.apply(ctx)
        name = getattr(stage, "name", None)
        if name is not None:
            ctx.taps[name] = ctx.image
    assert sorted(ctx.taps) == ["early"]


def test_source_input_survives_later_stages():
    """`input: source` is the untouched frame even after the chain rewrote ctx.image."""
    sink = RecordingSink(input="source")
    _run_with_sinks(TAPPED_CONFIG, [sink])
    assert sink.seen[0].max() == 0, "source must not carry the stages' increments"


def test_duplicate_stage_names_are_rejected():
    config = yaml.safe_load(
        """
        source: {type: _test_array, count: 1}
        stages:
          - {type: _test_tag, label: a, name: same}
          - {type: _test_tag, label: b, name: same}
        sinks: []
        """
    )
    with pytest.raises(ValueError, match="duplicate stage name 'same'"):
        Pipeline.from_config(config)


def test_name_is_not_passed_to_the_stage_constructor():
    """Stage classes never see `name`; the pipeline strips it before building."""
    pipeline = Pipeline.from_config(yaml.safe_load(TAPPED_CONFIG))
    assert pipeline.stages[0].name == "early"
    assert pipeline.stages[0].label == "first", "the real params still reached the class"


def test_frame_for_falls_back_to_image_artifacts():
    """`input: mask` reaches the artifact static_mask publishes, which is not a tap."""
    dark = np.zeros((4, 4), np.uint8)
    ctx = Ctx(image=dark, source=dark)
    StaticMask(threshold=127, invert=True).apply(ctx)
    assert frame_for(ctx, "mask") is ctx.store["mask"]


def test_frame_for_rejects_unknown_and_non_image_names():
    frame = np.zeros((4, 4), np.uint8)
    ctx = Ctx(image=frame, source=frame)
    ctx.taps["early"] = frame
    ctx.store["contours"] = [1, 2, 3]

    with pytest.raises(KeyError, match="no image named 'nope'") as excinfo:
        frame_for(ctx, "nope")
    assert "early" in str(excinfo.value), "the error should list the available names"

    with pytest.raises(KeyError, match="not an image"):
        frame_for(ctx, "contours")


def test_display_sink_closes_without_ever_opening_a_window():
    """A run that ends before its first frame must not crash on the way out.

    `cv2.destroyWindow` on a window that was never created is a null-pointer
    error out of highgui, and an empty source or `--max-frames 0` is not an
    error worth crashing on.
    """
    from segmentator.io import DisplaySink

    DisplaySink().close()
