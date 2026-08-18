"""Self-check for the pipeline core. Run directly (`python test_pipeline.py`) or under pytest.

Synthetic arrays only — no video file needed.
"""

import numpy as np
import yaml

from core.pipeline import Ctx, Pipeline, build, register
from core.stages import MeanBackground, StaticMask, Threshold


class ArraySource:
    """Stand-in for VideoSource over a list of frames."""

    def __init__(self, frames):
        self.frames = frames
        self.fps = 30.0
        self.size = (frames[0].shape[1], frames[0].shape[0])
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        self.closed = True


class RecordingSink:
    """Records every frame it sees; optionally stops the run at frame `stop_at`."""

    def __init__(self, stop_at=None):
        self.seen = []
        self.stop_at = stop_at
        self.closed = False

    def write(self, ctx):
        self.seen.append(ctx.image.copy())
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


def test_build_from_spec():
    """A {'type': ..., **params} mapping becomes a configured instance."""
    stage = build("stage", {"type": "threshold", "value": 15, "mode": "tozero"})
    assert isinstance(stage, Threshold)
    assert (stage.value, stage.mode) == (15, "tozero")

    try:
        build("stage", {"type": "nope"})
    except KeyError as exc:
        assert "nope" in str(exc) and "threshold" in str(exc), exc
    else:
        raise AssertionError("unknown stage type should raise")


def test_yaml_roundtrip_runs_stages_in_order():
    """A YAML document builds a pipeline whose stages run top to bottom."""
    cfg = yaml.safe_load(
        """
        source: {type: _test_array}
        stages:
          - {type: _test_tag, label: first}
          - {type: _test_tag, label: second}
        sinks: []
        """
    )
    frames = [np.zeros((4, 4), np.uint8)]
    register("source", "_test_array")(lambda **_: ArraySource(frames))
    pipe = Pipeline.from_config(cfg)
    recorder = RecordingSink()
    pipe.sinks = [recorder]
    pipe.run()

    ctx = Ctx(image=frames[0], source=frames[0])
    for stage in pipe.stages:
        stage.apply(ctx)
    assert ctx.store["order"] == ["first", "second"]
    assert recorder.seen[0].max() == 2, "both stages should have incremented the frame"


def test_mean_background_cancels_a_static_scene():
    """On an unchanging scene the mean model subtracts the frame down to zero."""
    scene = np.full((8, 8), 100, np.uint8)
    stage = MeanBackground(n_frames=5, use_mask=False)
    for i in range(6):
        ctx = Ctx(image=scene.copy(), source=scene, index=i)
        stage.apply(ctx)
    assert stage.ready
    assert ctx.image.max() == 0, ctx.image


def test_static_mask_is_fitted_once():
    """The mask comes from the first frame and is reused, not refitted."""
    dark = np.zeros((4, 4), np.uint8)
    bright = np.full((4, 4), 255, np.uint8)

    stage = StaticMask(threshold=127, invert=True)
    stage.apply(Ctx(image=dark, source=dark, index=0))
    first = stage.mask.copy()
    assert first.max() == 255, "invert=True selects the dark region"

    ctx = Ctx(image=bright, source=bright, index=1)
    stage.apply(ctx)
    assert np.array_equal(stage.mask, first), "mask must not be refitted"
    assert ctx.store["mask"] is stage.mask


def test_stop_reaches_every_sink_then_closes():
    """A sink returning False stops the run only after all sinks saw that frame."""
    frames = [np.full((4, 4), i, np.uint8) for i in range(5)]
    source = ArraySource(frames)
    stopper = RecordingSink(stop_at=2)
    other = RecordingSink()

    with Pipeline(source, [], [stopper, other]) as pipe:
        processed = pipe.run()

    assert processed == 3, processed
    assert len(stopper.seen) == len(other.seen) == 3, "the later sink must see the last frame too"
    assert stopper.closed and other.closed and source.closed


def test_max_frames_caps_the_run():
    frames = [np.zeros((4, 4), np.uint8) for _ in range(10)]
    sink = RecordingSink()
    with Pipeline(ArraySource(frames), [], [sink]) as pipe:
        assert pipe.run(max_frames=4) == 4
    assert len(sink.seen) == 4


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok  {name}")
    print("all checks passed")
