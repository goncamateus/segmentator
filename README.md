# segmentator

Configuration-driven image and video segmentation. A pipeline is described in
YAML, so trying a different preprocessing chain, detector or background model is
a config edit, not a code edit.

```bash
uv run segmentator configs/baseline.yaml
uv run segmentator configs/structure.yaml
uv run segmentator configs/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
```

`--video` and `--output` override the source path and the first ffmpeg sink path
for one run.

There is also an editor, which authors these configs with a live preview of every
stage and every sink:

```bash
uv sync --extra gui
uv run segmentator-gui configs/motion.yaml
```

Part of the [goncanalyser](https://github.com/goncamateus/goncanalyser) suite.
goncanalyser is the Qt workspace where you *tune* an operator chain by hand;
segmentator is the headless engine that *runs* the tuned recipe over a batch. Every
operator goncanalyser exposes exists here as a stage, and the two agree
numerically — see [Parity](#parity).

## Documentation

| | |
|---|---|
| [docs/configuration.md](docs/configuration.md) | the config format: `Ctx`, taps, `input:`, branching with `select` |
| [docs/stages.md](docs/stages.md) | every source, stage and sink, and what its parameters mean |
| [docs/gui.md](docs/gui.md) | the editor, and every operation it allows |

## Architecture

A pipeline is `source → stages → sinks`. Stages are an ordered list applied to every
frame; nothing branches on runtime state, so there is no state machine here — the only
"state" is background-warmup vs steady, which `BackgroundModel.ready` handles with one `if`.

```python
@dataclass
class Ctx:
    image: np.ndarray   # working array; stages rebind it
    source: np.ndarray  # original BGR frame, untouched
    index: int
    store: dict         # side channel within one frame
    taps: dict          # named stages' outputs, for sinks to pick from
    metrics: dict       # per-frame scalars, what the json sink writes
    rows: dict          # per-object detail, what the csv and crops sinks write

class Stage(Protocol):
    def apply(self, ctx: Ctx) -> None: ...

class Sink(Protocol):
    def write(self, ctx: Ctx) -> bool: ...   # False stops the run
    def close(self) -> None: ...
```

Components register themselves by name (`@register("stage", "median_blur")`), and
`build()` turns a `{type: ..., **params}` mapping into an instance. An unknown name
raises with the list of known ones. Nothing else has to be told: the config format,
the error messages and the editor's palette and forms are all read off the registry
and the constructors.

| File | Contents |
|---|---|
| [cli.py](cli.py) | CLI |
| [segmentator/pipeline.py](segmentator/pipeline.py) | `Ctx`, protocols, registry, `Pipeline` |
| [segmentator/io.py](segmentator/io.py) | sources and sinks |
| [segmentator/ops/](segmentator/ops/) | pure operators, no stage protocol, no config object |
| [segmentator/stages/](segmentator/stages/) | the registered stages, by family |
| [segmentator/background_model.py](segmentator/background_model.py) | fixed mean-of-N background |
| [segmentator/video_writer.py](segmentator/video_writer.py) | ffmpeg/libx264 pipe |
| [segmentator/gui/](segmentator/gui/) | the PyQt6 editor — optional, `--extra gui` |

## Parity

Every operator is ported from goncanalyser's `features/*` — the function bodies,
not a reimplementation — with each `Settings` field turned into a constructor
argument on the stage that uses it. On the same frame with the same parameters
the two produce identical output: `edge_px`, corner and keypoint counts, HOG
vectors, LBP codes, histogram plots, contour rows and Hough rows all match
exactly, across the two repos' different OpenCV builds.

One collector is deliberately **not** ported. goncanalyser gathers deferred draw
callables in `Result.ops` so an overlay can be composited onto whichever canvas the
user picks later. In a linear chain, ordering *is* the composition — `canny` then
`harris(draw_on: image)` puts the corners on the edge map — so there is nothing for
a deferred list to buy.

goncanalyser's `MotionState` also defends against re-analysing the same frame and
against seeking backwards. Neither can happen in a batch run: `Pipeline.run` is a
single forward pass over a monotonic `ctx.index`. Only the shape guard survives on
the stages themselves — a resized frame invalidates a flow model. Both other rules
come straight back the moment there is a GUI, and
[docs/gui.md](docs/gui.md#moving-through-the-video) is where they now live.

`goncanalyser/dataset/` (COCO, rosbag, Optuna parameter search, dataset statistics)
is dataset tooling rather than image processing, and is not ported.

## Tests

```bash
uv run pytest
```

`tests/test_stages.py` holds goncanalyser's own `_demo()` assertions, re-pointed at
the stages — ported rather than invented, so a passing suite means the two repos
agree on numbers, not merely that both run. `tests/test_gui.py` covers the editor
and runs headless under the offscreen Qt platform.
