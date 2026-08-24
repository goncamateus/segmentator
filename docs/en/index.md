# Segmentator

Configuration-driven image and video segmentation. A pipeline is described in
YAML, so trying a different preprocessing chain, detector or background model is
a config edit, not a code edit.

```bash
uv run segmentator examples/baseline.yaml
uv run segmentator examples/structure.yaml
uv run segmentator examples/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
```

`--video` and `--output` override the source path and the first ffmpeg sink path
for one run.

There is also an editor, which authors these configs with a live preview of every
stage and every sink:

```bash
uv sync --extra gui
uv run segmentator-gui examples/motion.yaml
```

Part of the [goncanalyser](https://github.com/goncamateus/goncanalyser) suite.
goncanalyser is the Qt workspace where you *tune* an operator chain by hand;
segmentator is the headless engine that *runs* the tuned recipe over a batch. Every
operator goncanalyser exposes exists here as a stage, and the two agree
numerically on the same frame with the same parameters.

## Where to go next

| | |
|---|---|
| [Installation](installation.md) | `uv sync`, the `gui` extra, and the prebuilt AppImage / dmg |
| [Writing a config](configuration.md) | the config format: `Ctx`, taps, `input:`, branching with `select` |
| [The catalogue](stages.md) | every source, stage and sink, and what its parameters mean |
| [The editor](gui.md) | the editor, and every operation it allows |

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
| [cli.py](https://github.com/goncamateus/segmentator/blob/main/cli.py) | CLI |
| [segmentator/pipeline.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/pipeline.py) | `Ctx`, protocols, registry, `Pipeline` |
| [segmentator/io.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/io.py) | sources and sinks |
| [segmentator/ops/](https://github.com/goncamateus/segmentator/tree/main/segmentator/ops) | pure operators, no stage protocol, no config object |
| [segmentator/stages/](https://github.com/goncamateus/segmentator/tree/main/segmentator/stages) | the registered stages, by family |
| [segmentator/background_model.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/background_model.py) | fixed mean-of-N background |
| [segmentator/video_writer.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/video_writer.py) | ffmpeg/libx264 pipe |
| [segmentator/gui/](https://github.com/goncamateus/segmentator/tree/main/segmentator/gui) | the PyQt6 editor — optional, `--extra gui` |

## Parity with goncanalyser

Every operator is ported from goncanalyser's `features/*` — the function bodies,
not a reimplementation — with each `Settings` field turned into a constructor
argument on the stage that uses it. On the same frame with the same parameters
the two produce identical output: `edge_px`, corner and keypoint counts, HOG
vectors, LBP codes, histogram plots, contour rows and Hough rows all match
exactly, across the two repos' different OpenCV builds.

`tests/test_stages.py` holds goncanalyser's own `_demo()` assertions, re-pointed
at the stages — ported rather than invented, so a passing suite means the two
repos agree on numbers, not merely that both run.
