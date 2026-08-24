# Writing a config

**What this is.** The file format: what a pipeline is made of, how a sink picks
what it writes, and the one shape the format cannot express. Everything here
applies equally to a batch run and to [the editor](gui.md).

A pipeline is `source → stages → sinks`. Stages are an ordered list applied to
every frame; nothing branches on runtime state, so there is no state machine —
the only "state" is background-warmup vs steady, which `BackgroundModel.ready`
handles with one `if`.

```yaml
name: baseline

source:
  type: video
  path: inputs/gasvid.mp4

stages:
  - {type: gray}
  - {type: median_blur, ksize: 7, name: smoothed}
  - {type: static_mask, threshold: 127}
  - {type: mean_background, n_frames: 60}
  - {type: apply_mask, name: masked}

sinks:
  - {type: ffmpeg, path: outputs/result.mp4}
  - {type: display, input: source}
```

Every entry is a `{type: ..., **params}` mapping. `type` picks the registered
class, the rest are its constructor arguments, and an unknown name raises with
the list of known ones. There is no schema to keep in step: the constructor
*is* the schema.

## The context

![Anatomy of a pipeline](../assets/pipeline-anatomy.svg)

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

Two kinds of state, deliberately kept apart:

| State | Lives in | Example |
|---|---|---|
| Across frames | the stage instance | `StaticMask.mask`, `MeanBackground._model`, `Farneback._history` |
| Within one frame | `ctx.store` | `store["mask"]` written by `static_mask`, read by `apply_mask` |

That side channel is what a plain `ndarray -> ndarray` chain cannot express: the
mean background model accumulates the *masked* frame but subtracts from the
*unmasked* one.

**Contract for stages:** rebind `ctx.image`, never mutate the array in place —
`image` and `source` alias the same buffer when a frame enters the chain, and no
defensive copy is taken (these videos are large enough that per-frame copies are
pure waste).

## Choosing what a sink outputs

Give a stage a `name:` to tap its output, then point a sink at it with `input:`:

```yaml
stages:
  - {type: gray}
  - {type: median_blur, ksize: 7, name: smoothed}
  - {type: static_mask, threshold: 127}
  - {type: mean_background, n_frames: 60}
  - {type: apply_mask, name: masked}

sinks:
  - {type: ffmpeg, path: outputs/result.mp4}                 # default: the final image
  - {type: display, input: source}                           # untouched footage
  - {type: display, input: smoothed}                         # mid-chain
  - {type: display, input: mask}                             # the ROI mask artifact
```

`input:` resolves in this order — `image` (the chain's final output, the
default), `source`, a named stage's tap, then any image-valued entry in
`ctx.store`. Anything else raises, listing what is available. Display sinks title
their window after `input` unless given a `window:`, so several of them coexist
instead of fighting over one window.

Taps are opt-in: an unnamed stage costs nothing, and a duplicated `name:` is
rejected.

> **Gotcha:** a tap holds `ctx.image` *after* the stage ran. `static_mask`
> publishes a mask but leaves the frame alone, so `name:` on it taps its input
> image — reach the mask itself with `input: mask`.

## Branching with `select`

![Branching with taps and select](../assets/config-branching.svg)

A stage that draws consumes the working frame. `select` reaches back past it,
using the same resolution rules as a sink's `input:`:

```yaml
- {type: canny, name: edges}
- {type: hough_lines, draw_on: source}   # leaves a colour overlay in ctx.image
- {type: select, input: edges}           # …so reach back to the edge map
- {type: contours, draw_on: image}
```

That is enough for a *tree* — one producer, several consumers. It is not enough
for a *merge*, two producers feeding one stage, which the config format cannot
express. Masking this frame by that other branch, differencing two preprocessing
variants, compositing an overlay onto a differently-processed base: all of those
need stages to take an `input:` of their own, a dict of named buffers instead of
one `ctx.image`, and a topological sort — plus a decision about what `ctx.store`
means when two branches both write `contours`.

## Running it

```bash
uv run segmentator examples/baseline.yaml
uv run segmentator examples/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
```

`--video` and `--output` override the source path and the first ffmpeg sink path
for one run; `--max-frames` stops early. Nothing else about a config is
overridable from the command line, on purpose — a run that differs in more than
its input and its output is a different config, and configs are cheap.

## Composing in Python

The `Pipeline` constructor already composes, so there is no builder class:

```python
from segmentator.io import DisplaySink, VideoSource
from segmentator.pipeline import Pipeline
from segmentator.stages.motion import Mog2
from segmentator.stages.preprocess import GaussianBlur, Gray, Threshold

with Pipeline(
    VideoSource("inputs/gasvid1.mp4"),
    [Gray(), GaussianBlur(ksize=9), Mog2(history=200), Threshold(value=128)],
    [DisplaySink()],
) as pipeline:
    pipeline.run(max_frames=300)
```

`Pipeline.apply(ctx)` runs one frame's worth of chain and fills the taps — what
`run()` does per frame, and what the editor's preview thread drives one frame at
a time.
