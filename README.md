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

Part of the [goncanalyser](https://github.com/goncamateus/goncanalyser) suite.
goncanalyser is the Qt workspace where you *tune* an operator chain by hand;
segmentator is the headless engine that *runs* the tuned recipe over a batch. Every
operator goncanalyser exposes exists here as a stage, and the two agree
numerically — see [Parity](#parity).

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

Two kinds of state, deliberately kept apart:

| State | Lives in | Example |
|---|---|---|
| Across frames | the stage instance | `StaticMask.mask`, `MeanBackground._model`, `Farneback._history` |
| Within one frame | `ctx.store` | `store["mask"]` written by `static_mask`, read by `apply_mask` |

That side channel is what a plain `ndarray -> ndarray` chain cannot express: the mean
background model accumulates the *masked* frame but subtracts from the *unmasked* one.

**Contract for stages:** rebind `ctx.image`, never mutate the array in place — `image`
and `source` alias the same buffer when a frame enters the chain, and no defensive copy
is taken (these videos are large enough that per-frame copies are pure waste).

Components register themselves by name (`@register("stage", "median_blur")`), and
`build()` turns a `{type: ..., **params}` mapping into an instance. An unknown name
raises with the list of known ones.

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

`input:` resolves in this order — `image` (the chain's final output, the default),
`source`, a named stage's tap, then any image-valued entry in `ctx.store`. Anything else
raises, listing what is available. Display sinks title their window after `input` unless
given a `window:`, so several of them coexist instead of fighting over one window.

Taps are opt-in: an unnamed stage costs nothing, and a duplicated `name:` is rejected.

> **Gotcha:** a tap holds `ctx.image` *after* the stage ran. `static_mask` publishes a mask
> but leaves the frame alone, so `name:` on it taps its input image — reach the mask itself
> with `input: mask`.

### Branching with `select`

A stage that draws consumes the working frame. `select` reaches back past it, using
the same resolution rules as a sink's `input:`:

```yaml
- {type: canny, name: edges}
- {type: hough_lines, draw_on: source}   # leaves a colour overlay in ctx.image
- {type: select, input: edges}           # …so reach back to the edge map
- {type: contours, draw_on: image}
```

That is enough for a *tree* — one producer, several consumers. It is not enough for a
*merge*, two producers feeding one stage, which the config format cannot express.

| File | Contents |
|---|---|
| [cli.py](cli.py) | CLI |
| [segmentator/pipeline.py](segmentator/pipeline.py) | `Ctx`, protocols, registry, `Pipeline` |
| [segmentator/io.py](segmentator/io.py) | sources and sinks |
| [segmentator/ops/](segmentator/ops/) | pure operators, no stage protocol, no config object |
| [segmentator/stages/](segmentator/stages/) | the registered stages, by family |
| [segmentator/background_model.py](segmentator/background_model.py) | fixed mean-of-N background |
| [segmentator/video_writer.py](segmentator/video_writer.py) | ffmpeg/libx264 pipe |
| [docs/gui-evaluation.md](docs/gui-evaluation.md) | evaluation of a node-based config editor |

## Stages

**Adjust** — `brightness_contrast(brightness, contrast)`, `saturation(gain)`,
`gamma(value)`, `gray`, `colorspace(to)`, `resize(size)`, `select(input)`

**Blur / morphology** — `median_blur(ksize)`, `gaussian_blur(ksize, sigma)`,
`clahe(clip_limit, tile_grid)`, `morphology(op, ksize, iterations)`

**Threshold** — `threshold(value, mode, otsu)` — modes `binary`, `binary_inv`,
`trunc`, `tozero`, `tozero_inv` — `adaptive_threshold(method, block, c, invert)`

**Region of interest** — `roi(x, y, w, h)`, `paste_roi(border, draw_on)`. The crop runs
*before* the operators that read it, not after: Otsu takes its level from whatever
histogram it is handed and a blur reads neighbours across the border, so analysing
the whole frame and cropping at the end lets outside pixels change inside numbers.
`paste_roi` puts the region back and shifts every coordinate in `ctx.rows` into
frame space.

**Edges** — `canny(lo, hi)`, `sobel(ksize, dx, dy)`, `laplacian(ksize)`

**Geometry** — `hough_lines(...)`, `hough_circles(...)`, `harris(k, quality, max)`,
`shi_tomasi(max, quality, min_dist)`, `contours(min_area, mode, boxes, draw_on)` —
modes `external`, `list`, `tree` — `bounding_boxes(...)`,
`blobs(min_area, max_area, circularity, convexity, dark)`

**Keypoints** — `keypoints(detector, max, sensitivity, octaves, edge, rich)`, where
`detector` is `sift` or `orb`. `sensitivity` is normalised 0..1 and mapped onto each
detector's own native threshold, because SIFT wants ~0.04 where ORB wants ~20 and no
config should have to know that.

**Texture / colour** — `hog(orientations, cell, block)`, `lbp(points, radius, method)`,
`histogram(space, replace)`. HOG costs 150-300 ms on a 640x512 frame — fine for a batch,
too slow for a preview.

**Masking** — `static_mask(threshold, invert)`, `apply_mask(fill)`,
`mean_background(n_frames, use_mask)`

**Motion** — `mog2(...)`, `knn(...)`, `frame_diff(lag)`, `three_frame_diff()`,
`farneback(pyr_scale, levels, winsize, iterations, gain)`,
`lucas_kanade(max_points, win, gain)`. Every one of them emits the same thing: a
single-channel 0..255 heat image. What comes after is ordinary stages —

```yaml
- {type: farneback}
- {type: threshold, value: 25, mode: binary}
- {type: morphology, op: open, ksize: 3}
- {type: motion_objects, min_area: 50}
```

— so a threshold means the same thing whichever algorithm is above it, and swapping
algorithms is one line. Then `motion_heat(window)` accumulates,
`heatmap(opacity, threshold, draw_on)` paints, and
`motion_objects(min_area, max_travel, boxes, labels)` measures.

> `gain` on the flow stages is a **calibration constant, not a derived one**. At the
> default 32, 8 px/frame reads as full scale; a plume drifting under 1 px/frame needs
> it several times higher or a morphological open will erase it. `configs/motion.yaml`
> uses 160 for exactly that reason.

**Sources** — `video(path)`, `image(path)`, `folder(path, pattern, fps)`

**Sinks** — `display(window, size, delay, quit_key)`, `ffmpeg(path, fps)`,
`image(path, input)` (`{index}` in the path numbers a sequence), `csv(dir, kinds)`,
`json(path)` (JSON Lines), `crops(dir, kind, input, pad)`

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
against seeking backwards. Neither can happen here: `Pipeline.run` is a single
forward pass over a monotonic `ctx.index`. Only the shape guard survives, and it is
the one rule that still fires in a batch — a resized frame invalidates a flow model.
Both rules come straight back the moment there is a GUI, which
[docs/gui-evaluation.md](docs/gui-evaluation.md) covers.

`goncanalyser/dataset/` (COCO, rosbag, Optuna parameter search, dataset statistics)
is dataset tooling rather than image processing, and is not ported.

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

## Tests

```bash
uv run pytest
```

`tests/test_stages.py` holds goncanalyser's own `_demo()` assertions, re-pointed at
the stages — ported rather than invented, so a passing suite means the two repos
agree on numbers, not merely that both run.
