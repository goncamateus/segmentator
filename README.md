# gasvid-seg

Gas-plume segmentation from video. Pipelines are described in YAML, so trying a
different preprocessing chain or background model is a config edit, not a code edit.

```bash
uv run python gasvid.py configs/baseline.yaml
uv run python gasvid.py configs/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
```

`--video` and `--output` override the source path and the ffmpeg sink path for one run.

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

class Stage(Protocol):
    def apply(self, ctx: Ctx) -> None: ...

class Sink(Protocol):
    def write(self, ctx: Ctx) -> bool: ...   # False stops the run
    def close(self) -> None: ...
```

Two kinds of state, deliberately kept apart:

| State | Lives in | Example |
|---|---|---|
| Across frames | the stage instance | `StaticMask.mask`, `MeanBackground._model` |
| Within one frame | `ctx.store` | `store["mask"]` written by `static_mask`, read by `apply_mask` |

That side channel is what a plain `ndarray -> ndarray` chain cannot express: the mean
background model accumulates the *masked* frame but subtracts from the *unmasked* one.

**Contract for stages:** rebind `ctx.image`, never mutate the array in place — `image`
and `source` alias the same buffer when a frame enters the chain, and no defensive copy
is taken (these videos are large enough that per-frame copies are pure waste).

Components register themselves by name (`@register("stage", "median_blur")`), and
`build()` turns a `{type: ..., **params}` mapping into an instance. An unknown name
raises with the list of known ones.

| File | Contents |
|---|---|
| [gasvid.py](gasvid.py) | CLI |
| [core/pipeline.py](core/pipeline.py) | `Ctx`, protocols, registry, `Pipeline` |
| [core/stages.py](core/stages.py) | every stage |
| [core/io.py](core/io.py) | `VideoSource`, `DisplaySink`, `FfmpegSink` |
| [core/background_model.py](core/background_model.py) | fixed mean-of-N background |
| [core/video_writer.py](core/video_writer.py) | ffmpeg/libx264 pipe |

## Stages

**Pre:** `gray`, `colorspace(to)`, `median_blur(ksize)`, `gaussian_blur(ksize, sigma)`,
`clahe(clip_limit, tile_grid)`, `morphology(op, ksize, iterations)`, `resize(size)`

**Mask / detect:** `static_mask(threshold, invert)`, `apply_mask(fill)`,
`mean_background(n_frames, use_mask)`, `mog2(...)`, `knn(...)`, `frame_diff(lag)`

**Post:** `threshold(value, mode, otsu)` — modes `binary`, `binary_inv`, `trunc`,
`tozero`, `tozero_inv` — `contours(min_area, color, thickness, draw_on)`,
`bounding_boxes(...)`

**Sources / sinks:** `video(path)`; `display(window, size, delay, quit_key)`,
`ffmpeg(path, fps)`

## Composing in Python

The `Pipeline` constructor already composes, so there is no builder class:

```python
from core.io import DisplaySink, VideoSource
from core.pipeline import Pipeline
from core.stages import GaussianBlur, Gray, Mog2, Threshold

with Pipeline(
    VideoSource("inputs/gasvid1.mp4"),
    [Gray(), GaussianBlur(ksize=9), Mog2(history=200), Threshold(value=128)],
    [DisplaySink()],
) as pipeline:
    pipeline.run(max_frames=300)
```

## Tests

```bash
uv run python test_pipeline.py   # also runs under pytest
```
