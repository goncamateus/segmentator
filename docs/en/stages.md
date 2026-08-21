# The catalogue

**What this is.** Every registered component and what its parameters mean.
Sources produce frames, stages transform them, sinks consume them; the format
that wires them together is in [Writing a config](configuration.md).

![The stage catalogue, by family](../assets/stage-families.svg)

A stage marked with an amber dot carries state from one frame to the next. That
is worth knowing for two reasons: it is the only thing a stage is allowed to
keep, and it is what [the editor](gui.md) has to reset rather than re-feed when
you tune a paused frame.

## Sources

| | |
|---|---|
| `video(path)` | frames from a video file, in order |
| `image(path, fps)` | a single still, as a one-frame sequence |
| `folder(path, pattern, fps)` | every image in a directory, sorted — a video you cannot scrub |

All three also answer `count` and `read(index)`, which is what makes seeking in
the editor possible. `VideoSource.read` tracks its own position so playing
forward never touches `CAP_PROP_POS_FRAMES` — setting that forces a keyframe hunt
and a re-decode, many times slower than reading the next frame.

## Stages

**Adjust** — `brightness_contrast(brightness, contrast)`, `saturation(gain)`,
`gamma(value)`, `gray`, `colorspace(to)`, `resize(size)`, `select(input)`

**Blur / morphology** — `median_blur(ksize)`, `gaussian_blur(ksize, sigma)`,
`clahe(clip_limit, tile_grid)`, `morphology(op, ksize, iterations)`

**Threshold** — `threshold(value, mode, otsu)` — modes `binary`, `binary_inv`,
`trunc`, `tozero`, `tozero_inv` — `adaptive_threshold(method, block, c, invert)`

**Region of interest** — `roi(x, y, w, h)`, `paste_roi(border, draw_on)`. The crop
runs *before* the operators that read it, not after: Otsu takes its level from
whatever histogram it is handed and a blur reads neighbours across the border, so
analysing the whole frame and cropping at the end lets outside pixels change
inside numbers. `paste_roi` puts the region back and shifts every coordinate in
`ctx.rows` into frame space.

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
slow enough in the editor that you will feel it.

**Masking** — `static_mask(threshold, invert)`, `apply_mask(input, mask, fill)`,
`mean_background(n_frames, use_mask, buffer)`, `color_select(space, ch0, ch1, ch2, fill)`

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

## Sinks

| | |
|---|---|
| `display(window, size, delay, quit_key)` | an OpenCV window; the quit key stops the run |
| `ffmpeg(path, fps, input)` | encode to mp4 through a libx264 pipe |
| `image(path, input)` | one file per frame — `{index}` in the path numbers a sequence |
| `csv(dir, kinds)` | one CSV per row kind, with a `frame` column prepended |
| `json(path)` | one JSON object of `ctx.metrics` per frame (JSON Lines) |
| `crops(dir, kind, input, pad)` | cut every row's rectangle out of a frame and save it |

`input:` on the image-producing sinks picks what they write; see
[Choosing what a sink outputs](configuration.md#choosing-what-a-sink-outputs).

## Adding one

Register it and it exists everywhere — in configs, in the error messages, in the
editor's palette, with a generated form:

```python
@register("stage", "my_stage")
class MyStage:
    def __init__(self, radius: int = 3):
        self.radius = radius

    def apply(self, ctx: Ctx) -> None:
        ctx.image = something(ctx.image, self.radius)
```

Two things the rest of the system reads off that class without being told:
`inspect.signature` gives the parameter form, and keeping `radius` as an
attribute is what marks it a *live* knob the editor can assign while paused. A
parameter consumed at construction — handed to an OpenCV object and forgotten —
is classified as needing a rebuild, which is correct, because that is exactly
what it needs.
