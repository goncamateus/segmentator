# Segmentator

Configuration-driven image and video segmentation. A pipeline reads frames from a
source, runs them through an ordered list of stages, and hands the result to one or
more sinks — described entirely in YAML, so changing the recipe is a config edit,
not a code edit.

## Language

**Pipeline**:
The `source → stages → sinks` run over a video or image: a single forward pass,
one `Ctx` per frame, no branching on runtime state.
_Avoid_: workflow, chain (chain refers to the stage list specifically, see Stage)

**Ctx**:
The per-frame state object carried down the stage chain — `image`, `source`,
`index`, plus the four output channels (`store`, `taps`, `metrics`, `rows`).
_Avoid_: context (too easily confused with `CONTEXT.md`'s own sense of "context"),
frame state

**Stage**:
One transformation applied to every frame, in the order it appears in the config.
A stage rebinds `ctx.image` to a new array; it never edits the array in place.
_Avoid_: operator, filter, step

**Sink**:
A pipeline endpoint that consumes `Ctx` to write output; returning `False` from
its write stops the run. One pipeline may have several.
_Avoid_: writer, output, exporter

**Source**:
What a pipeline reads frames from (video file, image, camera). Feeds `Ctx.source`
and, before the first stage runs, `Ctx.image`.

**store**:
The side channel on `Ctx` for an artifact one stage produces and a later stage in
the *same frame* consumes (e.g. `store["mask"]`). Discarded once the frame ends;
state that must survive across frames belongs on the stage instance instead.
_Avoid_: scratch, cache (this is per-frame, not persistent)

**taps**:
Snapshots of `ctx.image` taken after each *named* stage, so a sink can select a
mid-chain frame instead of only the final one. Filled by the pipeline runner, never
by a stage — kept apart from `store` specifically so a stage named `mask` can't
shadow the unrelated `store["mask"]` artifact.
_Avoid_: snapshot, checkpoint

**metrics**:
Per-frame scalars a stage measured (`motion_px`, `contours`, `lbp_entropy`) — what
the `json` sink writes. One dict per frame.
_Avoid_: stats, measurements

**rows**:
Per-object detail: one list of flat dicts per kind (`contours`, `keypoints`,
`motion`) — what the `csv` and `crops` sinks consume. Kept apart from `metrics`
because a single frame's `contours=400` metric can hide four hundred of these.
_Avoid_: records, results

**registry**:
The name → constructor mapping populated by `@register(kind, name)` for every
stage, source, and sink. It is the single source of truth for the YAML schema, the
GUI editor's palette and forms, and unknown-name error messages — nothing about a
component is declared twice.
_Avoid_: catalogue (used loosely in docs prose, but "registry" is the precise term
for the mapping itself)

**Parity**:
The invariant that a stage's numeric output (`edge_px`, contour/keypoint counts,
HOG vectors, LBP codes, ...) matches goncanalyser's same-named function exactly, on
the same input. A hard requirement, not a style preference — `tests/test_stages.py`
exists to enforce it.

**goncanalyser**:
The sibling interactive Qt workspace where an operator chain is tuned by hand.
Segmentator is the headless engine that runs a goncanalyser-tuned recipe over a
batch; every operator goncanalyser exposes exists here as a Stage, ported body-for-
body rather than reimplemented.
_Avoid_: the GUI (ambiguous — segmentator has its own editor too; "goncanalyser"
always means the sibling project)

**gas-plume recipe**:
The stages with no goncanalyser counterpart — `static_mask` and the fixed
mean-of-N background — built directly for this project's own original use case
rather than ported. Contrasted with every other stage, which *is* a goncanalyser
port. The name comes from the shape of that original footage: a fixed dark region
of interest in an otherwise static scene.

**BackgroundModel.ready**:
The one real state machine in the codebase: `False` while the model is still
accumulating its first N frames (warmup), `True` once the mean background is fit
and usable (steady). Every other stage is stateless or purely accumulative — this
is the only place that branches on "have I seen enough yet."
