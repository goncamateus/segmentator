"""Find a shorter stage chain that produces the same sink output.

A chain tuned by hand in the editor is a ratchet: a stage gets added to fix what
is on screen, a slider moves, another stage goes on top, and nothing is ever
taken back out. What was load-bearing three edits ago may contribute nothing now.

Two kinds of finding come out of here, and they are *not* equally strong:

**Structural** findings are sound by construction and need no video at all — an
identity parameter, a stage whose output nothing reads, a run of per-pixel
operations that collapses into one lookup table. The argument is over the config,
or over the whole 256-value input domain, so there is nothing to sample.

**Sampled** findings come from dropping a stage and re-running: they can only
ever say *no counterexample was found in N frames*. That is falsification, not
proof, and the sample size matters more than it looks — on the config this was
built against, one stage looked redundant at four sampled frames and was
demonstrably load-bearing at eight. Hence :data:`SAMPLE_FLOOR`, and hence
nothing here applies a finding on its own: the caller proposes, a human accepts.

Everything a caller hands us is a plain ``{type: ..., **params}`` mapping, the
same shape :func:`segmentator.pipeline.build` takes, so this module stays
importable with no Qt and no editor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# Importing these is what fills the registry, and every table below is keyed by a
# registered name — without it the analysis quietly finds nothing at all.
from segmentator import io, stages  # noqa: F401
from segmentator.pipeline import Ctx, _build_stages, frame_for

# Everything a stage writes *besides* ``ctx.image``, qualified by namespace so one
# table covers all three: ``store:``, ``metrics:`` and ``rows:``. PUBLISHES above
# is the image-valued subset of the ``store:`` entries, which is all a preview can
# resolve; this is the whole picture, which is what a dead-stage analysis needs —
# a stage feeding nothing but a ``json`` sink's metric is still very much alive.
#
# ``metrics:*`` means the key set is data- or config-dependent and cannot be
# written down: ``histogram`` names its metrics after the chosen colour space, and
# ``keypoints`` only writes ``descriptor`` on frames where the detector returned
# one. A consumer must treat those stages as writing an unknown metric.
#
# Hand-maintained like STATEFUL, and checked by _demo() in both directions: that
# the names are real stages, and that `spec.PUBLISHES` never drifts out of it.
WRITES: dict[str, tuple[str, ...]] = {
    "canny": ("metrics:edge_px",),
    "sobel": ("metrics:edge_px",),
    "laplacian": ("metrics:edge_px",),
    "hough_lines": ("store:lines", "metrics:lines", "rows:lines"),
    "hough_circles": ("store:circles", "metrics:circles", "rows:circles"),
    "harris": ("store:corners", "metrics:corners", "rows:corners"),
    "shi_tomasi": ("store:corners", "metrics:corners", "rows:corners"),
    "contours": (
        "store:contours",
        "store:contour_mask",
        "metrics:contours",
        "metrics:contour_area",
        "rows:contours",
    ),
    "bounding_boxes": ("store:boxes", "metrics:boxes", "rows:boxes"),
    "blobs": ("metrics:blobs", "rows:blobs"),
    "keypoints": ("store:keypoints", "store:descriptors", "metrics:*", "rows:keypoints"),
    "hog": ("store:hog", "metrics:hog_dim", "metrics:hog_mean"),
    "lbp": ("metrics:lbp_bins", "metrics:lbp_entropy", "rows:lbp"),
    "histogram": ("store:histogram", "metrics:*", "rows:histogram"),
    "motion_objects": (
        "store:boxes",
        "metrics:motion_px",
        "metrics:motion_frac",
        "metrics:motion_objects",
        "metrics:motion_speed",
        "rows:motion",
    ),
    "motion_heat": ("store:heat",),
    "static_mask": ("store:mask",),
    "color_select": ("store:mask",),
    "roi": ("store:roi",),
}


# Store keys a stage reads without naming them in a parameter — the analysis
# cannot see these by looking at the spec, because there is nothing in the spec
# to see. Same maintenance deal as WRITES.
READS: dict[str, tuple[str, ...]] = {
    "mean_background": ("store:mask",),
    "heatmap": ("store:heat",),
    "bounding_boxes": ("store:contours",),
    "paste_roi": ("store:roi",),
}

# Parameters whose value names an image a stage reads.
IMAGE_PARAMS = ("input", "mask", "draw_on")


# Frames the sampled search runs each candidate over. Below the floor the answer
# is not worth reporting — see the module docstring.
SAMPLE_DEFAULT = 16
SAMPLE_FLOOR = 8

# Stages whose output pixel is a function of the input pixel alone, so a run of
# them composes into a single 256-entry table. ``threshold`` qualifies only with
# ``otsu: false`` — Otsu derives its level from the frame histogram, which makes
# the mapping frame-dependent. Checked exhaustively before any fusion is
# proposed, so a wrong entry here costs a rejected candidate, not a wrong answer.
POINT_OPS = ("brightness_contrast", "gamma", "threshold")

# Parameter values that make a stage a passthrough. Read off the spec, never off
# a built instance: `Morphology.ksize` and friends are consumed at construction
# and leave no attribute behind, which is what `spec.rebuild_params` exists for.
IDENTITY = {
    "gaussian_blur": {"ksize": (0, 1)},
    "median_blur": {"ksize": (0, 1)},
    "gamma": {"value": (1.0, 1)},
    "saturation": {"gain": (1.0, 1)},
    "morphology": {"iterations": (0,)},
}

Progress = Callable[[int, int, str], bool]


@dataclass(frozen=True)
class Finding:
    """One proposed rewrite of the stage list."""

    positions: tuple[int, ...]
    replacement: tuple[dict[str, Any], ...]  # empty tuple = delete outright
    label: str
    basis: str  # identity | dataflow | exhaustive | sampled
    saved_ms: float = 0.0
    detail: str = ""

    @property
    def proven(self) -> bool:
        """True when the argument does not rest on a finite sample of frames."""
        return self.basis != "sampled"


# --------------------------------------------------------------------------- #
# What a run of this config can be observed to produce
# --------------------------------------------------------------------------- #


def observable(sink_specs: Sequence[dict[str, Any]]) -> set[str]:
    """The qualified keys the sinks actually read.

    Anything not in here is invisible from outside the run, so a rewrite is free
    to change it. Two of these are easy to forget: ``csv`` reads ``ctx.rows`` and
    ``json`` reads the whole of ``ctx.metrics``, so a stage that contributes
    nothing but a metric is still observable when either is present.

    A config with no sinks at all — the editor previewing a chain — is treated as
    observing the final image, which is what is on screen.
    """
    seen: set[str] = set()
    for spec in sink_specs:
        kind = spec.get("type")
        if kind in ("display", "ffmpeg", "image"):
            seen.add(f"image:{spec.get('input', 'image')}")
        elif kind == "crops":
            seen.add(f"image:{spec.get('input', 'source')}")
            seen.add(f"rows:{spec.get('kind', 'contours')}")
        elif kind == "csv":
            kinds = spec.get("kinds") or ()
            seen.update(f"rows:{k}" for k in kinds) if kinds else seen.add("rows:*")
        elif kind == "json":
            seen.add("metrics:*")
    return seen or {"image:image"}


def _extract(ctx: Ctx, keys: Iterable[str]) -> tuple:
    """Snapshot the observable part of one frame's context.

    Arrays are copied: taps are references into ``ctx.image``, ``static_mask``
    republishes one array every frame and ``motion_heat`` publishes its live
    accumulator, so holding a reference here would compare a value against a
    later version of itself.
    """
    out = []
    for key in sorted(keys):
        namespace, _, name = key.partition(":")
        if namespace == "image":
            out.append(frame_for(ctx, name).copy())
        elif namespace == "metrics":
            # Ordered: the json sink writes the dict as it stands, key order
            # included, so a reordering is a real difference in the output file.
            out.append(tuple(ctx.metrics.items()))
        elif name == "*":
            out.append(tuple((k, _rows(v)) for k, v in ctx.rows.items()))
        else:
            out.append(_rows(ctx.rows.get(name, [])))
    return tuple(out)


def _rows(rows: Sequence[dict[str, Any]]) -> tuple:
    return tuple(tuple(sorted(row.items())) for row in rows)


def _same(left: tuple, right: tuple) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)):
                return False
        elif a != b:
            return False
    return True


def run_chain(specs: Sequence[dict[str, Any]], frames: Sequence[tuple[int, np.ndarray]], keys: Iterable[str]) -> list[tuple]:
    """Observations of ``specs`` over ``frames``, in order.

    Stages are built once and driven in frame order, so a stateful chain sees the
    history it expects. Named stages fill ``ctx.taps`` exactly as
    :meth:`Pipeline.apply` does, or a later ``mask:`` would not resolve.
    """
    stages = _build_stages(list(specs))
    keys = tuple(keys)
    out = []
    for index, frame in frames:
        ctx = Ctx(image=frame, source=frame, index=index)
        for stage in stages:
            stage.apply(ctx)
            name = getattr(stage, "name", None)
            if name is not None:
                ctx.taps[name] = ctx.image
        out.append(_extract(ctx, keys))
    return out


# Frames each timing run averages over. One is not enough: a chain ending in
# `contours` costs whatever that frame's contour count happens to be, and the
# spread across frames is larger than most of the stages being measured.
COST_FRAMES = 4

# Passes run before the clock starts, to get OpenCV's thread pool off the ground.
WARMUP_PASSES = 2


def stage_costs(
    specs: Sequence[dict[str, Any]],
    frames: Sequence[tuple[int, np.ndarray]],
    repeats: int = 3,
) -> list[float]:
    """Mean ms per frame for each stage, from one instrumented pass over the sample.

    Timing every candidate chain end to end was the obvious way to price a
    finding and it did not work: two repeats put the noise well above a stage
    worth 19 ms, and enough repeats to settle cost more than the search itself.
    Pricing every stage once and reading the answers off is both steadier and
    O(1) in the number of candidates.

    The figure is steady-state, and the difference is not small. OpenCV's pthread
    pool parks its workers, and waking them dominates a short burst: ``saturation``
    on a 1080p frame prices at 13 ms from cold and 2.1 ms once the pool has been
    running, and the second number holds indefinitely. Hence WARMUP_PASSES — but
    what actually makes the numbers usable is that every stage in a chain is timed
    inside the same pass, so whatever state the pool is in, it is the same state
    for all of them and the comparison between them stands.

    ponytail: a hint for ranking findings in a dialog, not a benchmark. Do not
    read an absolute ms off this and expect a stopwatch to agree.
    """
    stages = _build_stages(list(specs))
    sample = list(frames)[:COST_FRAMES] or list(frames)
    totals = [0.0] * len(stages)

    for repeat in range(-WARMUP_PASSES, repeats):
        for index, frame in sample:
            ctx = Ctx(image=frame, source=frame, index=index)
            for position, stage in enumerate(stages):
                began = time.perf_counter()
                stage.apply(ctx)
                elapsed = (time.perf_counter() - began) * 1000
                name = getattr(stage, "name", None)
                if name is not None:
                    ctx.taps[name] = ctx.image
                if repeat >= 0:
                    totals[position] += elapsed
    return [total / (repeats * len(sample)) for total in totals]


def chain_cost(specs: Sequence[dict[str, Any]], frames: Sequence[tuple[int, np.ndarray]]) -> float:
    """What one frame costs this chain, in ms. Only meaningful against the same frames."""
    return sum(stage_costs(specs, frames))


# --------------------------------------------------------------------------- #
# Structural findings — sound by construction, no video required
# --------------------------------------------------------------------------- #

# Channel count after a stage, where it is a function of the stage type alone.
# Anything absent from both sets is either channel-preserving (`threshold`,
# `morphology`) or depends on an image this walk cannot see (`apply_mask`,
# `select`), and the walk gives up rather than guess.
_MAKES_GRAY = frozenset(
    {
        "gray", "canny", "sobel", "laplacian", "hog", "lbp", "adaptive_threshold",
        "mog2", "knn", "frame_diff", "three_frame_diff", "farneback", "lucas_kanade",
    }
)
_MAKES_BGR = frozenset(
    {
        "hough_lines", "hough_circles", "harris", "shi_tomasi", "contours",
        "bounding_boxes", "blobs", "keypoints", "heatmap", "motion_objects",
        "paste_roi", "saturation", "colorspace",
    }
)
_UNKNOWN_CHANNELS = frozenset({"apply_mask", "select", "resize", "roi", "histogram", "color_select"})


def _default(type_name: str, param: str) -> Any:
    """A stage parameter's constructor default, or None if it has no such parameter."""
    import inspect

    from segmentator.pipeline import _REGISTRY

    cls = _REGISTRY["stage"].get(type_name)
    if cls is None:
        return None
    found = inspect.signature(cls.__init__).parameters.get(param)
    if found is None or found.default is inspect.Parameter.empty:
        return None
    return found.default


def named_reads(spec: dict[str, Any]) -> set[str]:
    """Image names this stage resolves through ``frame_for`` — taps and store keys alike."""
    kind = spec.get("type", "")
    names = {spec.get(p, _default(kind, p)) for p in IMAGE_PARAMS if _default(kind, p) is not None or p in spec}
    names |= {key.partition(":")[2] for key in READS.get(kind, ())}
    return {n for n in names if isinstance(n, str)}


def consumes_image(spec: dict[str, Any]) -> bool:
    """Whether this stage reads the working image it was handed.

    Almost every stage does. The exceptions are the two that rebind ``ctx.image``
    wholesale from somewhere else — a ``select`` or an ``apply_mask`` pointed at
    ``source`` or a tap. That is the barrier a dead-stage analysis hangs on: it
    is exactly the edit that silently orphans everything above it.
    """
    kind = spec.get("type")
    if kind == "select":
        return spec.get("input", _default("select", "input")) == "image"
    if kind == "apply_mask":
        return "image" in (
            spec.get("input", _default("apply_mask", "input")),
            spec.get("mask", _default("apply_mask", "mask")),
        )
    return True


def _image_is_read(specs: Sequence[dict[str, Any]], position: int, observed: set[str]) -> bool:
    """Does anything downstream look at the image this stage leaves behind?"""
    for spec in specs[position + 1 :]:
        if consumes_image(spec):
            return True
        return False  # a barrier: the working image is discarded unread
    return "image:image" in observed


def dead_stages(specs: Sequence[dict[str, Any]], observed: set[str]) -> list[Finding]:
    """Stages whose every output — image, tap and store key — goes unread."""
    found = []
    metrics_watched = any(k.startswith("metrics") for k in observed)
    for position, spec in enumerate(specs):
        if _image_is_read(specs, position, observed):
            continue
        name = spec.get("name")
        downstream = {n for later in specs[position + 1 :] for n in named_reads(later)}
        if name is not None and (name in downstream or f"image:{name}" in observed):
            continue
        alive = False
        for key in WRITES.get(spec.get("type", ""), ()):
            namespace, _, written = key.partition(":")
            if namespace == "store":
                alive |= written in downstream or f"image:{written}" in observed
            elif namespace == "metrics":
                alive |= metrics_watched
            else:
                alive |= "rows:*" in observed or f"rows:{written}" in observed
        if alive:
            continue
        found.append(
            Finding(
                positions=(position,),
                replacement=(),
                label=f"drop {spec.get('type')}",
                basis="dataflow",
                detail="nothing downstream reads its output",
            )
        )
    return found


def identity_stages(specs: Sequence[dict[str, Any]]) -> list[Finding]:
    """Stages parameterised into being a passthrough."""
    found = []
    for position, spec in enumerate(specs):
        if spec.get("name"):
            continue  # the tap is still a name something may resolve
        rules = IDENTITY.get(spec.get("type", ""))
        if rules is None:
            continue
        values = {p: spec.get(p, _default(spec["type"], p)) for p in rules}
        if all(values[p] in allowed for p, allowed in rules.items()):
            shown = ", ".join(f"{p}: {values[p]}" for p in rules)
            found.append(
                Finding(
                    positions=(position,),
                    replacement=(),
                    label=f"drop {spec['type']}",
                    basis="identity",
                    detail=f"{shown} is a passthrough",
                )
            )
    return found


def redundant_grays(specs: Sequence[dict[str, Any]]) -> list[Finding]:
    """``gray`` applied to an image that is already single-channel.

    Costs almost nothing to run — ``Gray.apply`` is a guarded no-op — so this is
    reported as a clarity finding with no saving attached. The walk stops the
    moment a stage's output channel count stops being knowable from its type.
    """
    found: list[Finding] = []
    channels = 3  # a decoded frame, or a folder of images
    for position, spec in enumerate(specs):
        kind = spec.get("type", "")
        if kind == "gray":
            if channels == 1 and not spec.get("name"):
                found.append(
                    Finding(
                        positions=(position,),
                        replacement=(),
                        label="drop gray",
                        basis="dataflow",
                        detail="the image is already single-channel here",
                    )
                )
            channels = 1
        elif kind in _UNKNOWN_CHANNELS:
            break
        elif kind in _MAKES_GRAY:
            channels = 1
        elif kind in _MAKES_BGR:
            channels = 3
    return found


def _point_table(specs: Sequence[dict[str, Any]]) -> np.ndarray | None:
    """The 256-entry table a run of point ops collapses to, or None if it does not.

    Exhaustive rather than algebraic, which matters: ``brightness_contrast`` is
    ``convertScaleAbs``, i.e. ``|src * alpha + beta|`` saturated, and that
    absolute value means two of them do not compose as an affine map. Feeding all
    256 values through the real stages gets the reflection right; composing the
    parameters by hand would not.

    Both a 3-channel and a 1-channel probe are checked, and all three channels
    must agree, or the run is not a single-table function and is left alone.
    """
    ramp = np.arange(256, dtype=np.uint8)
    colour = Ctx(image=np.repeat(ramp[None, :, None], 3, axis=2), source=None, index=0)
    grey = Ctx(image=ramp[None, :].copy(), source=None, index=0)
    try:
        for stage in _build_stages(list(specs)):
            stage.apply(colour)
        for stage in _build_stages(list(specs)):
            stage.apply(grey)
    except Exception:
        return None
    if colour.image.shape != (1, 256, 3) or grey.image.shape != (1, 256):
        return None
    table = colour.image[0, :, 0]
    if not (np.array_equal(colour.image[0, :, 1], table) and np.array_equal(colour.image[0, :, 2], table)):
        return None  # channels disagree: not one table
    if not np.array_equal(grey.image[0], table):
        return None  # behaves differently on a single-channel image
    return table


def _is_point_op(spec: dict[str, Any]) -> bool:
    kind = spec.get("type")
    if kind not in POINT_OPS or spec.get("name"):
        return False
    return not (kind == "threshold" and spec.get("otsu", _default("threshold", "otsu")))


def point_op_runs(specs: Sequence[dict[str, Any]]) -> list[Finding]:
    """Maximal runs of adjacent per-pixel stages, each collapsed into one ``lut``."""
    found = []
    start = 0
    while start < len(specs):
        if not _is_point_op(specs[start]):
            start += 1
            continue
        stop = start
        while stop + 1 < len(specs) and _is_point_op(specs[stop + 1]):
            stop += 1
        if stop > start:
            table = _point_table(specs[start : stop + 1])
            if table is not None:
                names = " + ".join(s["type"] for s in specs[start : stop + 1])
                found.append(
                    Finding(
                        positions=tuple(range(start, stop + 1)),
                        replacement=({"type": "lut", "table": [int(v) for v in table]},),
                        label=f"fuse {names} into one lut",
                        basis="exhaustive",
                        detail="verified over all 256 input values, per channel",
                    )
                )
        start = stop + 1
    return found


def structural(specs: Sequence[dict[str, Any]], observed: set[str]) -> list[Finding]:
    """Every finding that needs no video, most valuable first."""
    return [
        *identity_stages(specs),
        *dead_stages(specs, observed),
        *point_op_runs(specs),
        *redundant_grays(specs),
    ]


# --------------------------------------------------------------------------- #
# Sampled findings — falsification against real frames
# --------------------------------------------------------------------------- #


def sample_frames(source: Any, count: int = SAMPLE_DEFAULT, contiguous: bool = False) -> list[tuple[int, np.ndarray]]:
    """Frames to judge candidates on, as ``(index, frame)``.

    Spread across the whole source by default, because a chain tuned on one part
    of a video is exactly where a redundant-looking stage turns out to matter.
    ``contiguous`` walks a single run instead, which is the only valid sampling
    for a chain holding state: a background model handed six scattered frames has
    not seen the history its output depends on.

    ``count`` is raised to :data:`SAMPLE_FLOOR` rather than honoured below it. A
    caller asking for four frames is asking for an answer this cannot give: four
    was enough to call a load-bearing stage redundant on the chain this was built
    against, and eight was not.
    """
    count = max(count, SAMPLE_FLOOR)
    total = int(getattr(source, "count", 0) or 0)
    if contiguous or total <= 0:
        start = max(0, total // 2 - count // 2) if total else 0
        wanted = range(start, start + count)
    else:
        wanted = (int(i) for i in np.linspace(0, max(0, total - 1), min(count, total)))
    frames = []
    for index in wanted:
        frame = source.read(int(index))
        if frame is None:
            break
        frames.append((int(index), frame))
    return frames


def needs_contiguous(specs: Sequence[dict[str, Any]]) -> bool:
    """Whether this chain remembers anything between frames."""
    from segmentator.gui.spec import STATEFUL

    return any(spec.get("type") in STATEFUL for spec in specs)


def apply_findings(specs: Sequence[dict[str, Any]], findings: Sequence[Finding]) -> list[dict[str, Any]]:
    """``specs`` with every finding applied. Highest position first, so indices hold."""
    out = [dict(spec) for spec in specs]
    for finding in sorted(findings, key=lambda f: f.positions[0], reverse=True):
        first, last = finding.positions[0], finding.positions[-1]
        out[first : last + 1] = [dict(spec) for spec in finding.replacement]
    return out


def _drop_candidates(specs: Sequence[dict[str, Any]], skip: set[int]) -> list[Finding]:
    return [
        Finding((position,), (), f"drop {spec.get('type')}", "sampled")
        for position, spec in enumerate(specs)
        if position not in skip
    ]


def _merge_candidates(specs: Sequence[dict[str, Any]], skip: set[int]) -> list[Finding]:
    """Adjacent morphology with the same op and kernel, folded into one call."""
    found = []
    for position in range(len(specs) - 1):
        first, second = specs[position], specs[position + 1]
        if {position, position + 1} & skip or first.get("name"):
            continue
        if first.get("type") != "morphology" or second.get("type") != "morphology":
            continue
        default_k = _default("morphology", "ksize")
        if first.get("op") != second.get("op") or first.get("ksize", default_k) != second.get("ksize", default_k):
            continue
        merged = dict(second)
        merged["iterations"] = first.get("iterations", 1) + second.get("iterations", 1)
        found.append(
            Finding(
                (position, position + 1),
                (merged,),
                f"merge two morphology/{first.get('op')} into iterations: {merged['iterations']}",
                "sampled",
            )
        )
    return found


def check(
    specs: Sequence[dict[str, Any]],
    findings: Sequence[Finding],
    frames: Sequence[tuple[int, np.ndarray]],
    observed: set[str],
    reference: Sequence[tuple] | None = None,
) -> bool:
    """Does applying ``findings`` leave every observed output untouched on every frame?

    Also the combination check: findings that each hold alone do not necessarily
    hold together, so a caller applying a selection must run this over the whole
    selection, not trust the individual results.
    """
    if reference is None:
        reference = run_chain(specs, frames, observed)
    try:
        candidate = run_chain(apply_findings(specs, findings), frames, observed)
    except Exception:
        return False  # a dangling tap reference or a shape contract broken
    return len(candidate) == len(reference) and all(_same(a, b) for a, b in zip(candidate, reference))


def search(
    specs: Sequence[dict[str, Any]],
    observed: set[str],
    frames: Sequence[tuple[int, np.ndarray]],
    skip: set[int] | None = None,
    progress: Progress | None = None,
) -> list[Finding]:
    """Candidates that survive every sampled frame and are measurably faster.

    Each candidate is judged against the *original* chain, not against a chain
    with earlier findings already applied — so the results are independent and a
    user can accept any subset. The price is that accepting several still needs
    :func:`check` over the selection.

    ponytail: one round. A drop that only becomes possible once another stage is
    gone is not found until Optimize is run a second time, which is a button
    press rather than a nested search.
    """
    skip = skip or set()
    proposals = [*_drop_candidates(specs, skip), *_merge_candidates(specs, skip)]
    reference = run_chain(specs, frames, observed)
    survivors = []
    for done, proposal in enumerate(proposals):
        if progress is not None and not progress(done, len(proposals), proposal.label):
            break
        if not check(specs, [proposal], frames, observed, reference):
            continue
        # No "must be faster" gate: the ask is a straighter chain, and a stage
        # that can come out is worth proposing even when the clock cannot tell.
        survivors.append(replace(proposal, detail=f"no counterexample in {len(frames)} frames"))
    return price(specs, survivors, frames)


def price(
    specs: Sequence[dict[str, Any]],
    findings: Sequence[Finding],
    frames: Sequence[tuple[int, np.ndarray]],
) -> list[Finding]:
    """Attach a measured ``saved_ms`` to each finding, biggest saving first.

    Prices are read out of :func:`stage_costs`, where every stage is timed inside
    one interleaved pass. Timing a whole candidate chain against a whole base
    chain instead — the obvious approach — does not survive contact with a warm
    cache: the same 19-stage chain over the same four 1080p frames measured 77 ms
    cold and 40 ms after the search had been hammering those arrays for fifteen
    seconds, which is drift far larger than the stages being priced. Within a
    single pass there is nothing to drift against.
    """
    if not findings:
        return []
    costs = stage_costs(specs, frames)
    priced = []
    for finding in findings:
        removed = sum(costs[position] for position in finding.positions)
        if finding.replacement:
            # Time the replacement in its own interleaved pass and net it off.
            first = finding.positions[0]
            after = stage_costs(apply_findings(specs, [finding]), frames)
            removed -= sum(after[first : first + len(finding.replacement)])
        priced.append(replace(finding, saved_ms=removed))
    return sorted(priced, key=lambda f: -f.saved_ms)


def analyse(
    cfg: dict[str, Any],
    frames: Sequence[tuple[int, np.ndarray]] | None = None,
    progress: Progress | None = None,
) -> list[Finding]:
    """Every rewrite worth proposing for this config, strongest argument first.

    With no ``frames`` only the structural findings come back, which is the whole
    analysis for a caller that has no video to hand. With frames, the structural
    findings are re-checked against them too — they are meant to be sound, so a
    failure there is a bug in one of this module's tables, and silently dropping
    the finding is better than acting on it.
    """
    specs = [dict(spec) for spec in cfg.get("stages", [])]
    observed = observable(cfg.get("sinks", []))
    findings = structural(specs, observed)
    if not frames:
        return findings

    reference = run_chain(specs, frames, observed)
    confirmed = price(
        specs,
        [f for f in findings if check(specs, [f], frames, observed, reference)],
        frames,
    )

    covered = {position for finding in confirmed for position in finding.positions}
    sampled = search(specs, observed, frames, skip=covered, progress=progress)
    return [*confirmed, *sampled]
