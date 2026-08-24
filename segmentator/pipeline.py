"""Pipeline core: per-frame context, component protocols, registry, runner.

A pipeline is an ordered list of :class:`Stage` objects applied to every frame,
plus a :class:`Source` to read from and one or more :class:`Sink` objects to
write to. Components are looked up by name so a whole pipeline can be described
in YAML, or composed directly in Python.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import yaml


@dataclass
class Ctx:
    """State for a single frame, passed down the stage chain.

    Attributes:
        image: The working array. Stages read it and **assign a new array** to it.
        source: The original BGR frame, untouched by any stage.
        index: Zero-based frame number.
        store: Side channel for artifacts one stage produces and a later stage
            consumes within the same frame (e.g. ``store["mask"]``). Discarded
            after the frame; state that must survive across frames belongs on
            the stage instance.
        taps: Snapshots of ``image`` taken after each *named* stage, so a sink can
            output a mid-chain frame. Filled by the pipeline runner, never by a
            stage — keeping it apart from ``store`` means a stage named ``mask``
            cannot shadow the ``store["mask"]`` artifact.
        metrics: Per-frame scalars a stage measured (``motion_px``, ``contours``,
            ``lbp_entropy``). What the ``json`` sink writes.
        rows: Per-object detail, one list of flat dicts per kind (``contours``,
            ``keypoints``, ``motion``). Kept apart from ``metrics`` because these
            are what the ``csv`` and ``crops`` sinks consume, and there can be
            four hundred contours behind a single ``contours=400`` metric.

    Note:
        ``image`` and ``source`` alias the same buffer when the frame enters the
        chain, so stages must never modify the array in place — always rebind
        ``ctx.image``. No defensive copy is taken, since these videos are large
        enough that a per-frame copy is pure waste.
    """

    image: np.ndarray
    source: np.ndarray
    index: int = 0
    store: dict[str, Any] = field(default_factory=dict)
    taps: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def frame_for(ctx: Ctx, key: str = "image") -> np.ndarray:
    """Resolve a sink's ``input`` to an array.

    Order: ``image`` (the chain's final output), ``source`` (the untouched frame),
    a named stage's tap, then any image-valued entry in ``ctx.store``. That last
    fallback is what makes ``input: mask`` work — ``static_mask`` publishes a mask
    but leaves ``ctx.image`` alone, so naming that stage taps its *input* image
    rather than the mask it produced.

    Raises:
        KeyError: If nothing under ``key`` is an image, listing what is available.
    """
    if key == "image":
        return ctx.image
    if key == "source":
        return ctx.source
    if key in ctx.taps:
        return ctx.taps[key]
    artifact = ctx.store.get(key)
    if isinstance(artifact, np.ndarray):
        return artifact
    available = ["image", "source", *sorted(ctx.taps)]
    available += sorted(k for k, v in ctx.store.items() if isinstance(v, np.ndarray))
    detail = " (not an image)" if key in ctx.store else ""
    raise KeyError(f"no image named {key!r}{detail}; available: {available}")


@runtime_checkable
class Stage(Protocol):
    """One transformation applied to every frame."""

    def apply(self, ctx: Ctx) -> None:
        """Transform ``ctx.image`` in place on the context (rebinding, not mutating)."""


class StageInfo:
    """Class-level self-description every registered stage class exposes.

    A stage overrides only what differs from the defaults below, which
    describe the common case: an ordinary per-frame, stateless stage that
    touches nothing but ``ctx.image``. Read this off the class — via
    :func:`component`, e.g. ``component("stage", "canny").WRITES`` — rather
    than a hand-maintained table elsewhere, so a stage's behavioural contract
    lives in exactly one place.

    Precedent: ``StaticMask.RECONSTRUCT`` in
    :mod:`segmentator.stages.mask` is the same idea already, one attribute
    early — a plain class attribute the editor reads instead of asking the
    stage a question at runtime. These attributes generalise that to every
    registered stage.

    As of this writing these attributes are additive: :mod:`segmentator.optimize`
    and :mod:`segmentator.gui.spec` still carry their own hand-maintained tables
    (``WRITES``/``READS``/``POINT_OPS``/``IDENTITY`` and
    ``STATEFUL``/``PUBLISHES``/``CHANNEL_PARAMS`` respectively) covering the same
    ground; nothing here has replaced them yet.

    Attributes:
        STATEFUL: Carries something across frames on the instance (a
            background model, a previous frame, an accumulator) rather than
            being a pure function of the current ``ctx``.
        PUBLISHES: ``ctx.store`` keys this stage writes that hold an *image* —
            i.e. resolvable through :func:`frame_for` by a later stage's
            ``input``/``mask``/``draw_on`` or a sink's ``input:``.
        CHANNEL_PARAMS: Constructor parameter names that are a per-channel
            setting (e.g. ``color_select``'s ``ch0``/``ch1``/``ch2``), so a
            caller that also knows the colour space can label each with its
            own channel letter instead of the generic parameter name.
        READS: Qualified ``store:``/``metrics:``/``rows:`` keys this stage
            reads that are **not** already visible as one of its own
            constructor parameters — an ``input``/``mask``/``draw_on``-style
            parameter already names what it reads; this is for implicit
            reads beyond that (e.g. ``mean_background`` reading
            ``store:mask`` without a parameter naming it).
        WRITES: Qualified ``store:``/``metrics:``/``rows:`` keys this stage
            writes, besides ``ctx.image`` itself. ``"metrics:*"`` marks a
            data- or config-dependent key set that cannot be written down in
            full (e.g. ``histogram`` names its metrics after the chosen
            colour space).
        POINT_OP: The output pixel is a function of the corresponding input
            pixel alone, so a run of these can be losslessly fused into one
            lookup table. Only ``True`` for stages verified exhaustively over
            all 256 input values (``brightness_contrast``, ``gamma``,
            ``threshold``) — an instance-level exception (``threshold`` with
            ``otsu: true``, which derives its level from the whole frame) is
            still the caller's concern to check, exactly as it is today.
        IDENTITY_PARAMS: Parameter name -> tuple of values that make this
            stage a passthrough, read off the spec rather than a built
            instance (e.g. ``{"iterations": (0,)}`` for ``morphology``).
    """

    STATEFUL: bool = False
    PUBLISHES: tuple[str, ...] = ()
    CHANNEL_PARAMS: tuple[str, ...] = ()
    READS: tuple[str, ...] = ()
    WRITES: tuple[str, ...] = ()
    POINT_OP: bool = False
    IDENTITY_PARAMS: dict[str, tuple] = {}


@runtime_checkable
class Source(Protocol):
    """A frame producer."""

    fps: float
    size: tuple[int, int]  # (width, height)

    def __iter__(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


@runtime_checkable
class Sink(Protocol):
    """A consumer of finished frames."""

    def write(self, ctx: Ctx) -> bool:
        """Consume the frame. Return ``False`` to ask the pipeline to stop."""

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, dict[str, type]] = {"source": {}, "stage": {}, "sink": {}}


def register(kind: str, name: str) -> Callable[[type], type]:
    """Class decorator registering a component under ``kind``/``name``."""

    def decorate(cls: type) -> type:
        _REGISTRY[kind][name] = cls
        return cls

    return decorate


def build(kind: str, spec: Mapping[str, Any]) -> Any:
    """Instantiate one component from a ``{"type": name, **kwargs}`` mapping."""
    params = dict(spec)
    try:
        name = params.pop("type")
    except KeyError:
        raise KeyError(f"{kind} spec is missing a 'type' key: {spec!r}") from None
    table = _REGISTRY[kind]
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; known: {sorted(table)}")
    return table[name](**params)


def registered(kind: str) -> list[str]:
    """Names available for a component kind — handy for error messages and docs."""
    return sorted(_REGISTRY[kind])


def component(kind: str, name: str) -> type:
    """The registered class behind a ``kind``/``name`` pair, unbuilt.

    What a caller wants when it needs to read a class's own attributes — e.g. a
    :class:`StageInfo` field — rather than an instance built from a spec, which
    is what :func:`build` is for. Raises the same way :func:`build` does.
    """
    table = _REGISTRY[kind]
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; known: {sorted(table)}")
    return table[name]


def _build_stages(specs: Sequence[Mapping[str, Any]]) -> list[Stage]:
    """Build the stage chain, moving each spec's optional ``name`` onto the instance.

    ``name`` is stripped here rather than in :func:`build` so stage constructors
    never have to accept it and ``build`` stays a plain ``{type, **kwargs}`` mapping.
    """
    stages: list[Stage] = []
    seen: dict[str, int] = {}
    for position, spec in enumerate(specs):
        params = dict(spec)
        name = params.pop("name", None)
        stage = build("stage", params)
        if name is not None:
            if name in seen:
                raise ValueError(
                    f"duplicate stage name {name!r} at positions {seen[name]} and {position}"
                )
            seen[name] = position
            stage.name = name
        stages.append(stage)
    return stages


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    """Runs ``source -> stages -> sinks``. Use as a context manager to release both."""

    def __init__(self, source: Source, stages: Sequence[Stage], sinks: Sequence[Sink]):
        self.source = source
        self.stages = list(stages)
        self.sinks = list(sinks)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Pipeline:
        """Build from a parsed config mapping (``source`` / ``stages`` / ``sinks``)."""
        # Imported here, not at module scope: these modules import `register` from
        # this one, so a top-level import would be circular. Importing them is what
        # populates the registry with the built-in components.
        from segmentator import io, stages  # noqa: F401

        source = build("source", cfg["source"])
        built_stages = _build_stages(cfg.get("stages", []))
        built_sinks = [build("sink", s) for s in cfg.get("sinks", [])]
        for sink in built_sinks:
            bind = getattr(sink, "bind_source", None)
            if bind is not None:
                bind(source)
        return cls(source, built_stages, built_sinks)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        """Build from a YAML file."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_config(yaml.safe_load(handle))

    def apply(self, ctx: Ctx) -> None:
        """Run every stage over one frame, filling ``ctx.taps`` as it goes.

        What :meth:`run` does per frame, named so the GUI editor's preview worker
        can drive one frame at a time without re-deriving what a frame goes
        through.
        """
        for stage in self.stages:
            stage.apply(ctx)
            name = getattr(stage, "name", None)
            if name is not None:
                # No copy: stages rebind ctx.image rather than mutating it.
                ctx.taps[name] = ctx.image

    def run(self, max_frames: int | None = None) -> int:
        """Process frames until the source ends, a sink stops, or ``max_frames``.

        Returns:
            Number of frames processed.
        """
        count = 0
        for index, frame in enumerate(self.source):
            if max_frames is not None and index >= max_frames:
                break
            ctx = Ctx(image=frame, source=frame, index=index)
            self.apply(ctx)
            count += 1
            # List comprehension, not a generator: `all` short-circuits, which would
            # skip the video writer the moment a display window returned False.
            if not all([sink.write(ctx) for sink in self.sinks]):
                break
        return count

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
        self.source.close()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
