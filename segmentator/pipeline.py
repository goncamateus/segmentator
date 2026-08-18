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
            for stage in self.stages:
                stage.apply(ctx)
                name = getattr(stage, "name", None)
                if name is not None:
                    # No copy: stages rebind ctx.image rather than mutating it.
                    ctx.taps[name] = ctx.image
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
