"""The preview thread. Decoding and the whole stage chain happen here, never on
the GUI thread.

Communication is the same lock-free deal goncanalyser's ``core/worker.py`` uses:

    GUI -> worker : plain attribute assignment (``specs``, ``wanted``, ``playing``).
                    Rebinding an attribute is atomic and the specs handed over are
                    fresh copies, so the worker always reads a consistent snapshot.
    worker -> GUI : Qt signals, which Qt queues across the thread boundary.

Three mechanisms make a *paused* preview trustworthy, which is the whole point of
previewing at all. They are goncanalyser's three ``MotionState`` rules, restated
for a chain whose state lives on the stage instances:

1. **The same frame must not be pushed through a stateful stage twice.** Dragging
   a knob re-renders the frame on screen over and over; feeding MOG2 that image
   fifty times teaches it that the plume *is* the background. The prefix cache
   below means an edit only re-runs the stages below it, and any stateful stage
   that would be re-run gets a fresh instance instead.
2. **A jump resets.** Seeking or stepping backwards means the previous frame is
   no longer the previous frame.
3. **A changed model resets.** A construction parameter — ``history``,
   ``lag``, ``n_frames`` — cannot be assigned to a live instance, so the stage is
   rebuilt. Per-frame knobs are assigned instead, and history survives.

After a reset the stage has no history and would emit a black frame under the
user's hands, so a short **warm-up replay** runs the frames just before the
current one back through the chain prefix first.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from segmentator.gui.spec import STATEFUL, params, rebuild_params
from segmentator.pipeline import Ctx, build, frame_for

# Frames replayed through the chain prefix after a reset, so differencing and
# flow stages have a previous frame again.
#
# ponytail: eight is enough for anything that looks one or two frames back and
# does nothing at all for `mog2 history: 500` — the status line says "reset" so
# the number on screen is not mistaken for a warmed model. Raise it, or replay
# from the last keyframe, only if that turns out to matter.
WARMUP = 8


def to_qimage(image: np.ndarray) -> QImage:
    """Any frame the chain can produce -> a QImage that owns its pixels.

    The ``.copy()`` is not optional: QImage wraps the numpy buffer without taking
    a reference to it, so the array would be freed the moment this returns and
    the widget would paint garbage.

    Handles the three things ``frame_for`` can hand back — BGR, a single-channel
    heat or mask image, and the float32 accumulator ``motion_heat`` publishes.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        buffer = np.ascontiguousarray(image)
        height, width = buffer.shape
        return QImage(buffer.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def _snapshot(ctx: Ctx) -> Ctx:
    """A copy of the context cheap enough to keep one per stage.

    Only the containers are copied, never the arrays: stages rebind ``ctx.image``
    rather than mutating it, so holding the reference is enough to hold the frame
    as it was at that point in the chain.
    """
    return Ctx(
        image=ctx.image,
        source=ctx.source,
        index=ctx.index,
        store=dict(ctx.store),
        taps=dict(ctx.taps),
        metrics=dict(ctx.metrics),
        rows={kind: list(rows) for kind, rows in ctx.rows.items()},
    )


def preview_key(position: int) -> str:
    """The tap name a stage gets when the config did not give it one.

    Every stage is previewable, named or not, and ``#3`` cannot collide with a
    name a config would use.
    """
    return f"#{position}"


class PreviewWorker(QThread):
    """Runs one config over one source and emits whatever the editor is looking at."""

    images_ready = pyqtSignal(dict)  # {preview key: QImage}
    measured = pyqtSignal(int, dict, dict)  # frame index, ctx.metrics, ctx.rows
    position = pyqtSignal(int, int)  # frame index, frame count
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        # --- written by the GUI thread, read here. Rebound, never mutated. ---
        self.specs: tuple[dict[str, Any], ...] = tuple(dict(s) for s in cfg.get("stages", []))
        self.source_spec: dict[str, Any] = dict(cfg["source"])
        self.wanted: tuple[str, ...] = ()
        self.playing = False
        # --- owned here ------------------------------------------------------
        self.source = build("source", dict(self.source_spec))
        self._stages: list[Any] = []
        self._built: tuple[dict[str, Any], ...] = ()
        self._built_source: dict[str, Any] = dict(self.source_spec)
        self._cache: list[Ctx] = []  # _cache[i] is the context after stage i
        self._cached_index: int | None = None
        self._index = 0
        self._seek: int | None = 0
        self._running = True
        self._last_wanted: tuple[str, ...] = ()
        self._note = ""

    # --- control surface, all called from the GUI thread -------------------- #

    def stop(self) -> None:
        """Ask the loop to finish and block until it has. Safe to call twice."""
        self._running = False
        self.wait()
        self.source.close()

    def seek(self, index: int) -> None:
        self._seek = max(0, index)

    def step(self, frames: int) -> None:
        """Pause and move a few frames. Negative steps backwards."""
        self.playing = False
        self._seek = max(0, self._index + frames)

    # --- keeping the built chain in step with the specs --------------------- #

    def _build_stage(self, position: int, spec: dict[str, Any]) -> Any:
        stage = build("stage", {k: v for k, v in spec.items() if k != "name"})
        stage.name = spec.get("name")
        return stage

    def _first_change(self, specs: tuple[dict[str, Any], ...]) -> int | None:
        """Position of the first stage whose spec differs, or None if none do."""
        for position, (old, new) in enumerate(zip(self._built, specs)):
            if old != new:
                return position
        if len(self._built) != len(specs):
            return min(len(self._built), len(specs))
        return None

    def _sync(self, specs: tuple[dict[str, Any], ...], same_frame: bool, reset: bool) -> int | None:
        """Bring the built stages in line with the specs. Returns where to re-run from.

        Rebuild or assign, per rule 3; and per rule 1, any stateful stage that is
        about to see a frame it has already seen is rebuilt whether or not its own
        parameters moved.
        """
        start = self._first_change(specs)
        if start is None and not reset:
            return None
        start = 0 if start is None else start
        warm = reset
        reset_stages: list[str] = []

        for position in range(start, len(specs)):
            spec = specs[position]
            old = self._built[position] if position < len(self._built) else None
            stateful = spec.get("type") in STATEFUL
            changed = set() if old is None else {k for k in old.keys() | spec.keys() if old.get(k) != spec.get(k)}
            rebuild = (
                old is None
                or old.get("type") != spec.get("type")
                or bool(changed & rebuild_params("stage", spec))
                # Rules 1 and 2: it must not be handed a frame it has already
                # learnt from, and after a jump its history is a lie either way.
                or (stateful and (same_frame or reset))
            )
            if rebuild:
                stage = self._build_stage(position, spec)
                if position < len(self._stages):
                    self._stages[position] = stage
                else:
                    self._stages.append(stage)
                if stateful:
                    warm = True
                    reset_stages.append(spec.get("type", "?"))
            else:
                defaults = {p.name: p.default for p in params("stage", spec.get("type", ""))}
                for key in changed - {"name"}:
                    setattr(self._stages[position], key, spec.get(key, defaults.get(key)))
                self._stages[position].name = spec.get("name")

        del self._stages[len(specs) :]
        self._built = specs
        self._invalidate(start)
        if warm:
            self._warm_up(start, reset_stages)
        return start

    def _sync_source(self) -> None:
        spec = dict(self.source_spec)
        if spec == self._built_source:
            return
        self.source.close()
        self.source = build("source", dict(spec))
        self._built_source = spec
        self._cached_index = None
        self._seek = self._index

    def _invalidate(self, start: int) -> None:
        del self._cache[start:]

    # --- rendering ---------------------------------------------------------- #

    def _context(self, index: int) -> Ctx | None:
        frame = self.source.read(index)
        if frame is None:
            return None
        return Ctx(image=frame, source=frame, index=index)

    def _run_chain(self, ctx: Ctx, start: int, stop: int | None = None, cache: bool = True) -> Ctx:
        """Apply stages ``[start:stop]``, caching each one's output as it goes."""
        for position in range(start, len(self._stages) if stop is None else stop):
            stage = self._stages[position]
            stage.apply(ctx)
            # Two keys for the same picture: the position, so an unnamed stage is
            # still previewable, and the config's `name:` if it has one, so a
            # sink's `input:` resolves exactly as it does in a batch run.
            ctx.taps[preview_key(position)] = ctx.image
            if stage.name is not None:
                ctx.taps[stage.name] = ctx.image
            if cache:
                self._cache.append(_snapshot(ctx))
        return ctx

    def _warm_up(self, start: int, reset_stages: list[str]) -> None:
        """Replay the frames before this one so reset stages have history again.

        Only the prefix through the last stateful stage is replayed — everything
        below it is a pure function of the frame it is handed and has nothing to
        remember, so re-running it would be pure cost.
        """
        last = max(
            (i for i, s in enumerate(self._built) if s.get("type") in STATEFUL),
            default=-1,
        )
        if last < start:
            return
        for index in range(max(0, self._index - WARMUP), self._index):
            ctx = self._context(index)
            if ctx is None:
                continue
            try:
                self._run_chain(ctx, 0, last + 1, cache=False)
            except Exception:
                # A half-typed config fails the same way every frame; the render
                # below reports it once instead of eight times.
                break
        # Named, not just announced: "reset" after moving a threshold is confusing
        # until it says that what reset was the object tracker below it.
        self._note = "reset: " + ", ".join(dict.fromkeys(reset_stages)) if reset_stages else ""
        self._cached_index = None

    def _render(self, start: int) -> Ctx | None:
        """The current frame, re-running only what the cache cannot supply."""
        if self._cached_index != self._index:
            self._cache.clear()
            start = 0
        if start > 0 and start <= len(self._cache):
            ctx = _snapshot(self._cache[start - 1])
            del self._cache[start:]
        else:
            ctx = self._context(self._index)
            if ctx is None:
                return None
            self._cache.clear()
            start = 0
        self._cached_index = self._index
        return self._run_chain(ctx, start)

    def _images(self, ctx: Ctx) -> dict[str, QImage]:
        """Convert only what the visible tab asked for.

        Nothing off-screen is ever converted, which is why there is no thumbnail
        coalescer here: the expensive part was never the frame rate, it was
        converting forty canvases nobody was looking at.
        """
        images: dict[str, QImage] = {}
        for key in self.wanted:
            try:
                images[key] = to_qimage(frame_for(ctx, key))
            except KeyError:
                continue  # a sink pointing at a tap that does not exist yet
            except (ValueError, cv2.error) as exc:
                # Unlike a missing tap, this is a real frame that `to_qimage`
                # could not render (e.g. a colour space packed into 2 channels,
                # or carrying a 4th) — worth a status line instead of leaving
                # the tab reading "nothing resolves" with no reason why.
                self.status.emit(f"{key}: {type(exc).__name__}: {exc}")
        return images

    # --- the loop ----------------------------------------------------------- #

    def run(self) -> None:
        fps = float(getattr(self.source, "fps", 0.0) or 0.0)
        delay = max(1, int(1000 / fps)) if fps > 1 else 40
        drawn: tuple | None = None

        while self._running:
            specs, wanted = self.specs, self.wanted
            self._sync_source()

            reset = False
            if self._seek is not None:
                self._index, self._seek = self._seek, None
                self._cached_index = None
                reset = True  # rule 2
            elif self.playing:
                if self._context(self._index + 1) is None:
                    self.playing = False
                    self.status.emit("end of source")
                    self.msleep(50)
                    continue
                self._index += 1
                self._cached_index = None

            same_frame = self._cached_index == self._index
            began = time.perf_counter()  # the warm-up replay is part of what an edit costs
            try:
                start = self._sync(specs, same_frame, reset)
            except Exception as exc:
                # A spec that will not *build* is reported exactly like one that
                # will not *apply*, below: a half-typed parameter is something
                # the editor is mid-edit on, not a reason to take the preview
                # thread down and leave a live-looking window that is dead.
                # `self._built` is left untouched, so the next tick retries and
                # recovers the moment the parameter is corrected.
                self._cached_index = None
                self.status.emit(f"{type(exc).__name__}: {exc}")
                self.msleep(120)
                continue
            if start is None and (specs, wanted, self._index) == drawn:
                self.msleep(20)  # paused on a frame nobody re-tuned
                continue
            drawn = (specs, wanted, self._index)

            try:
                ctx = self._render(0 if start is None else start)
            except Exception as exc:
                self._cached_index = None
                self.status.emit(f"{type(exc).__name__}: {exc}")
                self.msleep(120)
                continue
            if ctx is None:
                self.playing = False
                if self._index == 0:
                    self.failed.emit(f"no frames in {self.source_spec.get('path')}")
                    return
                self._index -= 1
                continue

            self.images_ready.emit(self._images(ctx))
            self.measured.emit(ctx.index, dict(ctx.metrics), dict(ctx.rows))
            count = int(getattr(self.source, "count", 0) or 0)
            self.position.emit(self._index, count)
            end = f"/{count - 1}" if count else ""
            took = (time.perf_counter() - began) * 1000
            note = f"  {self._note}" if self._note else ""
            self._note = ""
            self.status.emit(
                f"frame {self._index}{end}  {ctx.image.shape[1]}x{ctx.image.shape[0]}  "
                f"{took:.0f} ms  {'playing' if self.playing else 'paused'}{note}"
            )
            self.msleep(delay if self.playing else 10)
