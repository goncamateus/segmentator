"""The document: the spec list, its labels, marks, and default-sink lookups.

This is the Qt-free half of what :class:`~segmentator.gui.window.MainWindow`
used to do itself under "the document" — CRUD on the ``stages``/``sinks``
sequences of the ruamel config, plus the read-only lookups a row's label, its
:class:`~segmentator.gui.window.RowDelegate` decorations, and a sink's default
``input:`` all derive from the same spec dict.

Constructible with nothing but the loaded config (``spec_module.load(path)``),
so it is testable without a live ``QMainWindow`` and reusable if the window
ever needs to swap documents (``open()``) or hand the same one to another
controller (an edit-ops or optimize controller, say).

Every edit goes straight into the ruamel document handed in at construction —
never into a parallel model — so what a caller reads back (and what a save
writes out) is the same object with the edit applied, comments included.
"""

from __future__ import annotations

from typing import Any

from ruamel.yaml.comments import CommentedMap

from segmentator.gui import spec as spec_module


def _move_comment_preserving(seq: Any, src: int, dst: int) -> None:
    """Move one item of a ruamel sequence, taking its comment with it.

    ponytail: a standalone comment block *between* two items belongs to
    whichever item ruamel attached it to, so reordering can leave one behind.
    Reordering a commented stage is rare enough to not be worth a
    comment-reparenting pass.
    """
    comments = {index: seq.ca.items.get(index) for index in range(len(seq))}
    item = seq.pop(src)
    moved = comments.pop(src, None)
    order = [comments[index] for index in sorted(comments)]
    seq.insert(dst, item)
    order.insert(dst, moved)
    seq.ca.items.clear()
    for index, comment in enumerate(order):
        if comment is not None:
            seq.ca.items[index] = comment


class DocumentController:
    """Owns the spec list: CRUD, labels, marks, default-sink resolution."""

    def __init__(self, cfg: CommentedMap) -> None:
        self.cfg = cfg

    # --- spec-list CRUD ------------------------------------------------- #

    def specs(self, kind: str) -> Any:
        """The live ``stages``/``sinks`` sequence, created empty if absent."""
        key = "stages" if kind == "stage" else "sinks"
        if key not in self.cfg:
            self.cfg[key] = []
        return self.cfg[key]

    def insert(self, kind: str, position: int, entry: Any) -> None:
        self.specs(kind).insert(position, entry)

    def delete(self, kind: str, position: int) -> None:
        del self.specs(kind)[position]

    def move(self, kind: str, src: int, dst: int) -> None:
        """Reorder one entry, keeping its ruamel comment attached to it."""
        _move_comment_preserving(self.specs(kind), src, dst)

    # --- derived, read-only -------------------------------------------- #

    def image_keys(self, upto: int | None = None) -> tuple[str, ...]:
        """Everything an ``input:`` or ``draw_on:`` resolves to at this point.

        A sink runs after the whole chain and sees all of it (``upto=None``). A
        stage sees only what the stages above it already produced — offering a
        name from below would write a config that raises on every frame.
        """
        entries = self.specs("stage")
        above = entries if upto is None else entries[:upto]
        names = [s.get("name") for s in above if s.get("name")]
        published = [key for s in above for key in spec_module.publishes(s.get("type", ""))]
        return ("image", "source", *names, *published)

    def label(self, kind: str, position: int, entry: dict[str, Any]) -> str:
        """The row's text. The tap and the state dot are drawn, not written."""
        type_name = entry.get("type", "?")
        if kind == "sink":
            return f"{type_name}  ← {entry.get('input', self.sink_default(type_name))}"
        return f"{position + 1}.  {type_name}"

    def marks(self, kind: str, entry: dict[str, Any]) -> dict[str, Any]:
        """What ``RowDelegate`` paints beside the label."""
        type_name = entry.get("type", "?")
        return {
            "kind": kind,
            "type": type_name,
            "name": entry.get("name") if kind == "stage" else None,
            "stateful": kind == "stage" and spec_module.is_stateful(type_name),
        }

    def sink_default(self, type_name: str) -> str:
        if type_name == "csv":
            return "rows"
        if type_name == "json":
            return "metrics"
        return "source" if type_name == "crops" else "image"
