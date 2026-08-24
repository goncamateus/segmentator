"""Add/remove/select/shift/reorder of stages and sinks, against the document.

Index bookkeeping — where a fresh row lands, what should be selected after a
delete, whether a move is in range — lives here so it is testable without a
live window. There is no separate ``select`` method: each of ``add``,
``remove`` and ``shift`` already returns the row selection should land on, so
"select" is realized as part of the operation that changed the list rather
than as a second call a caller has to remember to make. Reading the
*current* row off a ``QListWidget`` and applying the new selection to it stay
in :class:`~segmentator.gui.window.MainWindow`, which is the only place that
touches those widgets.
"""

from __future__ import annotations

from segmentator.gui import spec as spec_module
from segmentator.gui.document_controller import DocumentController


class EditOpsController:
    """Owns add/remove/shift/reorder of one document's stage and sink lists."""

    def __init__(self, document: DocumentController) -> None:
        self.document = document

    def add(self, kind: str, type_name: str, at: int) -> int:
        """Insert a fresh ``type_name`` spec at ``at``.

        Returns ``at`` unchanged, for symmetry with :meth:`remove`/:meth:`shift`
        — a caller selects whatever this returns without special-casing add.
        """
        self.document.insert(kind, at, spec_module.new_spec(kind, type_name))
        return at

    def remove(self, kind: str, at: int) -> int:
        """Delete the entry at ``at``. Returns the row that should now be selected.

        ``-1`` once the list is empty, matching Qt's own "nothing selected".
        """
        self.document.delete(kind, at)
        return min(at, len(self.document.specs(kind)) - 1)

    def shift(self, kind: str, at: int, delta: int) -> int | None:
        """Move one entry by ``delta`` rows. ``None`` and unmoved if that would go out of range."""
        target = at + delta
        if at < 0 or not 0 <= target < len(self.document.specs(kind)):
            return None
        self.document.move(kind, at, target)
        return target

    def move(self, kind: str, src: int, dst: int) -> None:
        """Reorder one entry, e.g. from a list-widget drag. Always in range by construction."""
        self.document.move(kind, src, dst)
