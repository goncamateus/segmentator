"""Preview tabs: what should exist, and which one is currently shown.

Reads available stages/sinks through the document controller; creating tabs on
the widget and switching which one has focus stay in
:class:`~segmentator.gui.window.MainWindow`, which is the only place that
touches ``QTabBar``.
"""

from __future__ import annotations

from segmentator.gui.document_controller import DocumentController
from segmentator.gui.worker import preview_key

# Sink kinds whose `input:` resolves to an image a tab can show.
SINK_IMAGE_TYPES = ("display", "ffmpeg", "image", "crops")


class PreviewTabsController:
    """Computes the ``(label, key)`` pairs the tab strip should show, and tracks which is current."""

    def __init__(self, document: DocumentController) -> None:
        self.document = document
        self.current = "source"

    def tabs(self, selected_stage: int | None) -> list[tuple[str, str]]:
        """One entry per tab, in display order.

        The selected stage (if any and in range) leads, then ``source``, then
        one entry per image-producing sink.
        """
        wanted = [("source", "source")]
        entries = self.document.specs("stage")
        if selected_stage is not None and 0 <= selected_stage < len(entries):
            entry = entries[selected_stage]
            wanted.insert(0, (f"selected: {entry.get('type', '?')}", preview_key(selected_stage)))
        for entry in self.document.specs("sink"):
            type_name = entry.get("type", "?")
            if type_name not in SINK_IMAGE_TYPES:
                continue
            key = entry.get("input", self.document.sink_default(type_name))
            wanted.append((f"{type_name} ← {key}", key))
        return wanted

    def select(self, key: str) -> None:
        self.current = key
