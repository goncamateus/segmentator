"""Preview-worker lifecycle: build/launch, stop, push edits, and transport.

Owns the :class:`~segmentator.gui.worker.PreviewWorker` instance and every
control call made to it. Connecting the worker's signals to whatever paints
or reports them, and reflecting its playing/position state back into widgets,
stay in :class:`~segmentator.gui.window.MainWindow` — this controller never
touches a widget.
"""

from __future__ import annotations

from segmentator.gui.document_controller import DocumentController
from segmentator.gui.worker import PreviewWorker


class PlaybackController:
    """Owns one :class:`PreviewWorker` at a time, built from the document's config."""

    def __init__(self, document: DocumentController) -> None:
        self.document = document
        self.worker: PreviewWorker | None = None

    def build(self) -> PreviewWorker:
        """Construct a fresh worker over the document's current source and stages.

        Stops whatever worker was running first. May raise ``OSError``,
        ``KeyError`` or ``ValueError`` if the source spec does not build — the
        thread is never started in that case. Split from :meth:`launch` so a
        caller can connect the worker's signals in between, before any frame
        is emitted.
        """
        self.stop()
        self.worker = PreviewWorker(self.document.cfg)
        return self.worker

    def launch(self) -> None:
        """Start the built worker's thread."""
        if self.worker is not None:
            self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def push(self) -> None:
        """Hand the worker a fresh snapshot of the specs. Never mutate in place."""
        if self.worker is None:
            return
        self.worker.source_spec = dict(self.document.cfg["source"])
        self.worker.specs = tuple(dict(entry) for entry in self.document.specs("stage"))

    def set_wanted(self, keys: tuple[str, ...]) -> None:
        """Tell the worker which converted images the caller wants back."""
        if self.worker is not None:
            self.worker.wanted = keys

    def toggle_play(self) -> bool:
        """Flip play/pause. Returns the worker's new state, or ``False`` with no worker."""
        if self.worker is None:
            return False
        self.worker.playing = not self.worker.playing
        return self.worker.playing

    def jump(self, frames: int) -> None:
        """Pause and move a few frames. Negative steps backwards."""
        if self.worker is not None:
            self.worker.step(frames)

    def seek(self, index: int) -> None:
        """Pause and jump to an absolute frame index."""
        if self.worker is not None:
            self.worker.playing = False
            self.worker.seek(index)
