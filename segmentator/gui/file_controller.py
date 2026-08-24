"""File I/O: open/save/save-as/the copy-command string, through the document.

Path bookkeeping and the load/save round trip live here so they are testable
without a live window; picking a path (``QFileDialog``) and announcing the
result (title bar, status bar, clipboard) stay in
:class:`~segmentator.gui.window.MainWindow`, which is the only place they can
sensibly happen.
"""

from __future__ import annotations

from pathlib import Path

from segmentator.gui import spec as spec_module
from segmentator.gui.document_controller import DocumentController


class FileController:
    """Owns the open config's path and the load/save round trip."""

    def __init__(self, document: DocumentController, path: str | Path) -> None:
        self.document = document
        self.path = Path(path)

    def open(self, path: str | Path) -> None:
        """Load a config from ``path``, replacing the document's ``cfg`` in place.

        The document controller itself is not replaced — other controllers
        already hold a reference to it, and swapping its ``cfg`` is what keeps
        them all pointed at the newly opened document.
        """
        self.path = Path(path)
        self.document.cfg = spec_module.load(self.path)

    def save(self) -> None:
        spec_module.save(self.path, self.document.cfg)

    def save_as(self, path: str | Path) -> None:
        """Redirect future saves to ``path``. Does not itself write — call :meth:`save` after."""
        self.path = Path(path)

    def run_command(self) -> str:
        """The batch run this editor deliberately does not do."""
        return f"uv run segmentator {self.path}"
