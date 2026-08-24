"""Optimizer-worker lifecycle: build, re-check, and apply its findings.

Owns the :class:`~segmentator.gui.optimize_worker.OptimizeWorker` instance and
the arithmetic of turning a chosen set of :class:`~segmentator.optimize.Finding`
into edits on the document. Connecting the worker's signals to a progress
dialog, and showing the findings for the user to tick, stay in
:class:`~segmentator.gui.window.MainWindow` — this controller never touches a
widget.
"""

from __future__ import annotations

from typing import Sequence

from segmentator.gui import spec as spec_module
from segmentator.gui.document_controller import DocumentController
from segmentator.gui.optimize_worker import OptimizeWorker
from segmentator.optimize import SAMPLE_DEFAULT, Finding, check, observable


class OptimizeController:
    """Owns one :class:`OptimizeWorker` at a time, built from the document's config."""

    def __init__(self, document: DocumentController) -> None:
        self.document = document
        self.worker: OptimizeWorker | None = None

    def build(self, samples: int = SAMPLE_DEFAULT) -> OptimizeWorker:
        """Construct a fresh worker over the document's current source and stages."""
        self.worker = OptimizeWorker(self.document.cfg, samples)
        return self.worker

    def release(self) -> None:
        """Drop the worker and the frames it sampled, once its findings are settled."""
        if self.worker is not None:
            self.worker.release()
            self.worker = None

    def recheck(self, findings: Sequence[Finding]) -> bool:
        """Whether ``findings`` still hold together, against the built worker's sample.

        Each finding was judged alone; a subset the user actually picks has to be
        judged again as a whole — see :func:`segmentator.optimize.check`.
        """
        assert self.worker is not None, "recheck needs a built worker's sampled frames"
        specs = [dict(entry) for entry in self.document.specs("stage")]
        observed = observable([dict(entry) for entry in self.document.specs("sink")])
        return check(specs, findings, self.worker.frames, observed)

    def apply(self, findings: Sequence[Finding]) -> int:
        """Rewrite the stage list with ``findings`` applied. Returns the row to select.

        Row by row rather than replacing the sequence: ruamel hangs each comment
        off its item's index, and assigning a fresh list drops every one of them,
        which would show up as the whole file rewritten on the next save.
        """
        for finding in sorted(findings, key=lambda f: f.positions[0], reverse=True):
            first, last = finding.positions[0], finding.positions[-1]
            for position in range(last, first - 1, -1):
                self.document.delete("stage", position)
            for offset, replacement in enumerate(finding.replacement):
                self.document.insert("stage", first + offset, spec_module.as_spec(replacement))
        return min(findings[0].positions[0], len(self.document.specs("stage")) - 1)
