"""Drift guard: the docs' stated stage count vs. the live registry.

`docs/en/gui.md` and `docs/pt/gui.md` each state how many stages
`registered("stage")` returns, as part of explaining the `+` palette in
"Adding, removing and reordering stages". That number has gone stale twice
before (see issue #4) with nothing to catch it. This test is the catch.

    uv run pytest tests/test_docs.py
"""

import re
from pathlib import Path

from segmentator import io, stages  # noqa: F401 - populates the registry
from segmentator.pipeline import registered

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
GUI_DOCS = (DOCS_DIR / "en" / "gui.md", DOCS_DIR / "pt" / "gui.md")

# Matches "... 41 stages, 3 sources, 6 sinks ..." in either language: the
# surrounding prose differs between docs/en/gui.md and docs/pt/gui.md, but
# this fragment ("stages"/"sources"/"sinks" stay in English in both) does not.
# `\s+` (not a literal space) between the words: the source wraps this line
# in both files, so a plain space would miss the "6\nsinks" line break.
STAGE_COUNT_RE = re.compile(r"(\d+)\s+stages,\s+\d+\s+sources,\s+\d+\s+sinks")


def test_gui_docs_stage_count_matches_registry():
    """Each doc's stated stage count must match `len(registered("stage"))`."""
    # A leading underscore (test_pipeline.py's "_test_tag") marks a registration
    # that exists only for a test's own process, never meant to be documented —
    # same filter `segmentator/gui/spec.py`'s own FAMILIES self-check uses,
    # since the registry is process-global and a full test run can leave one
    # behind by the time this runs.
    live_count = len([n for n in registered("stage") if not n.startswith("_")])
    stale = []
    for doc in GUI_DOCS:
        text = doc.read_text(encoding="utf-8")
        match = STAGE_COUNT_RE.search(text)
        assert match, f"{doc}: no '<N> stages, <N> sources, <N> sinks' line found"
        documented_count = int(match.group(1))
        if documented_count != live_count:
            stale.append(f"{doc}: says {documented_count}, registry has {live_count}")
    assert not stale, "stage count drifted from the registry:\n" + "\n".join(stale)
