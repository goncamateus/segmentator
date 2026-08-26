"""Diff the predecessor against the reconstructed MOG2 version of itself.

Writes ``generated/legacy-mog2.diff``: a summary stat, then the two hunks that
carry the change. The deleted background-model module is left as a stat line
rather than 28 lines of removed source — the point of the figure is how much
had to move, not what was in it.

    uv run python docs/whitepaper/legacy_diff.py

See legacy/README.md for what ``before/`` and ``after/`` are and which of the
two is a reconstruction.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
LEGACY = HERE / "legacy"
GENERATED = HERE / "generated"

# Files whose hunks are shown in full. Anything else in the tree appears only in
# the stat summary above them.
SHOWN = ("core/config.py", "gasvid_0.py")


def _git_diff(*args: str) -> str:
    """`git diff --no-index` between the two trees; exit code 1 just means differences."""
    result = subprocess.run(
        ["git", "diff", "--no-index", "--no-color", *args],
        cwd=LEGACY,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _tidy(text: str) -> str:
    """Strip the before/after scaffolding so the diff reads as one project's history."""
    text = re.sub(r"\{before => after\}/", "", text)
    text = re.sub(r"\b([ab])/(?:before|after)/", r"\1/", text)
    text = re.sub(r"^ before/", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^index .*\n", "", text, flags=re.MULTILINE)
    return text


def _macros(stat: str) -> str:
    """The comparison's numbers, read off the diff rather than counted by hand."""
    files, insertions, deletions = (
        int(re.search(rf"(\d+) {word}", stat).group(1))
        for word in ("files? changed", "insertions?", "deletions?")
    )
    loc = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in sorted((LEGACY / "before").rglob("*.py"))
    )
    modules = len(list((LEGACY / "before").rglob("*.py")))
    return "\n".join(
        [
            rf"\newcommand{{\LegacyLOC}}{{{loc}}}",
            rf"\newcommand{{\LegacyModules}}{{{modules}}}",
            rf"\newcommand{{\LegacyFiles}}{{{files}}}",
            rf"\newcommand{{\LegacyInsertions}}{{{insertions}}}",
            rf"\newcommand{{\LegacyDeletions}}{{{deletions}}}",
            rf"\newcommand{{\LegacyChanged}}{{{insertions + deletions}}}",
        ]
    )


def main() -> None:
    GENERATED.mkdir(exist_ok=True)
    stat = _git_diff("--stat", "before", "after")
    parts = [_tidy(stat).rstrip(), ""]
    for name in SHOWN:
        parts.append(_tidy(_git_diff("-U2", f"before/{name}", f"after/{name}")).rstrip())
    body = "\n".join(parts) + "\n"
    (GENERATED / "legacy-mog2.diff").write_text(body, encoding="utf-8")
    (GENERATED / "legacy-numbers.tex").write_text(_macros(stat) + "\n", encoding="utf-8")
    print(f"{len(body.splitlines())} lines -> generated/legacy-mog2.diff")


if __name__ == "__main__":
    main()
