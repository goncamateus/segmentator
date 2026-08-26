"""Generate the appendix catalogue straight out of the component registry.

The registry is the single source of truth for the config schema, the editor's
palette and the CLI's error messages; making it the source of the paper's
appendix too is what stops the document from claiming a stage that does not
exist, or missing one that does. Nothing here is hand-maintained: names come
from ``@register``, parameters from the constructor signature, and the one-line
purpose from the class docstring's first line.

    uv run python docs/whitepaper/catalogue.py
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

from segmentator import io, stages  # noqa: F401  (importing them fills the registry)
from segmentator.pipeline import _REGISTRY

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
GENERATED = HERE / "generated"

# Sphinx roles and reST literals in the docstrings, rewritten for LaTeX.
ROLE = re.compile(r":[a-z]+:`~?(?:[\w.]+\.)?(\w+)`")
LITERAL = re.compile(r"``([^`]+)``")


def _tex(text: str) -> str:
    """One docstring line as LaTeX: roles and literals become \\texttt, rest escaped."""
    text = ROLE.sub(r"@\1@", text)
    text = LITERAL.sub(r"@\1@", text)
    text = text.replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    text = text.replace("_", r"\_")
    parts = text.split("@")
    # Odd indices are the literal runs marked out above.
    return "".join(p if i % 2 == 0 else f"\\texttt{{{p}}}" for i, p in enumerate(parts))


def _summary(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    return _tex(first.rstrip("."))


def _params(cls: type) -> str:
    """Constructor parameters as ``name=default``, in declaration order."""
    signature = inspect.signature(cls.__init__)
    shown = []
    for name, parameter in list(signature.parameters.items())[1:]:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue  # *args/**kwargs are not settable from a config
        if parameter.default is inspect.Parameter.empty:
            shown.append(name)
        else:
            shown.append(f"{name}={parameter.default!r}")
    joined = ", ".join(shown) or "--"
    return _tex(joined)


def _family(cls: type) -> str:
    """The module a component lives in: its family, for the catalogue's grouping."""
    return cls.__module__.rsplit(".", 1)[-1]


def rows(kind: str, by_family: bool = False) -> list[tuple[str, str, str, str]]:
    """Every registered component of one kind, by name — or by family then name."""
    entries = [
        (name, _family(cls), _params(cls), _summary(cls))
        for name, cls in sorted(_REGISTRY[kind].items())
    ]
    return sorted(entries, key=lambda row: (row[1], row[0])) if by_family else entries


def write(kind: str, grouped: bool) -> int:
    """Emit ``catalogue-<kind>.tex``, optionally sectioned by family."""
    entries = rows(kind, by_family=grouped)
    lines: list[str] = []
    last_family = None
    for name, family, params, summary in entries:
        if grouped and family != last_family:
            if last_family is not None:
                lines.append(r"\addlinespace")
            lines.append(rf"\multicolumn{{3}}{{l}}{{\itshape {family}}} \\")
            last_family = family
        lines.append(
            rf"\texttt{{{name.replace('_', chr(92) + '_')}}} & {summary} & {params} \\"
        )
    (GENERATED / f"catalogue-{kind}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(entries)


def _loc(*paths: str) -> int:
    """Physical lines of Python under each path, which is what the paper counts."""
    total = 0
    for name in paths:
        target = ROOT / name
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        total += sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)
    return total


def _tests() -> int:
    """What pytest actually collects — parametrised cases and all.

    Counting ``def test_`` undercounts by the parametrisations, and the number
    the paper quotes should be the one a reader gets from running the suite.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if match is None:
        raise SystemExit(f"could not read a test count from pytest:\n{result.stdout[-500:]}")
    return int(match.group(1))


def write_compact() -> None:
    """A names-only view, grouped by family: the catalogue at one sixth the length.

    The full table --- purpose and parameters per component --- is the published
    documentation's job; what a paper appendix owes a reader is the vocabulary,
    complete and demonstrably in step with the code.
    """
    lines = []
    for kind in ("stage", "source", "sink"):
        grouped: dict[str, list[str]] = {}
        for name, family, _params, _summary in rows(kind, by_family=True):
            grouped.setdefault(family if kind == "stage" else kind, []).append(name)
        for family, names in grouped.items():
            listed = ", ".join(f"\\texttt{{{n.replace('_', chr(92) + '_')}}}" for n in names)
            lines.append(rf"\textit{{{family}}} & {listed} \\")
        lines.append(r"\addlinespace")
    (GENERATED / "catalogue-compact.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    GENERATED.mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    for kind, grouped in (("stage", True), ("source", False), ("sink", False)):
        counts[kind] = write(kind, grouped)

    write_compact()
    families = sorted({family for _, family, _, _ in rows("stage")})
    macros = [
        rf"\newcommand{{\NumStages}}{{{counts['stage']}}}",
        rf"\newcommand{{\NumSources}}{{{counts['source']}}}",
        rf"\newcommand{{\NumSinks}}{{{counts['sink']}}}",
        rf"\newcommand{{\NumFamilies}}{{{len(families)}}}",
        rf"\newcommand{{\FamilyList}}{{{', '.join(f'\\texttt{{{f}}}' for f in families)}}}",
        rf"\newcommand{{\NumTests}}{{{_tests()}}}",
        rf"\newcommand{{\RepoLOC}}{{{_loc('segmentator', 'cli.py'):,}}}",
        rf"\newcommand{{\TestLOC}}{{{_loc('tests'):,}}}",
        rf"\newcommand{{\NumExamples}}{{{len(list((ROOT / 'examples').glob('*.yaml')))}}}",
    ]
    (GENERATED / "catalogue-numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"{counts} across {len(families)} families: {', '.join(families)}")


if __name__ == "__main__":
    main()
