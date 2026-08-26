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
from pathlib import Path

from segmentator import io, stages  # noqa: F401  (importing them fills the registry)
from segmentator.pipeline import _REGISTRY

HERE = Path(__file__).parent
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


def main() -> None:
    GENERATED.mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    for kind, grouped in (("stage", True), ("source", False), ("sink", False)):
        counts[kind] = write(kind, grouped)

    families = sorted({family for _, family, _, _ in rows("stage")})
    macros = [
        rf"\newcommand{{\NumStages}}{{{counts['stage']}}}",
        rf"\newcommand{{\NumSources}}{{{counts['source']}}}",
        rf"\newcommand{{\NumSinks}}{{{counts['sink']}}}",
        rf"\newcommand{{\NumFamilies}}{{{len(families)}}}",
        rf"\newcommand{{\FamilyList}}{{{', '.join(f'\\texttt{{{f}}}' for f in families)}}}",
    ]
    (GENERATED / "catalogue-numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"{counts} across {len(families)} families: {', '.join(families)}")


if __name__ == "__main__":
    main()
