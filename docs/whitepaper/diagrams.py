"""Restyle the repo's documentation diagrams for print and convert them to PDF.

The geometry, the labels and the colours are the site's own — these diagrams are
already correct and already the paper's argument in picture form. The only edit
is typographic: the web font stack becomes the document's, so a diagram on the
page reads as part of the page rather than as a screenshot of a website.

    uv run --with cairosvg python docs/whitepaper/diagrams.py

Libertinus has to be visible to fontconfig for the text to come out right; the
Makefile's ``fonts`` target copies it out of the TeX Live image into ~/.fonts.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE.parent / "assets"
FIGURES = HERE / "figures"

DIAGRAMS = ("pipeline-anatomy", "stage-families", "config-branching")

# The document's faces, in the same shape the source stacks have: a named face
# first, a generic that always resolves last.
SUBSTITUTIONS = {
    "system-ui, -apple-system, Segoe UI, sans-serif": "Libertinus Sans, sans-serif",
    "ui-monospace, SFMono-Regular, Menlo, monospace": "Libertinus Mono, monospace",
}


def restyle(name: str) -> Path:
    """Write the print version of one diagram, and return its path."""
    svg = (ASSETS / f"{name}.svg").read_text(encoding="utf-8")
    for web, print_face in SUBSTITUTIONS.items():
        svg = svg.replace(web, print_face)
    target = FIGURES / f"{name}.svg"
    target.write_text(svg, encoding="utf-8")
    return target


def to_pdf(svg: Path) -> Path:
    """Convert to PDF, so the diagram stays vector all the way into the document."""
    import cairosvg

    pdf = svg.with_suffix(".pdf")
    cairosvg.svg2pdf(url=str(svg), write_to=str(pdf))
    return pdf


def check_fonts() -> None:
    """Warn rather than fail: a missing face substitutes, it does not crash."""
    if shutil.which("fc-list") is None:
        return
    listed = subprocess.run(["fc-list"], capture_output=True, text=True).stdout
    if "Libertinus" not in listed:
        print("warning: Libertinus is not installed — run `make fonts` or the diagram")
        print("         text will be substituted and will not match the document")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    check_fonts()
    for name in DIAGRAMS:
        pdf = to_pdf(restyle(name))
        print(f"{pdf.relative_to(HERE)}")


if __name__ == "__main__":
    main()
