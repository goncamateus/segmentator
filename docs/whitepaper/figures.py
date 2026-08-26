"""Render the paper's pipeline-output panels and the optimizer's own findings.

Every panel here is a real tap out of a real run of a shipped config over the
synthetic media — nothing is drawn by hand and nothing is retouched. Panels are
written one file per picture rather than pre-composed into a strip, so the
arrangement and the labelling stay in LaTeX where the document's typography is.

    uv run python docs/whitepaper/figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from bench import GENERATED, HERE, paper_config
from segmentator import optimize
from segmentator.pipeline import Ctx, Pipeline, build, frame_for

FIGURES = HERE / "figures"

# The frame the plume panels are taken from: far enough in that the background
# model is fitted (60 frames) and the plume has developed.
PLUME_FRAME = 400


def capture(cfg: dict[str, Any], index: int = 0) -> Ctx:
    """Run the chain up to ``index`` and hand back that frame's context.

    Frames are walked in order rather than seeked to, because a chain holding a
    background model has to see the history its output depends on.
    """
    pipeline = Pipeline(build("source", cfg["source"]), [], [])
    pipeline.stages = Pipeline.from_config({**cfg, "sinks": []}).stages
    ctx = None
    for position, frame in enumerate(pipeline.source):
        ctx = Ctx(image=frame, source=frame, index=position)
        pipeline.apply(ctx)
        if position >= index:
            break
    pipeline.source.close()
    assert ctx is not None, "source produced no frames"
    return ctx


def save(ctx: Ctx, key: str, name: str) -> None:
    """Write one named image out of a captured context."""
    image = frame_for(ctx, key)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(FIGURES / f"{name}.png"), image)


def plume_panels() -> None:
    """Figure 6: what the background/motion chains see in the plume clip."""
    baseline = capture(paper_config("baseline", 320, 240), PLUME_FRAME)
    save(baseline, "source", "plume-source")
    save(baseline, "mask", "plume-roi")
    save(baseline, "image", "plume-subtracted")

    motion = capture(paper_config("motion", 320, 240), PLUME_FRAME)
    save(motion, "mask", "plume-motion-mask")
    save(motion, "image", "plume-objects")
    print(f"plume: {motion.metrics}")


def geometry_panels() -> None:
    """Figure 7: the structure and texture families on the geometry still."""
    structure = capture(paper_config("structure", 320, 240))
    save(structure, "edges", "geom-edges")
    save(structure, "lines", "geom-lines")
    save(structure, "image", "geom-contours")

    texture = capture(paper_config("texture", 320, 240))
    save(texture, "hog", "geom-hog")
    save(texture, "lbp", "geom-lbp")
    print(f"structure: {structure.metrics}\ntexture: {texture.metrics}")


# --------------------------------------------------------------------------- #
# Figure 5: the optimizer's report on a deliberately ratcheted chain
# --------------------------------------------------------------------------- #

BASIS_GLOSS = {
    "identity": "proved: parameterised into a passthrough",
    "dataflow": "proved: nothing downstream reads it",
    "exhaustive": "proved: over all 256 input values",
    "sampled": "not proved: no counterexample found",
}


def optimizer_report() -> None:
    """Run ``analyse`` on the ratcheted config and emit both halves of its answer."""
    path = HERE / "configs" / "optimize-demo.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = [dict(spec) for spec in cfg["stages"]]

    source = build("source", cfg["source"])
    frames = optimize.sample_frames(source, contiguous=optimize.needs_contiguous(specs))
    source.close()
    findings = optimize.analyse(cfg, frames)

    rows = []
    for finding in findings:
        where = ", ".join(str(position) for position in finding.positions)
        proof = "proved" if finding.proven else "sampled"
        rows.append(
            f"{where} & \\texttt{{{finding.label.replace('_', chr(92) + '_')}}} & {proof} "
            f"& {finding.detail} & {finding.saved_ms:.2f} \\\\"
        )
    (GENERATED / "optimize-findings.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    touched = {position for finding in findings for position in finding.positions}
    kept = [
        f"\\texttt{{{spec['type'].replace('_', chr(92) + '_')}}}"
        for position, spec in enumerate(specs)
        if position not in touched
    ]
    macros = [
        rf"\newcommand{{\OptStages}}{{{len(specs)}}}",
        rf"\newcommand{{\OptFindings}}{{{len(findings)}}}",
        rf"\newcommand{{\OptProved}}{{{sum(1 for f in findings if f.proven)}}}",
        rf"\newcommand{{\OptSampled}}{{{sum(1 for f in findings if not f.proven)}}}",
        rf"\newcommand{{\OptKept}}{{{len(kept)}}}",
        rf"\newcommand{{\OptFrames}}{{{len(frames)}}}",
        rf"\newcommand{{\OptKeptList}}{{{', '.join(kept)}}}",
    ]
    (GENERATED / "optimize-numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    print(f"optimizer: {len(findings)} findings over {len(specs)} stages, {len(frames)} frames")
    for finding in findings:
        print(f"  {finding.basis:<11} {finding.label:<40} {finding.saved_ms:6.2f} ms  {finding.detail}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    GENERATED.mkdir(exist_ok=True)
    plume_panels()
    geometry_panels()
    optimizer_report()


if __name__ == "__main__":
    main()
