"""Measure the shipped example configs on synthetic media, and emit LaTeX.

Three tables come out of here, all written to ``generated/`` so no number in the
paper is ever typed by hand:

* ``bench-throughput.tex`` — end-to-end ms/frame and fps per config per resolution
  (sinks included: this is what a batch run actually costs)
* ``bench-stagecost.tex`` — where one frame's budget goes, stage by stage, via
  :func:`segmentator.optimize.stage_costs` (sinks excluded)
* ``bench-sweep.dat`` — the same throughput numbers as a pgfplots table
* ``numbers.tex`` — ``\\newcommand`` macros for the figures quoted in prose

The configs measured are ``examples/*.yaml`` with two mechanical edits: the
source path points at synthetic media, and ``display`` sinks are dropped (they
need a window). The stage lists are untouched, which is the whole point — these
are the shipped recipes, not benchmark-shaped rewrites of them.

    uv run python docs/whitepaper/bench.py            # full run, minutes
    uv run python docs/whitepaper/bench.py --quick    # one repeat, short clips
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from segmentator import io, optimize  # noqa: F401  (io fills the registry)
from segmentator.pipeline import Pipeline, build

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
MEDIA = HERE / "media"
OUT = HERE / "out"
GENERATED = HERE / "generated"

VIDEO_CONFIGS = ("baseline", "gasvid1", "mog2_contours", "motion")
STILL_CONFIGS = ("structure", "texture")

# Frames per resolution. 720p is 12x the pixels of the base size, so holding the
# frame count fixed would spend most of the run on the largest size for no extra
# information — ms/frame is the comparable quantity, not total wall time.
FRAMES = {(320, 240): 900, (640, 480): 300, (1280, 720): 150}
REPEATS = 3

# The config whose per-stage breakdown the paper prints: the longest chain, and
# the one whose cost is least obvious by inspection.
COST_CONFIG = "motion"


def paper_config(name: str, width: int, height: int) -> dict[str, Any]:
    """An ``examples/*.yaml`` re-pointed at synthetic media, display sinks dropped."""
    cfg = yaml.safe_load((ROOT / "examples" / f"{name}.yaml").read_text(encoding="utf-8"))
    tag = f"{width}x{height}"
    still = cfg["source"]["type"] == "image"
    cfg["source"]["path"] = str(MEDIA / (f"geometry-{tag}.png" if still else f"plume-{tag}.mp4"))
    cfg["sinks"] = [s for s in cfg.get("sinks", []) if s["type"] != "display"]
    for sink in cfg["sinks"]:
        for key in ("path", "dir"):
            if key in sink:
                sink[key] = str(OUT / tag / name / Path(sink[key]).name)
    return cfg


def _run_once(cfg: dict[str, Any], max_frames: int | None) -> tuple[int, float]:
    """Frames processed and wall-clock seconds for one full pipeline run."""
    with Pipeline.from_config(cfg) as pipeline:
        began = time.perf_counter()
        frames = pipeline.run(max_frames=max_frames)
        return frames, time.perf_counter() - began


def throughput(quick: bool) -> list[dict[str, Any]]:
    """Median-of-N end-to-end timing for every config at every resolution."""
    repeats = 1 if quick else REPEATS
    rows = []
    for width, height in FRAMES:
        cap = FRAMES[(width, height)] // (10 if quick else 1)
        for name in (*VIDEO_CONFIGS, *STILL_CONFIGS):
            cfg = paper_config(name, width, height)
            still = cfg["source"]["type"] == "image"
            # A still is one frame, and one frame is not a measurement: run the
            # whole pipeline repeatedly instead and divide.
            passes = (20 if not quick else 3) if still else 1
            samples = []
            for _ in range(repeats):
                total = 0.0
                for _ in range(passes):
                    frames, seconds = _run_once(cfg, None if still else cap)
                    total += seconds
                samples.append(total / (passes * frames))
            per_frame = statistics.median(samples)
            rows.append(
                {
                    "config": name,
                    "width": width,
                    "height": height,
                    "frames": frames if still else cap,
                    "ms": per_frame * 1000,
                    "fps": 1.0 / per_frame,
                }
            )
            print(f"  {name:<14} {width}x{height:<5} {per_frame * 1000:7.2f} ms  {1 / per_frame:7.1f} fps")
    return rows


def stage_costs(name: str, width: int, height: int) -> list[tuple[str, float]]:
    """Per-stage ms for one config, reusing the optimizer's interleaved timer."""
    cfg = paper_config(name, width, height)
    specs = [dict(spec) for spec in cfg["stages"]]
    source = build("source", cfg["source"])
    frames = optimize.sample_frames(source, contiguous=optimize.needs_contiguous(specs))
    source.close()
    costs = optimize.stage_costs(specs, frames)
    return [(spec["type"], cost) for spec, cost in zip(specs, costs)]


# --------------------------------------------------------------------------- #
# LaTeX emission
# --------------------------------------------------------------------------- #


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_throughput(rows: list[dict[str, Any]]) -> None:
    lines = []
    for name in (*VIDEO_CONFIGS, *STILL_CONFIGS):
        for index, row in enumerate([r for r in rows if r["config"] == name]):
            label = f"\\texttt{{{_tex_escape(name)}}}" if index == 0 else ""
            lines.append(
                f"{label} & ${row['width']}\\times{row['height']}$ & {row['frames']} "
                f"& {row['ms']:.2f} & {row['fps']:.1f} \\\\"
            )
        lines.append(r"\addlinespace")
    (GENERATED / "bench-throughput.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage_costs(costs: list[tuple[str, float]]) -> None:
    total = sum(cost for _, cost in costs)
    lines = [
        f"\\texttt{{{_tex_escape(kind)}}} & {cost:.2f} & {100 * cost / total:.1f} \\\\"
        for kind, cost in costs
    ]
    lines.append(r"\midrule")
    lines.append(f"total & {total:.2f} & 100.0 \\\\")
    (GENERATED / "bench-stagecost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# pgfplots column names, one per recipe. Kept free of underscores so a column
# can be named in a `y=` key without escaping.
COLUMNS = {
    "baseline": "baseline",
    "gasvid1": "gasvid",
    "mog2_contours": "mog",
    "motion": "motion",
    "structure": "structure",
    "texture": "texture",
}


def write_sweep(rows: list[dict[str, Any]]) -> None:
    """pgfplots table in wide form: one row per resolution, one column per recipe."""
    names = list(COLUMNS)
    lines = [" ".join(["pixels", *(COLUMNS[name] for name in names)])]
    for width, height in FRAMES:
        cells = []
        for name in names:
            row = next(r for r in rows if r["config"] == name and r["width"] == width)
            cells.append(f"{row['fps']:.3f}")
        lines.append(" ".join([str(width * height), *cells]))
    (GENERATED / "bench-sweep.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_numbers(rows: list[dict[str, Any]], costs: list[tuple[str, float]]) -> None:
    """The handful of figures quoted in prose, as macros."""

    def at(config: str, width: int) -> dict[str, Any]:
        return next(r for r in rows if r["config"] == config and r["width"] == width)

    # Only the video recipes are judged against a frame rate: `structure` and
    # `texture` describe a single still, so "fps" there is throughput per image,
    # not a rate anything has to keep up with.
    slowest = min(
        (r for r in rows if r["width"] == 320 and r["config"] in VIDEO_CONFIGS),
        key=lambda r: r["fps"],
    )
    dearest = max(costs, key=lambda item: item[1])
    total_cost = sum(cost for _, cost in costs)
    macros = {
        "BenchBaselineFps": f"{at('baseline', 320)['fps']:.0f}",
        "BenchMotionFps": f"{at('motion', 320)['fps']:.0f}",
        "BenchMotionFpsVGA": f"{at('motion', 640)['fps']:.0f}",
        "BenchMotionFpsHD": f"{at('motion', 1280)['fps']:.0f}",
        "BenchSlowestVideo": _tex_escape(slowest["config"]),
        "BenchSlowestVideoFps": f"{slowest['fps']:.0f}",
        "BenchTextureMs": f"{at('texture', 320)['ms']:.0f}",
        "BenchDearestStage": _tex_escape(dearest[0]),
        "BenchDearestMs": f"{dearest[1]:.1f}",
        "BenchDearestShare": f"{100 * dearest[1] / total_cost:.0f}",
        "BenchCostConfig": _tex_escape(COST_CONFIG),
    }
    body = "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items())
    (GENERATED / "numbers.tex").write_text(body + "\n", encoding="utf-8")


def write_configs() -> None:
    """Dump the re-pointed configs so a reader can see exactly what was measured."""
    target = HERE / "configs"
    target.mkdir(exist_ok=True)
    for name in (*VIDEO_CONFIGS, *STILL_CONFIGS):
        cfg = paper_config(name, 320, 240)
        header = (
            f"# Generated by bench.py from examples/{name}.yaml.\n"
            "# Only the source path and the display sinks differ; the stage list is untouched.\n"
        )
        text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        (target / f"{name}.yaml").write_text(header + text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one repeat, short clips")
    args = parser.parse_args()

    GENERATED.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    print("throughput:")
    rows = throughput(args.quick)
    print(f"per-stage cost for {COST_CONFIG}:")
    costs = stage_costs(COST_CONFIG, 320, 240)
    for kind, cost in costs:
        print(f"  {kind:<18} {cost:6.2f} ms")

    write_throughput(rows)
    write_stage_costs(costs)
    write_sweep(rows)
    write_numbers(rows, costs)
    write_configs()
    print(f"wrote {GENERATED}/bench-*.tex, numbers.tex, and configs/")


if __name__ == "__main__":
    main()
