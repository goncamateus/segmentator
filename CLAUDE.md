# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # base install
uv sync --extra gui               # + PyQt6 editor
uv run pytest                     # full suite
uv run pytest tests/test_stages.py::test_median_blur   # single test
uv run pytest tests/test_gui.py   # editor tests, run headless via offscreen Qt platform

uv run segmentator configs/baseline.yaml
uv run segmentator configs/structure.yaml --video inputs/gasvid1.mp4 --max-frames 300
uv run segmentator-gui configs/motion.yaml   # requires --extra gui
```

No configured linter/formatter — none is invoked in `.github/workflows/`; don't assume ruff/black are wired in.

## Architecture

Pipeline: `source → stages → sinks`, driven entirely by YAML (see `configs/*.yaml`). Nothing branches on runtime state at execution time — the linear stage list *is* the composition order (e.g. `canny` then `harris(draw_on: image)` draws corners onto the edge map because of ordering, not a flag).

- [segmentator/pipeline.py](segmentator/pipeline.py) — `Ctx` (per-frame state), `Stage`/`Sink` protocols, the `@register("stage"/"sink", name)` registry, `build()`, `Pipeline.run`. Read this file's docstrings first; they're the actual spec for `image`/`source`/`store`/`taps`/`metrics`/`rows`.
- [segmentator/io.py](segmentator/io.py) — concrete sources and sinks.
- [segmentator/ops/](segmentator/ops/) — pure functions (no `Stage` protocol, no config object), organized by family: color, common, keypoints, motion, structure, texture.
- [segmentator/stages/](segmentator/stages/) — registered `Stage` wrappers around `ops/`, same family split, plus `mask.py` and `preprocess.py`.
- [segmentator/optimize.py](segmentator/optimize.py) — chain simplification: a dataflow walk plus a sampled oracle, driving the editor's *Pipeline → Optimize*. Headless and Qt-free; `WRITES`/`READS` here are the full store/metrics/rows tables `spec.PUBLISHES` is the image-valued subset of. Read its module docstring before touching it — the split between *provable* and *sampled* findings is the whole design, not a detail.
- [segmentator/background_model.py](segmentator/background_model.py) — fixed mean-of-N background; `BackgroundModel.ready` is the only real state machine in the codebase (warmup vs steady).
- [segmentator/video_writer.py](segmentator/video_writer.py) — ffmpeg/libx264 pipe sink.
- [segmentator/gui/](segmentator/gui/) — PyQt6 editor (optional, `--extra gui`). Reads the same registry/`build()` as the CLI to generate its palette and forms — nothing about stages is hardcoded twice.
- [cli.py](cli.py) — CLI entry point (`segmentator = "cli:main"`).

Components self-register by name; `build()` turns `{type: ..., **params}` into an instance. Unknown `type` raises listing known names — the registry is the source of truth for the config schema, the editor's palette, and error messages simultaneously.

### Parity with goncanalyser

Every stage/op is ported from [goncanalyser](https://github.com/goncamateus/goncanalyser)'s `features/*` — the function bodies, not a reimplementation — with each `Settings` field becoming a constructor argument. `tests/test_stages.py` is goncanalyser's own `_demo()` assertions re-pointed at these stages, so a passing suite means numeric agreement with goncanalyser, not just "it runs." When porting/touching a stage, preserve exact numeric output (`edge_px`, contour/keypoint counts, HOG vectors, LBP codes, etc.) — this is a hard invariant, not a style preference.

Deliberately **not** ported from goncanalyser: the deferred-draw-callable overlay collector (`Result.ops`) — a linear chain doesn't need it, ordering already composes overlays — and `goncanalyser/dataset/` (COCO, rosbag, Optuna search, dataset stats), which is dataset tooling, not image processing. `MotionState`'s re-analysis/seek-backwards guards also don't apply to a batch `Pipeline.run` (single forward pass over monotonic `ctx.index`); only the shape guard (resized frame invalidates a flow model) survives on the stages themselves.

### Packaging gotcha

The GUI extra depends on plain `opencv-python` (not `-headless`) because `DisplaySink` needs the full wheel to open an OpenCV window even in a batch run. This collides with PyQt6's Qt plugins; `segmentator/gui/main.py` works around it by popping `QT_QPA_PLATFORM_PLUGIN_PATH` before PyQt6 loads. That workaround must ship in the frozen PyInstaller bundle too — don't move it into a dev-only setup step. Build commands need both `--extra gui --group build`; dropping either drops PyQt6 or PyInstaller from the synced env. Full build details are in [README.md](README.md#packaging).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`goncamateus/segmentator`), via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical labels used as-is. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.
