# Segmentator

[![Release](https://img.shields.io/github/v/release/goncamateus/segmentator?sort=semver)](https://github.com/goncamateus/segmentator/releases)
[![Docs](https://img.shields.io/github/actions/workflow/status/goncamateus/segmentator/deploy-docs.yml?label=docs)](https://goncamateus.github.io/segmentator/)
[![License: MIT](https://img.shields.io/github/license/goncamateus/segmentator)](LICENSE)

Configuration-driven image and video segmentation. A pipeline is described in
YAML, so trying a different preprocessing chain, detector or background model is
a config edit, not a code edit.

It started as a way to pull a gas plume out of thermal video frame by frame;
generalizing that one pipeline into a `source → stages → sinks` config format is
the whole shape of the project.

## Quickstart

```bash
uv sync
uv run segmentator examples/baseline.yaml
uv run segmentator examples/structure.yaml
uv run segmentator examples/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
```

`examples/*.yaml` are small runnable configs tracked in the repo — start from
one of these rather than writing YAML from scratch. `--video` and `--output`
override the source path and the first ffmpeg sink path for one run.
`inputs/`, like `configs/`, is gitignored: point these at your own footage.

There is also an editor, which authors these configs with a live preview of every
stage and every sink:

```bash
uv sync --extra gui
uv run segmentator-gui examples/motion.yaml
```

## See it run

<p align="center">
    <img src="docs/assets/demo-montage.png" alt="One source frame next to three stage-family outputs on it: structure (edges/lines/contours), texture (HOG), and motion (tracked objects)">
</p>

One source frame run through three of [the stage families](docs/en/stages.md):
structure, texture and motion.

Part of the [goncanalyser](https://github.com/goncamateus/goncanalyser) suite.
goncanalyser is the Qt workspace where you *tune* an operator chain by hand;
segmentator is the headless engine that *runs* the tuned recipe over a batch. Every
operator goncanalyser exposes exists here as a stage, and the two agree
numerically — see [Parity](#parity).

## Documentation

Published at [goncamateus.github.io/segmentator](https://goncamateus.github.io/segmentator/), in
English and pt-BR.

| | |
|---|---|
| [docs/en/installation.md](docs/en/installation.md) | `uv sync`, the `gui` extra, and the prebuilt AppImage / dmg |
| [docs/en/configuration.md](docs/en/configuration.md) | the config format: `Ctx`, taps, `input:`, branching with `select` |
| [docs/en/stages.md](docs/en/stages.md) | every source, stage and sink, and what its parameters mean |
| [docs/en/gui.md](docs/en/gui.md) | the editor, and every operation it allows |

## Architecture

`source → stages → sinks`. Stages are an ordered list applied to every frame —
nothing branches on runtime state at execution time, so the list order *is* the
composition.

```python
@dataclass
class Ctx:
    image: np.ndarray   # working array; stages rebind it
    source: np.ndarray  # original BGR frame, untouched
    index: int
    store: dict         # side channel within one frame
    taps: dict          # named stages' outputs, for sinks to pick from
    metrics: dict       # per-frame scalars, what the json sink writes
    rows: dict          # per-object detail, what the csv and crops sinks write

class Stage(Protocol):
    def apply(self, ctx: Ctx) -> None: ...

class Sink(Protocol):
    def write(self, ctx: Ctx) -> bool: ...   # False stops the run
    def close(self) -> None: ...
```

Components register themselves by name (`@register("stage", "median_blur")`), and
`build()` turns a `{type: ..., **params}` mapping into an instance. An unknown name
raises with the list of known ones — the registry is what drives the config schema,
the error messages and the editor's palette and forms alike.

| File | Contents |
|---|---|
| [cli.py](cli.py) | CLI |
| [segmentator/pipeline.py](segmentator/pipeline.py) | `Ctx`, protocols, registry, `Pipeline` |
| [segmentator/io.py](segmentator/io.py) | sources and sinks |
| [segmentator/ops/](segmentator/ops/) | pure operators, no stage protocol, no config object |
| [segmentator/stages/](segmentator/stages/) | the registered stages, by family |
| [segmentator/background_model.py](segmentator/background_model.py) | fixed mean-of-N background |
| [segmentator/video_writer.py](segmentator/video_writer.py) | ffmpeg/libx264 pipe |
| [segmentator/gui/](segmentator/gui/) | the PyQt6 editor — optional, `--extra gui` |

## Parity

Every operator is ported from goncanalyser's `features/*` — the function bodies,
not a reimplementation — with each `Settings` field turned into a constructor
argument on the stage that uses it. On the same frame with the same parameters
the two produce identical output: `edge_px`, corner and keypoint counts, HOG
vectors, LBP codes, histogram plots, contour rows and Hough rows all match
exactly, across the two repos' different OpenCV builds.

Not ported, on purpose: goncanalyser's deferred-draw-callable overlay collector
(`Result.ops`) — a linear chain composes overlays through ordering instead — and
its `MotionState` re-analysis/seek-backwards guards, which a single forward
batch pass can't trigger (only the shape guard survives, on the stages
themselves) — both rules come back the moment there is a GUI, and
[docs/en/gui.md](docs/en/gui.md#moving-through-the-video) is where they now
live. `goncanalyser/dataset/` (COCO, rosbag, Optuna search, dataset stats) is
dataset tooling, not image processing, and stays out too.

## Tests

```bash
uv run pytest
```

`tests/test_stages.py` holds goncanalyser's own `_demo()` assertions, re-pointed at
the stages — ported rather than invented, so a passing suite means the two repos
agree on numbers, not merely that both run. `tests/test_gui.py` covers the editor
and runs headless under the offscreen Qt platform. `tests/test_examples.py` drives
every `examples/*.yaml` through the CLI end to end, as a drift guard against the
example configs and the stage registry falling out of sync.

## Packaging

`segmentator.spec` is one PyInstaller recipe, and packages the editor (`segmentator-gui`)
only — the CLI is a `uv run segmentator` tool, not something a double-clickable app has a
use for. Linux and macOS each have a short script that turns the bundle into an installer.

```bash
uv sync --no-dev --extra gui --group build

# Linux: bundle first, then wrap it.
uv run --no-dev --extra gui --group build pyinstaller --noconfirm segmentator.spec
bash packaging/linux/build-appimage.sh   # -> dist/segmentator-VERSION-ARCH.AppImage

# macOS is one step, not two: the spec's BUNDLE needs the .icns, and this script is what
# generates it, so it calls PyInstaller itself.
bash packaging/macos/build-dmg.sh        # -> dist/segmentator-VERSION-arm64.dmg
```

`--extra gui --group build` belongs on every `uv run` in a build, including the ones that
only read the version — without `--group build` uv re-syncs and drops PyInstaller back out
of the environment, and without `--extra gui` there is no PyQt6 to bundle.

An installer can only be built on the platform it targets, which is why
`.github/workflows/release.yml` exists: on tag push it builds the AppImage and the dmg and
attaches both to a GitHub release. Windows is not in that matrix — build it by hand from the
same spec if you need it.

One packaging constraint worth knowing before touching dependencies: the GUI depends on
plain `opencv-python`, not `opencv-python-headless` — the opposite of goncanalyser's choice,
and for the opposite reason. `DisplaySink` needs the full wheel to open an OpenCV window in
a batch run; `segmentator/gui/main.py` works around the resulting Qt-plugin clash by popping
`QT_QPA_PLATFORM_PLUGIN_PATH` before PyQt6 loads (see [docs/en/gui.md](docs/en/gui.md#troubleshooting)).
That workaround has to survive into the frozen bundle too, which is why it lives in the entry
point PyInstaller packages rather than in a dev-only setup step.

`segmentator/gui/assets/icon.png` is the app icon, and 1024×1024 is the size to keep it at:
the macOS `.icns` ladder is derived from it at build time by `build-dmg.sh` — every size
down to 16×16 — and the Linux AppImage uses it as-is. It sits inside the package rather
than in `packaging/` because it is also read at runtime: the launcher window draws it, and
neither a wheel install nor the frozen bundle carries `packaging/`.

## License

[MIT](LICENSE) © goncamateus.
