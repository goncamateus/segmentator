# Installation

## From source (developing on it, or adding a stage)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
git clone https://github.com/goncamateus/segmentator.git
cd segmentator
uv sync                 # CLI only
uv run segmentator configs/baseline.yaml
```

The `ffmpeg` sink shells out to the `ffmpeg` binary — install it separately
(`apt install ffmpeg`, `brew install ffmpeg`) and make sure it is on `PATH`. The
`display` sink and everything else in `segmentator/ops/` need nothing beyond the
Python dependencies `uv sync` already installed.

The editor is behind an extra, so a headless install — a batch box, CI — never
pulls PyQt6 in:

```bash
uv sync --extra gui
uv run segmentator-gui configs/motion.yaml
```

Run the test suite with:

```bash
uv run pytest
```

## Prebuilt app (the editor only)

Every [tagged release](https://github.com/goncamateus/segmentator/releases) ships
a standalone build of the editor for Linux and macOS — no Python, no `uv`, no
dependencies to install. It bundles the same code as `segmentator-gui`; the CLI
itself is still run from source (above), since a batch tool has no use for a
double-clickable app.

**Linux — AppImage.**

```bash
chmod +x segmentator-*.AppImage
./segmentator-*.AppImage configs/motion.yaml
```

No installation step; it runs in place.

**macOS — dmg.**

Open the `.dmg`, drag `Segmentator.app` into `Applications`. The build is not
notarised, so Gatekeeper blocks the first launch — right-click the app and choose
*Open*, then confirm.

**Windows.** No prebuilt installer. Build it from source (above), then package it
yourself with `uv run --group build pyinstaller --noconfirm segmentator.spec` —
see [segmentator.spec](https://github.com/goncamateus/segmentator/blob/main/segmentator.spec)
and the `packaging/` scripts for how Linux and macOS do it.
