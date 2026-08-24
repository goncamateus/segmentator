"""Drift guard: every shipped example config must still run to completion.

`examples/*.yaml` are the configs the README and docs/en quickstarts point
readers at (restored in issue #23 after `3812eee` stopped tracking `configs/`
— see `docs/adr/0001-configs-gitignored-examples-tracked.md`). Nothing else
catches a stage/registry change that breaks one of them, so this is that
catch, in the same spirit as `test_docs.py`'s stage-count guard.

Each config is driven through `cli.main()` — the same in-process entry point
`test_cli.py` exercises — against a small synthetic frame or clip standing in
for the real (untracked, local-only) footage under `inputs/`, with every
source and sink path retargeted under `tmp_path` first so a run here never
touches the real `inputs/`/`outputs/` directories.

    uv run pytest tests/test_examples.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import cli

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))

MAX_FRAMES = 5
FRAME_SIZE = (64, 64)  # even dims: required for the ffmpeg sinks' yuv420p output


def _sample_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, FRAME_SIZE)
    rng = np.random.default_rng(0)
    for index in range(MAX_FRAMES + 2):
        frame = rng.integers(0, 255, (*FRAME_SIZE[::-1], 3), dtype=np.uint8)
        # A moving bright square gives the motion stages something to detect.
        offset = index * 4
        cv2.rectangle(frame, (offset, offset), (offset + 10, offset + 10), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _sample_image(path: Path) -> None:
    frame = np.zeros((*FRAME_SIZE[::-1], 3), np.uint8)
    cv2.rectangle(frame, (16, 16), (48, 48), (255, 255, 255), -1)
    cv2.imwrite(str(path), frame)


def _retarget(cfg: dict, tmp_path: Path) -> None:
    """Rewrite the source path and every sink's output path/dir under tmp_path."""
    source = cfg["source"]
    fixture = tmp_path / f"source{'.mp4' if source['type'] == 'video' else '.png'}"
    if source["type"] == "video":
        _sample_video(fixture)
    else:
        _sample_image(fixture)
    source["path"] = str(fixture)

    # `display` opens a real window via cv2.imshow — this build's OpenCV has no
    # offscreen fallback (see test_sinks.py), so it's dropped for a headless run.
    cfg["sinks"] = [s for s in cfg.get("sinks", []) if s.get("type") != "display"]
    for sink in cfg["sinks"]:
        for key in ("path", "dir"):
            if key in sink:
                sink[key] = str(tmp_path / "out" / Path(sink[key]))


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_example_config_runs_to_completion(config_path, tmp_path, monkeypatch, capsys):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _retarget(cfg, tmp_path)

    run_config = tmp_path / "config.yaml"
    run_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["segmentator", str(run_config), "--max-frames", str(MAX_FRAMES)]
    )
    cli.main()

    out = capsys.readouterr().out
    assert f"{cfg['name']}:" in out


def test_examples_directory_is_not_empty():
    """Guards against the parametrization above silently collecting zero cases."""
    assert EXAMPLE_CONFIGS, f"no *.yaml files found under {EXAMPLES_DIR}"
