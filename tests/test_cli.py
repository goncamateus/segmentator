"""Direct tests for the CLI entry point in cli.py.

Tests call `cli.load_config` and `cli.main` directly (in-process), capturing
raised exceptions rather than spawning a subprocess. `main` reads its
arguments from `sys.argv`, so tests drive it by monkeypatching that.

    uv run pytest tests/test_cli.py
"""

import sys

import cv2
import numpy as np
import pytest
import yaml

import cli


def _write_config(tmp_path, doc):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _write_image(path, value=0, size=(4, 4)):
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), value, np.uint8))


# --------------------------------------------------------------------------- #
# load_config: overrides land on the loaded config
# --------------------------------------------------------------------------- #


def test_load_config_video_override_replaces_source_path(tmp_path):
    config_path = _write_config(
        tmp_path,
        {"source": {"type": "video", "path": "original.mp4"}, "sinks": []},
    )
    video = tmp_path / "override.mp4"

    cfg = cli.load_config(config_path, video, None)

    assert cfg["source"]["path"] == str(video)


def test_load_config_output_override_replaces_ffmpeg_sink_path(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "source": {"type": "video", "path": "in.mp4"},
            "sinks": [
                {"type": "ffmpeg", "path": "original_out.mp4"},
                {"type": "csv", "dir": "out/"},
            ],
        },
    )
    output = tmp_path / "override_out.mp4"

    cfg = cli.load_config(config_path, None, output)

    ffmpeg_sinks = [sink for sink in cfg["sinks"] if sink["type"] == "ffmpeg"]
    assert ffmpeg_sinks[0]["path"] == str(output)
    assert cfg["sinks"][1]["dir"] == "out/", "a sink of another type is left untouched"


def test_load_config_without_overrides_leaves_config_untouched(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "source": {"type": "video", "path": "in.mp4"},
            "sinks": [{"type": "ffmpeg", "path": "out.mp4"}],
        },
    )

    cfg = cli.load_config(config_path, None, None)

    assert cfg["source"]["path"] == "in.mp4"
    assert cfg["sinks"][0]["path"] == "out.mp4"


# --------------------------------------------------------------------------- #
# load_config: the exit path when --output has nothing to retarget
# --------------------------------------------------------------------------- #


def test_load_config_output_override_without_ffmpeg_sink_exits(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "source": {"type": "video", "path": "in.mp4"},
            "sinks": [{"type": "csv", "dir": "out/"}],
        },
    )

    with pytest.raises(SystemExit, match="no 'ffmpeg' sink"):
        cli.load_config(config_path, None, tmp_path / "override_out.mp4")


def test_load_config_output_override_with_no_sinks_at_all_exits(tmp_path):
    config_path = _write_config(tmp_path, {"source": {"type": "video", "path": "in.mp4"}})

    with pytest.raises(SystemExit, match="no 'ffmpeg' sink"):
        cli.load_config(config_path, None, tmp_path / "o.mp4")


# --------------------------------------------------------------------------- #
# main(): drives argparse + load_config + Pipeline in-process
# --------------------------------------------------------------------------- #


def test_main_applies_video_override_and_runs_the_pipeline(tmp_path, monkeypatch, capsys):
    """The configured source path does not exist, so the run can only succeed
    if `--video` actually replaced it before the source was built — a broken
    override would fail this test with an OSError, not just a wrong count.
    """
    real_image = tmp_path / "real.png"
    _write_image(real_image, value=9)

    config_path = _write_config(
        tmp_path,
        {
            "name": "demo",
            "source": {"type": "image", "path": str(tmp_path / "missing.png")},
            "sinks": [],
        },
    )

    monkeypatch.setattr(sys, "argv", ["segmentator", str(config_path), "--video", str(real_image)])
    cli.main()

    out = capsys.readouterr().out
    assert "demo: 1 frames" in out


def test_main_exits_when_output_override_has_no_matching_sink(tmp_path, monkeypatch):
    image_path = tmp_path / "still.png"
    _write_image(image_path)

    config_path = _write_config(
        tmp_path,
        {
            "source": {"type": "image", "path": str(image_path)},
            "sinks": [{"type": "csv", "dir": str(tmp_path / "csvout")}],
        },
    )

    monkeypatch.setattr(
        sys, "argv", ["segmentator", str(config_path), "--output", str(tmp_path / "o.mp4")]
    )

    with pytest.raises(SystemExit, match="no 'ffmpeg' sink"):
        cli.main()
