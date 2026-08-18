"""Run a video-processing pipeline described by a YAML config.

    python gasvid.py configs/baseline.yaml
    python gasvid.py configs/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
"""

import argparse
from pathlib import Path

import yaml

from core.pipeline import Pipeline


def load_config(path: Path, video: Path | None, output: Path | None) -> dict:
    """Read the config and apply the two per-run CLI overrides."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if video is not None:
        cfg["source"]["path"] = str(video)
    if output is not None:
        sinks = [s for s in cfg.get("sinks", []) if s.get("type") == "ffmpeg"]
        if not sinks:
            raise SystemExit(f"--output given but {path} has no 'ffmpeg' sink to retarget")
        sinks[0]["path"] = str(output)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="pipeline YAML")
    parser.add_argument("--video", type=Path, help="override the source path")
    parser.add_argument("--output", type=Path, help="override the ffmpeg sink path")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    args = parser.parse_args()

    cfg = load_config(args.config, args.video, args.output)
    with Pipeline.from_config(cfg) as pipeline:
        processed = pipeline.run(max_frames=args.max_frames)
    print(f"{cfg.get('name', args.config.stem)}: {processed} frames")


if __name__ == "__main__":
    main()
