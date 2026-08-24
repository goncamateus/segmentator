"""Direct tests for the sinks in segmentator/io.py.

Each sink is constructed directly (no Pipeline, no YAML) and driven through its
own `write(ctx)` / `close()` protocol, the way
`test_display_sink_closes_without_ever_opening_a_window` in test_pipeline.py
already does for DisplaySink. Assertions land on observable output — files on
disk, their bytes/rows/pixels — not on how a sink gets there.

    uv run pytest
"""

import csv
import json

import cv2
import numpy as np

from segmentator.io import CropsSink, CsvSink, DisplaySink, FfmpegSink, ImageSink, JsonSink
from segmentator.pipeline import Ctx


def _frame(width=8, height=8, value=0):
    return np.full((height, width, 3), value, np.uint8)


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #


def test_ffmpeg_sink_writes_a_playable_video_with_one_frame_per_write(tmp_path):
    """Each `write` pipes one frame to ffmpeg; `close` must flush and finalize the file."""
    out = tmp_path / "out.mp4"
    sink = FfmpegSink(path=out)

    for index in range(3):
        ctx = Ctx(image=_frame(value=index * 10), source=_frame(value=index * 10), index=index)
        assert sink.write(ctx) is True
    sink.close()

    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (8, 8)
    finally:
        cap.release()


def test_ffmpeg_sink_promotes_grayscale_frames_to_bgr(tmp_path):
    """A single-channel `image` (e.g. after `threshold`) must still encode, not crash."""
    out = tmp_path / "gray.mp4"
    sink = FfmpegSink(path=out, fps=15.0)

    gray = np.full((8, 8), 200, np.uint8)
    ctx = Ctx(image=gray, source=gray, index=0)
    sink.write(ctx)
    sink.close()

    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# image
# --------------------------------------------------------------------------- #


def test_image_sink_writes_one_file_per_frame_via_the_index_template(tmp_path):
    """`{index}` in the path is substituted per frame, so a run numbers a sequence."""
    template = str(tmp_path / "frame_{index:03d}.png")
    sink = ImageSink(path=template)

    frames = [_frame(width=5, height=4, value=10), _frame(width=5, height=4, value=250)]
    for index, frame in enumerate(frames):
        ctx = Ctx(image=frame, source=frame, index=index)
        assert sink.write(ctx) is True
    sink.close()

    for index, frame in enumerate(frames):
        out = tmp_path / f"frame_{index:03d}.png"
        assert out.exists()
        written = cv2.imread(str(out), cv2.IMREAD_COLOR)
        assert np.array_equal(written, frame)


def test_image_sink_without_a_template_overwrites_every_frame(tmp_path):
    """A path with no `{index}` is the single-frame case: each write replaces the last."""
    out = tmp_path / "still.png"
    sink = ImageSink(path=str(out))

    first, second = _frame(value=1), _frame(value=222)
    sink.write(Ctx(image=first, source=first, index=0))
    sink.write(Ctx(image=second, source=second, index=1))

    written = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert np.array_equal(written, second)


# --------------------------------------------------------------------------- #
# csv
# --------------------------------------------------------------------------- #


def test_csv_sink_writes_one_file_per_kind_with_a_frame_column_prepended(tmp_path):
    """Rows go to `{dir}/{kind}.csv`; a kind that never appears never gets a file."""
    out_dir = tmp_path / "csvs"
    sink = CsvSink(dir=out_dir)

    ctx0 = Ctx(
        image=_frame(),
        source=_frame(),
        index=0,
        rows={"contours": [{"x": 1, "y": 2, "w": 3, "h": 4}, {"x": 5, "y": 6, "w": 7, "h": 8}]},
    )
    ctx1 = Ctx(
        image=_frame(),
        source=_frame(),
        index=1,
        rows={"contours": [{"x": 9, "y": 10, "w": 11, "h": 12}]},
    )
    sink.write(ctx0)
    sink.write(ctx1)
    sink.close()

    contours_csv = out_dir / "contours.csv"
    assert contours_csv.exists()
    with open(contours_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"frame": "0", "x": "1", "y": "2", "w": "3", "h": "4"},
        {"frame": "0", "x": "5", "y": "6", "w": "7", "h": "8"},
        {"frame": "1", "x": "9", "y": "10", "w": "11", "h": "12"},
    ]

    assert not (out_dir / "keypoints.csv").exists(), "a kind with no rows must produce no file"


def test_csv_sink_kinds_filter_restricts_which_files_get_written(tmp_path):
    """`kinds` narrows output to named row kinds; everything else is dropped."""
    out_dir = tmp_path / "csvs"
    sink = CsvSink(dir=out_dir, kinds=("contours",))

    ctx = Ctx(
        image=_frame(),
        source=_frame(),
        index=0,
        rows={
            "contours": [{"x": 1, "y": 1, "w": 1, "h": 1}],
            "motion": [{"dx": 1, "dy": 2}],
        },
    )
    sink.write(ctx)
    sink.close()

    assert (out_dir / "contours.csv").exists()
    assert not (out_dir / "motion.csv").exists()


# --------------------------------------------------------------------------- #
# json
# --------------------------------------------------------------------------- #


def test_json_sink_writes_one_metrics_object_per_line(tmp_path):
    """JSON Lines: each line is `{"frame": idx, **ctx.metrics}`, so a killed run stays readable."""
    out = tmp_path / "metrics.jsonl"
    sink = JsonSink(path=out)

    sink.write(Ctx(image=_frame(), source=_frame(), index=0, metrics={"edge_px": 42}))
    sink.write(
        Ctx(image=_frame(), source=_frame(), index=1, metrics={"edge_px": 84, "motion_px": 3})
    )
    sink.close()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"frame": 0, "edge_px": 42},
        {"frame": 1, "edge_px": 84, "motion_px": 3},
    ]


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #


def test_crops_sink_cuts_each_rectangle_row_out_of_the_frame(tmp_path):
    """One crop file per complete x/y/w/h row; incomplete or out-of-frame rows are skipped."""
    out_dir = tmp_path / "crops"
    sink = CropsSink(dir=out_dir, kind="contours", input="source", pad=0)

    frame = np.zeros((20, 20, 3), np.uint8)
    frame[3:8, 2:6] = (10, 20, 30)  # matches the first row's rectangle exactly
    ctx = Ctx(
        image=frame,
        source=frame,
        index=0,
        rows={
            "contours": [
                {"x": 2, "y": 3, "w": 4, "h": 5},
                {"x": 100, "y": 100, "w": 1, "h": 1},  # fully outside the frame: skipped
                {"x": 0, "y": 0, "w": 4},  # missing "h": skipped
            ]
        },
    )
    assert sink.write(ctx) is True

    written = sorted(out_dir.glob("*.png"))
    assert [p.name for p in written] == ["000000_000.png"]
    crop = cv2.imread(str(written[0]), cv2.IMREAD_COLOR)
    assert crop.shape == (5, 4, 3)
    assert np.array_equal(crop, frame[3:8, 2:6])
    assert sink.count == 1


def test_crops_sink_writes_nothing_when_its_kind_has_no_rows(tmp_path):
    """A kind that never produced rows (e.g. pointed at the wrong kind) writes no files."""
    out_dir = tmp_path / "crops"
    sink = CropsSink(dir=out_dir, kind="contours")

    frame = _frame()
    ctx = Ctx(image=frame, source=frame, index=0, rows={"keypoints": [{"x": 1, "y": 1}]})
    assert sink.write(ctx) is True

    assert not out_dir.exists()


# --------------------------------------------------------------------------- #
# display
# --------------------------------------------------------------------------- #


def test_display_sink_lifecycle_is_safe_without_a_window_ever_opening(tmp_path):
    """Construct + close must never touch highgui when nothing was shown.

    `cv2.imshow` needs a real display (this build's OpenCV ships only the Qt
    `xcb` platform plugin, no `offscreen` fallback — calling it with no `DISPLAY`
    aborts the whole process). So the only lifecycle a headless test can safely
    drive end to end is the one a run that never gets a frame actually takes:
    construct, then close without ever calling `write`.
    """
    sink = DisplaySink(window="preview", size=(64, 48))
    sink.close()  # must not raise, and must not call cv2.destroyWindow
    sink.close()  # idempotent: closing twice is still safe
