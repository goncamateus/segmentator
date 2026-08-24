"""Direct tests for the frame sources in segmentator/io.py.

Each source is constructed directly and driven through its existing protocol
(`__iter__`, `count`, `read`, `close`) against temporary files built in the
test, rather than checked-in fixtures.

    uv run pytest tests/test_sources.py
"""

import cv2
import numpy as np
import pytest

from segmentator.io import FolderSource, ImageSource, VideoSource


def _write_video(path, values, size=(8, 8), fps=10.0):
    """A synthetic .avi where frame ``i`` is flat-filled with ``values[i]``.

    MJPG round-trips a flat-colour frame byte-exact (verified empirically), so
    the written values can be asserted back exactly without a checked-in clip.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "test setup: MJPG/.avi writer failed to open"
    for value in values:
        frame = np.full((size[1], size[0], 3), value, np.uint8)
        writer.write(frame)
    writer.release()


def _write_image(path, color, size=(4, 4)):
    """A solid-colour BGR image written to ``path``. Returns the array written.

    ``size`` is ``(width, height)``, matching the sources' own ``size`` attribute.
    ``color`` may be a scalar (all channels equal) or a ``(b, g, r)`` triple.
    """
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:] = color
    cv2.imwrite(str(path), frame)
    return frame


# --------------------------------------------------------------------------- #
# VideoSource
# --------------------------------------------------------------------------- #


def test_video_source_reads_frames_in_order_and_count(tmp_path):
    values = [10, 40, 70, 100, 130]
    path = tmp_path / "clip.avi"
    _write_video(path, values)

    source = VideoSource(path)
    try:
        frames = list(source)
        assert source.count == len(values)
        assert source.size == (8, 8)
    finally:
        source.close()

    assert [int(frame[0, 0, 0]) for frame in frames] == values


def test_video_source_read_supports_random_access(tmp_path):
    values = [5, 15, 25, 35]
    path = tmp_path / "clip.avi"
    _write_video(path, values)

    source = VideoSource(path)
    try:
        assert int(source.read(2)[0, 0, 0]) == values[2]
        # Reading backwards forces the position-set path (index != self._next).
        assert int(source.read(0)[0, 0, 0]) == values[0]
        assert int(source.read(3)[0, 0, 0]) == values[3]
        assert source.read(len(values)) is None, "past the end reads as None"
    finally:
        source.close()


def test_video_source_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError, match="Could not open video"):
        VideoSource(tmp_path / "missing.avi")


# --------------------------------------------------------------------------- #
# ImageSource
# --------------------------------------------------------------------------- #


def test_image_source_yields_a_single_frame(tmp_path):
    path = tmp_path / "still.png"
    frame = _write_image(path, (10, 20, 30), size=(9, 6))

    source = ImageSource(path)
    frames = list(source)

    assert len(frames) == 1
    assert np.array_equal(frames[0], frame)
    assert source.count == 1
    assert source.size == (9, 6), "size is (width, height)"
    assert np.array_equal(source.read(0), frame)
    assert source.read(1) is None
    source.close()


def test_image_source_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError, match="Could not read image"):
        ImageSource(tmp_path / "missing.png")


# --------------------------------------------------------------------------- #
# FolderSource
# --------------------------------------------------------------------------- #


def test_folder_source_yields_frames_in_filename_sorted_order(tmp_path):
    """Files are written in a scrambled order to prove sorting drives iteration.

    If FolderSource iterated in directory/creation order instead of sorted
    filename order, this would catch it.
    """
    value_by_name = {
        "frame_003.png": 3,
        "frame_000.png": 0,
        "frame_002.png": 2,
        "frame_001.png": 1,
    }
    for name in ["frame_003.png", "frame_000.png", "frame_002.png", "frame_001.png"]:
        _write_image(tmp_path / name, value_by_name[name])

    source = FolderSource(tmp_path)
    frames = list(source)

    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2, 3]
    assert source.count == 4
    assert source.size == (4, 4)
    assert np.array_equal(source.read(2), frames[2])
    assert source.read(4) is None
    source.close()


def test_folder_source_filters_non_image_files(tmp_path):
    _write_image(tmp_path / "a.png", 0)
    (tmp_path / "notes.txt").write_text("not an image")

    source = FolderSource(tmp_path)

    assert source.count == 1


def test_folder_source_pattern_narrows_selection(tmp_path):
    _write_image(tmp_path / "keep_a.png", 0)
    _write_image(tmp_path / "skip_b.png", 0)

    source = FolderSource(tmp_path, pattern="keep_*")

    assert source.count == 1


def test_folder_source_empty_directory_raises_oserror(tmp_path):
    with pytest.raises(OSError, match="No images matching"):
        FolderSource(tmp_path)


def test_folder_source_missing_directory_raises_oserror(tmp_path):
    with pytest.raises(OSError, match="Not a directory"):
        FolderSource(tmp_path / "nope")
