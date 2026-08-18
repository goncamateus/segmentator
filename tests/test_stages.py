"""Tests for the operators ported from goncanalyser.

The assertions are goncanalyser's own — the ones its ``features/*._demo()``
functions already make — re-pointed at the stage classes. They are ported rather
than invented so that "parity" means the two repos agree on numbers, not just
that both run without raising.

    uv run pytest
"""

import cv2
import numpy as np
import pytest

from segmentator.ops.common import odd_kernel
from segmentator.ops.texture import LBP_METHODS
from segmentator.pipeline import Ctx, build
from segmentator.stages.preprocess import MIN_ROI, roi_rect


def run(specs, frame, source=None, index=0, ctx=None):
    """Apply a chain of ``{type: ..., **params}`` specs to one frame. Returns the Ctx."""
    if ctx is None:
        ctx = Ctx(image=frame, source=frame if source is None else source, index=index)
    else:
        ctx.image, ctx.source, ctx.index = frame, frame if source is None else source, index
    for spec in specs:
        build("stage", dict(spec)).apply(ctx)
    return ctx


def play(specs, frames):
    """Drive a stateful chain over a sequence, the way Pipeline.run does. Last Ctx out."""
    stages = [build("stage", dict(spec)) for spec in specs]
    ctx = None
    for index, frame in enumerate(frames):
        ctx = Ctx(image=frame, source=frame, index=index)
        for stage in stages:
            stage.apply(ctx)
    return ctx


@pytest.fixture
def square():
    """A 200x200 black frame with a white 100px square — something to find."""
    frame = np.zeros((200, 200, 3), np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
    return frame


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #


def test_identity_parameters_are_a_passthrough(square):
    """Defaults must not touch a single pixel, or a chain costs something for nothing."""
    specs = [
        {"type": "brightness_contrast"},
        {"type": "saturation"},
        {"type": "gamma"},
        {"type": "gaussian_blur", "ksize": 1},
        {"type": "median_blur", "ksize": 1},
    ]
    assert (run(specs, square).image == square).all()


def test_blur_off_is_off_not_a_one_pixel_kernel(square):
    """`ksize: 0` used to crash GaussianBlur; the odd_kernel clamp is what stops it."""
    assert odd_kernel(0) == 1 and odd_kernel(4) == 5 and odd_kernel(-3) == 1
    assert (run([{"type": "gaussian_blur", "ksize": 0}], square).image == square).all()
    for kind in ("gaussian_blur", "median_blur"):
        out = run([{"type": kind, "ksize": 5}], square).image
        assert out.shape == square.shape and not (out == square).all(), kind


@pytest.mark.parametrize("space", ["hsv", "lab", "hls", "ycrcb"])
def test_every_colour_space_stays_displayable(square, space):
    out = run([{"type": "colorspace", "to": space}], square).image
    assert out.ndim == 3 and out.shape[2] == 3


def test_colorspace_names_the_bad_value():
    with pytest.raises(ValueError, match="unknown colour space"):
        build("stage", {"type": "colorspace", "to": "nope"})


@pytest.mark.parametrize("mode", ["binary", "binary_inv"])
def test_threshold_modes_are_binary(square, mode):
    out = run([{"type": "gray"}, {"type": "threshold", "mode": mode}], square).image
    assert set(np.unique(out)) <= {0, 255}


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_adaptive_threshold_is_binary_and_clamps_its_block(square, method):
    stage = build("stage", {"type": "adaptive_threshold", "method": method, "block": 4})
    assert stage.block == 5, "an even block size must be clamped odd"
    out = run([{"type": "gray"}, {"type": "adaptive_threshold", "method": method}], square).image
    assert set(np.unique(out)) <= {0, 255}


def test_adaptive_threshold_block_floor():
    assert build("stage", {"type": "adaptive_threshold", "block": 1}).block == 3


def test_gamma_moves_the_levels_the_right_way(square):
    grey = np.full((8, 8, 3), 100, np.uint8)
    assert run([{"type": "gamma", "value": 2.0}], grey).image.mean() > 100
    assert run([{"type": "gamma", "value": 0.5}], grey).image.mean() < 100


def test_saturation_needs_colour():
    grey = np.zeros((8, 8), np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        run([{"type": "saturation", "gain": 2.0}], grey)


# --------------------------------------------------------------------------- #
# Region of interest
# --------------------------------------------------------------------------- #


def test_roi_rect_is_fitted_not_trusted():
    """Oversized, undersized and off-frame rectangles all resolve to something usable."""
    shape = (64, 96, 3)  # 64 tall, 96 wide
    assert roi_rect(shape, 10, 8, 40, 30) == (10, 8, 40, 30)
    # 0 means "out to the edge", which keeps the frame size out of the config.
    assert roi_rect(shape, 0, 0, 0, 0) == (0, 0, 96, 64)
    assert roi_rect(shape, 10, 8, 9000, 9000) == (0, 0, 96, 64)
    # A stray one-pixel rectangle is widened, and slid back inside the frame
    # rather than left hanging off the edge.
    assert roi_rect(shape, 95, 63, 1, 1) == (96 - MIN_ROI, 64 - MIN_ROI, MIN_ROI, MIN_ROI)
    assert roi_rect((8, 8, 3), 10, 8, 40, 30) == (0, 0, 8, 8), "frame smaller than MIN_ROI"


def test_roi_crops_to_the_rectangle_and_copies(square):
    ctx = run([{"type": "roi", "x": 10, "y": 8, "w": 40, "h": 30}], square)
    assert ctx.image.shape == (30, 40, 3)
    assert (ctx.image == square[8:38, 10:50]).all()
    ctx.image[:] = 0
    assert square.any(), "roi returned a view into the caller's frame"


def test_roi_isolates_the_region_from_its_surround():
    """Otsu and a blur are exactly the two operators that leak across the border."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 40), (100, 80), (255, 255, 255), -1)
    x, y, w, h = 40, 20, 80, 70

    scribbled = np.zeros_like(frame)  # obliterate everything…
    scribbled[y : y + h, x : x + w] = frame[y : y + h, x : x + w]  # …but the region

    specs = [
        {"type": "roi", "x": x, "y": y, "w": w, "h": h},
        {"type": "gray"},
        {"type": "gaussian_blur", "ksize": 5},
        {"type": "threshold", "otsu": True},
        {"type": "contours", "min_area": 1, "draw_on": "image"},
    ]
    assert run(specs, frame).metrics == run(specs, scribbled).metrics

    # The negative, or the assert above would also pass for a chain that ignored
    # its input entirely.
    no_roi = specs[1:]
    assert run(no_roi, frame).metrics != run(no_roi, scribbled).metrics


def test_paste_roi_restores_the_frame_and_translates_the_rows():
    frame = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(frame, (60, 40), (100, 80), (255, 255, 255), -1)
    x, y = 40, 20
    ctx = run(
        [
            {"type": "roi", "x": x, "y": y, "w": 80, "h": 70},
            {"type": "gray"},
            {"type": "contours", "min_area": 10, "draw_on": "image"},
            {"type": "paste_roi"},
        ],
        frame,
    )
    assert ctx.image.shape == frame.shape, "the region was not pasted back at full size"
    # The rows come back in frame coordinates, so they land inside the rectangle.
    row = ctx.rows["contours"][0]
    assert x <= row["x"] < x + 80 and y <= row["y"] < y + 70, row


def test_paste_roi_without_a_roi_names_the_missing_stage(square):
    with pytest.raises(KeyError, match="'roi' stage"):
        run([{"type": "paste_roi"}], square)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_canny_finds_the_outline_of_a_square(square):
    edges = run([{"type": "gray"}, {"type": "canny"}], square).metrics["edge_px"]
    assert edges > 300, f"a 100px square has ~400px of outline, got {edges}"


@pytest.mark.parametrize("kind", ["canny", "sobel", "laplacian"])
def test_every_edge_operator_emits_a_single_channel_map(square, kind):
    out = run([{"type": "gray"}, {"type": kind}], square)
    assert out.image.shape == square.shape[:2] and out.image.dtype == np.uint8
    assert out.metrics["edge_px"] > 0


def test_sobel_will_not_accept_both_derivatives_zero():
    assert build("stage", {"type": "sobel", "dx": 0, "dy": 0}).dx == 1


def test_hough_finds_the_four_sides(square):
    """Four sides, so at least four segments — Hough usually splits them further."""
    ctx = run([{"type": "gray"}, {"type": "hough_lines", "threshold": 50}], square)
    assert ctx.metrics["lines"] >= 4
    assert set(ctx.rows["lines"][0]) == {"x1", "y1", "x2", "y2"}


def test_hough_circles_tolerates_an_image_with_none(square):
    ctx = run([{"type": "gray"}, {"type": "hough_circles"}], square)
    assert ctx.metrics["circles"] == 0
    assert ctx.image.shape == square.shape, "the overlay must not resize the frame"


@pytest.mark.parametrize("kind", ["harris", "shi_tomasi"])
def test_both_corner_detectors_find_a_square(square, kind):
    found = run([{"type": "gray"}, {"type": kind}], square).metrics["corners"]
    assert 4 <= found <= 200, f"{kind} found {found} corners on a square"


def test_harris_keeps_the_strongest_not_the_first(square):
    """The cap must rank by response, or it truncates to the top-left of the image."""
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    from segmentator.ops.structure import harris_corners

    points = harris_corners(gray, max_corners=4)
    assert len(points) == 4
    # The four strongest responses on a square are its four corners, so every
    # kept point must sit near one of them rather than clustering in one region.
    assert len({(x > 100, y > 100) for x, y in points}) == 4, points


def test_contours_measure_the_square(square):
    ctx = run([{"type": "gray"}, {"type": "contours", "min_area": 50}], square)
    assert ctx.metrics["contours"] == 1
    row = ctx.rows["contours"][0]
    assert abs(row["area"] - 100 * 100) < 500, row
    assert (row["w"], row["h"]) == (101, 101), row
    assert ctx.store["contour_mask"].shape == square.shape[:2]


def test_contour_tree_mode_reports_the_parent_link():
    """A square inside a square: `external` sees one shape, `tree` sees both."""
    frame = np.zeros((200, 200), np.uint8)
    cv2.rectangle(frame, (20, 20), (180, 180), 255, -1)
    cv2.rectangle(frame, (60, 60), (140, 140), 0, -1)

    assert run([{"type": "contours", "mode": "external"}], frame).metrics["contours"] == 1
    tree = run([{"type": "contours", "mode": "tree"}], frame)
    assert tree.metrics["contours"] == 2
    assert sorted(r["parent"] for r in tree.rows["contours"]) == [-1, 0]


def test_contours_reject_an_unknown_mode(square):
    with pytest.raises(ValueError, match="unknown contour mode"):
        run([{"type": "gray"}, {"type": "contours", "mode": "nope"}], square)


def test_bounding_boxes_reuse_the_contours_already_found(square):
    ctx = run(
        [{"type": "gray"}, {"type": "contours"}, {"type": "bounding_boxes"}],
        square,
    )
    assert ctx.metrics["boxes"] == 1
    assert ctx.store["boxes"][0][2:] == (101, 101)


def test_blobs_find_a_dark_circle_on_light():
    """A dark blob on white is what SimpleBlobDetector looks for by default."""
    frame = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(frame, (100, 100), 30, (0, 0, 0), -1)
    ctx = run([{"type": "blobs", "max_area": 20000}], frame)
    assert ctx.metrics["blobs"] == 1
    assert set(ctx.rows["blobs"][0]) == {"x", "y", "diameter", "response"}


# --------------------------------------------------------------------------- #
# Keypoints
# --------------------------------------------------------------------------- #


@pytest.fixture
def checkerboard():
    frame = np.zeros((160, 160, 3), np.uint8)
    for row in range(4):
        for col in range(4):
            if (row + col) % 2:
                frame[row * 40 : row * 40 + 40, col * 40 : col * 40 + 40] = 255
    return frame


@pytest.mark.parametrize("detector", ["sift", "orb"])
def test_both_detectors_find_checkerboard_corners(checkerboard, detector):
    ctx = run([{"type": "keypoints", "detector": detector}], checkerboard)
    assert ctx.metrics["keypoints"] > 0
    assert set(ctx.rows["keypoints"][0]) == {
        "x", "y", "size", "angle", "response", "octave",
    }
    assert ctx.image.shape == checkerboard.shape, "the overlay resized the frame"


def test_keypoint_cap_is_honoured(checkerboard):
    ctx = run([{"type": "keypoints", "detector": "orb", "max": 5}], checkerboard)
    assert ctx.metrics["keypoints"] <= 5


def test_sensitivity_maps_onto_each_detectors_own_threshold():
    """One normalised knob, two unrelated native scales, both strict -> permissive."""
    from segmentator.ops.keypoints import threshold_for

    assert threshold_for("sift", 0.0) == pytest.approx(0.16)
    assert threshold_for("sift", 1.0) == pytest.approx(0.005)
    assert threshold_for("orb", 0.0) == pytest.approx(60.0)
    # Out-of-range values clamp rather than extrapolating off the end of the range.
    assert threshold_for("orb", 5.0) == threshold_for("orb", 1.0)


def test_unknown_detector_lists_the_known_ones():
    with pytest.raises(ValueError, match="unknown detector"):
        build("stage", {"type": "keypoints", "detector": "surf"})


# --------------------------------------------------------------------------- #
# Texture
# --------------------------------------------------------------------------- #


def test_hog_vector_length_matches_the_geometry():
    """Getting this wrong silently is the classic HOG bug, so pin it to the formula."""
    from segmentator.ops.texture import hog_of

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    orientations, cell, block = 9, 8, 2
    vector, image = hog_of(noise, orientations, cell, block)
    cells = 64 // cell
    blocks = cells - block + 1
    assert vector.size == blocks * blocks * block**2 * orientations
    assert image.shape == noise.shape and image.dtype == np.uint8


def test_hog_describes_textured_and_flat_differently():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    flat = np.full((64, 64), 128, np.uint8)
    busy = run([{"type": "hog"}], noise).metrics["hog_mean"]
    plain = run([{"type": "hog"}], flat).metrics["hog_mean"]
    assert busy > plain


def test_lbp_uniform_stays_inside_its_defined_range():
    """"uniform" is defined to produce codes in 0..P+1 and nothing outside it."""
    from segmentator.ops.texture import lbp_of

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    codes = lbp_of(noise, points=8, radius=1)
    assert codes.min() >= 0 and codes.max() <= 8 + 1


@pytest.mark.parametrize("method", LBP_METHODS)
def test_every_lbp_method_survives_a_flat_image(method):
    """"var" is NaN on a flat patch, which used to crash the histogram and the cast."""
    flat = np.full((64, 64), 128, np.uint8)
    ctx = run([{"type": "lbp", "method": method}], flat)
    assert np.isfinite(ctx.image).all(), method
    assert ctx.metrics["lbp_bins"] >= 1, method


def test_lbp_entropy_separates_varied_texture_from_uniform():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    flat = np.full((64, 64), 128, np.uint8)
    busy = run([{"type": "lbp"}], noise).metrics["lbp_entropy"]
    plain = run([{"type": "lbp"}], flat).metrics["lbp_entropy"]
    assert busy > plain


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #


def test_known_images_have_known_histograms():
    from segmentator.ops.color import histograms

    black = np.zeros((32, 32, 3), np.uint8)
    counts = histograms(black, "rgb")
    assert counts.shape == (3, 256)
    assert counts[0, 0] == 32 * 32 and counts[0, 1:].sum() == 0, "all black is bin 0"

    grey = np.full((32, 32, 3), 128, np.uint8)
    assert histograms(grey, "rgb")[1, 128] == 32 * 32


def test_histogram_measures_intensity_not_counts():
    grey = np.full((32, 32, 3), 128, np.uint8)
    ctx = run([{"type": "histogram"}], grey)
    assert ctx.metrics["B_mean"] == 128.0
    assert ctx.metrics["B_sd"] == 0.0, "a uniform image has no spread"


def test_histogram_leaves_the_frame_alone_unless_asked(square):
    ctx = run([{"type": "histogram"}], square)
    assert (ctx.image == square).all(), "the default must not hijack the chain"
    assert ctx.store["histogram"].shape == (256, 512, 3)
    assert run([{"type": "histogram", "replace": True}], square).image.shape == (256, 512, 3)


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

# A textured object, not a flat square: three-frame differencing takes the
# *minimum* of two consecutive differences, and the interior of a uniform object
# is identical in both, so a flat square is invisible to it by construction.
_RAMP = 40 + 7 * np.arange(20)[None, :] + 3 * np.arange(20)[:, None]
_PATCH = np.repeat(_RAMP.astype(np.uint8)[:, :, None], 3, axis=2)

MOTION_ALGOS = ["mog2", "knn", "farneback", "lucas_kanade", "frame_diff", "three_frame_diff"]


def clip(step, frames=6):
    """`frames` frames of a 20px textured square translating `step` px each time."""
    out = []
    for i in range(frames):
        img = np.zeros((120, 160, 3), np.uint8)
        left = 20 + i * step
        img[50:70, left : left + 20] = _PATCH
        out.append(img)
    return out


def motion_chain(algo, **objects):
    """The composed equivalent of one goncanalyser motion algorithm."""
    return [
        {"type": "gray"},
        {"type": algo},
        {"type": "threshold", "value": 25, "mode": "binary"},
        {"type": "motion_objects", "min_area": 20, "draw_on": "image", **objects},
    ]


@pytest.mark.parametrize("algo", MOTION_ALGOS)
def test_every_algorithm_sees_a_moving_square(algo):
    ctx = play(motion_chain(algo), clip(8))
    assert ctx.metrics["motion_objects"] >= 1, f"{algo} missed a moving square"
    assert ctx.metrics["motion_px"] > 0, algo
    assert set(ctx.rows["motion"][0]) == {"x", "y", "w", "h", "area", "speed"}


@pytest.mark.parametrize("algo", MOTION_ALGOS)
def test_no_algorithm_sees_motion_in_a_still_clip(algo):
    """The negative, and the one that catches a model being fed noise."""
    ctx = play(motion_chain(algo), [clip(0)[0]] * 6)
    assert ctx.metrics["motion_px"] == 0, f"{algo} found motion in a still clip"


def test_speed_is_measured_not_invented():
    """8 px a frame must read as roughly 8, and slower must read as slower."""
    fast = play(motion_chain("frame_diff"), clip(8)).metrics["motion_speed"]
    slow = play(motion_chain("frame_diff"), clip(2)).metrics["motion_speed"]
    assert 6 <= fast <= 10, fast
    assert slow < fast


def test_area_filter_removes_small_blobs():
    chain = motion_chain("frame_diff", min_area=100_000)
    assert not play(chain, clip(8)).rows["motion"]


@pytest.mark.parametrize("algo", ["farneback", "lucas_kanade", "three_frame_diff", "frame_diff"])
def test_a_frame_shape_change_resets_the_history(algo):
    """A resized frame mid-run is a raw OpenCV size assertion if it is not handled."""
    frames = clip(8)
    frames.append(frames[0][:60, :80])
    ctx = play([{"type": "gray"}, {"type": algo}], frames)
    assert ctx.image.shape == (60, 80), algo


def test_motion_heat_accumulates_and_leaves_the_frame_alone():
    """The accumulator is float32, so it is not a frame — `heatmap` is what draws it."""
    chain = [{"type": "gray"}, {"type": "frame_diff"}]
    without = play(chain, clip(8)).image
    ctx = play([*chain, {"type": "motion_heat", "window": 3}], clip(8))

    assert ctx.store["heat"].dtype == np.float32
    assert ctx.store["heat"].max() > 0
    assert (ctx.image == without).all(), "motion_heat must not rebind ctx.image"

    # A rolling average trails the instantaneous signal rather than tracking it.
    assert ctx.store["heat"].max() < ctx.image.max()


def test_heatmap_paints_something_and_tolerates_an_odd_canvas():
    ctx = play(
        [
            {"type": "gray"},
            {"type": "frame_diff"},
            {"type": "motion_heat"},
            {"type": "heatmap", "draw_on": "source"},
        ],
        clip(8),
    )
    assert ctx.image.ndim == 3 and ctx.image.any(), "the heat overlay painted nothing"

    # A mismatched canvas — the histogram plot is 512x256 whatever the frame is —
    # must be left alone rather than raising.
    from segmentator.ops.motion import blend

    plot = np.zeros((256, 512, 3), np.uint8)
    blend(plot, ctx.store["heat"], 0.5, 0.05)
    assert not plot.any()


def test_drop_shadows_removes_the_half_lit_label():
    """MOG2 labels a shadow 127; it is the moving thing's effect, not the thing."""
    from segmentator.stages.motion import Mog2

    stage = Mog2(detect_shadows=True, drop_shadows=True)
    ctx = None
    for index, frame in enumerate(clip(8)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ctx = Ctx(image=gray, source=gray, index=index)
        stage.apply(ctx)
    assert set(np.unique(ctx.image)) <= {0, 255}
