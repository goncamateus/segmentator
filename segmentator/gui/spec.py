"""What the editor needs to know about a config, with no Qt in it.

Three jobs, and each one is deliberately *derived* rather than maintained:

* **The palette and the parameter form** come from ``inspect.signature`` of the
  registered class, so adding a stage adds a form.
* **Live vs rebuild** — whether a knob can be assigned to a running stage or
  needs a new instance — is read off the instance's own attributes.
* **Load and save** go through ruamel round-trip, because ``configs/*.yaml`` are
  heavily commented on purpose and PyYAML's ``safe_load`` drops every one.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from segmentator import io, stages  # noqa: F401  — importing registers everything
from segmentator.pipeline import _REGISTRY, build, registered

# Stages that carry something from one frame to the next: a background model, a
# previous frame, an accumulator, the last set of centroids, a fitted mask. The
# editor treats these specially — re-running one over the frame already on screen
# would teach it that frame, and rebuilding one throws its history away.
STATEFUL = frozenset(
    {
        "mog2",  # _subtractor
        "knn",  # _subtractor
        "frame_diff",  # _history deque
        "three_frame_diff",  # _history
        "farneback",  # _history
        "lucas_kanade",  # _history
        "motion_heat",  # _heat accumulator
        "motion_objects",  # _previous centroids
        "mean_background",  # _model
        "static_mask",  # mask, fitted on the first frame
    }
)

# Keys of a stage spec that are not constructor parameters.
RESERVED = ("type", "name")

# Stages that publish an image into ``ctx.store`` under a fixed key a later
# ``input:`` or ``draw_on:`` can name — e.g. ``contours`` also publishes
# ``contour_mask``. Hand-maintained like STATEFUL, and checked by _demo()
# against the registry. Only image-valued keys are listed: a stage that
# publishes a list (``contours`` itself) or a vector (``hog``) has nothing a
# preview could resolve to.
PUBLISHES: dict[str, tuple[str, ...]] = {
    "static_mask": ("mask",),
    "motion_heat": ("heat",),
    "histogram": ("histogram",),
    "contours": ("contour_mask",),
}

# The "Add stage" palette's sections. Editorial, not derivable from the
# registry — a family is a grouping by what you reach for it *for*, which the
# registry has no notion of — so, like STATEFUL, it is hand-maintained and
# checked by _demo() rather than generated. Matches docs/stages.md and
# docs/assets/stage-families.svg exactly: same 12 families, same order, same
# membership, so the palette a user opens in the editor is the same shape as
# the catalogue they might have already read.
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Adjust", ("brightness_contrast", "saturation", "gamma", "gray", "colorspace", "resize", "select")),
    ("Blur / morphology", ("median_blur", "gaussian_blur", "clahe", "morphology")),
    ("Threshold", ("threshold", "adaptive_threshold")),
    ("Region of interest", ("roi", "paste_roi")),
    ("Edges", ("canny", "sobel", "laplacian")),
    ("Geometry", ("hough_lines", "hough_circles", "harris", "shi_tomasi", "contours", "bounding_boxes", "blobs")),
    ("Keypoints", ("keypoints",)),
    ("Texture / colour", ("hog", "lbp", "histogram")),
    ("Masking", ("static_mask", "apply_mask", "mean_background")),
    ("Motion — models", ("mog2", "knn", "frame_diff", "three_frame_diff")),
    ("Motion — flow", ("farneback", "lucas_kanade")),
    ("Motion — after", ("motion_heat", "heatmap", "motion_objects")),
)


def _choices() -> dict[tuple[str, str], tuple[str, ...]]:
    """String parameters with a fixed set of valid values, as a dropdown each.

    Read out of the tables the stages already validate against, so a mode added
    there becomes an option here without anything being maintained twice.
    """
    from segmentator.ops.color import SPACES
    from segmentator.ops.structure import CONTOUR_MODES
    from segmentator.ops.texture import LBP_METHODS
    from segmentator.stages import preprocess

    return {
        ("threshold", "mode"): tuple(preprocess._THRESHOLD_MODES),
        ("morphology", "op"): tuple(preprocess._MORPH_OPS),
        ("adaptive_threshold", "method"): tuple(preprocess._ADAPTIVE_METHODS),
        ("colorspace", "to"): tuple(sorted(preprocess._COLOR_CODES)),
        ("contours", "mode"): tuple(CONTOUR_MODES),
        ("histogram", "space"): tuple(SPACES),
        ("lbp", "method"): tuple(sorted(LBP_METHODS)),
        ("keypoints", "detector"): ("sift", "orb"),
    }


CHOICES = _choices()


@dataclass(frozen=True)
class Param:
    """One constructor parameter, as the form needs it."""

    name: str
    annotation: Any
    default: Any

    @property
    def required(self) -> bool:
        return self.default is inspect.Parameter.empty


def component(kind: str, type_name: str) -> type:
    """The registered class behind a ``type:``. Raises like :func:`build` does."""
    table = _REGISTRY[kind]
    if type_name not in table:
        raise KeyError(f"unknown {kind} {type_name!r}; known: {sorted(table)}")
    return table[type_name]


def params(kind: str, type_name: str) -> list[Param]:
    """Constructor parameters of a component, in declaration order.

    ``*args``/``**kwargs`` are dropped, which is how a stage that defines no
    ``__init__`` at all (``gray``) correctly reports no parameters instead of
    ``object.__init__``'s signature.
    """
    signature = inspect.signature(component(kind, type_name).__init__)
    return [
        Param(p.name, p.annotation, p.default)
        for p in signature.parameters.values()
        if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def rebuild_params(kind: str, spec: dict[str, Any]) -> frozenset[str]:
    """Parameters that cannot be assigned to a running instance.

    A parameter is *live* — assignable, history intact — exactly when the built
    instance carries an attribute of that name, because that is what a stage's
    ``apply`` reads. Anything consumed at construction time (``mog2``'s history
    goes into an OpenCV subtractor, ``frame_diff``'s lag into a deque's maxlen)
    leaves no such attribute and needs a new instance.

    ``RECONSTRUCT`` on the class is the opt-out for the one case the rule gets
    wrong: :class:`~segmentator.stages.mask.StaticMask` stores ``threshold`` but
    caches the mask it fitted from it.

    A spec that will not build is reported as all-rebuild — the editor is being
    typed into, and half a ``type:`` is not an error worth a dialog.
    """
    try:
        names = [p.name for p in params(kind, spec.get("type", ""))]
    except KeyError:
        return frozenset()  # unknown type: no form to draw, nothing to classify
    try:
        instance = build(kind, {k: v for k, v in spec.items() if k != "name"})
    except Exception:
        return frozenset(names)
    forced = getattr(type(instance), "RECONSTRUCT", ())
    return frozenset(n for n in names if not hasattr(instance, n) or n in forced)


def new_spec(kind: str, type_name: str) -> CommentedMap:
    """A minimal spec for a freshly added component, in the flow style the configs use.

    Required parameters get a placeholder rather than being left out, so the new
    line is visible and editable instead of raising on the next preview tick.
    """
    spec = CommentedMap(type=type_name)
    for param in params(kind, type_name):
        if param.required:
            choices = CHOICES.get((type_name, param.name))
            if choices:
                if (type_name, param.name) == ("colorspace", "to"):
                    # ponytail: `choices[0]` is alphabetical, and `bgr555` — a
                    # packed 2-channel format — is first. `to_qimage` can only
                    # render single- or 3-channel frames, so that placeholder
                    # previewed as a silent blank ("nothing resolves '#0'")
                    # instead of a picture. `hsv` is an ordinary 3-channel space.
                    spec[param.name] = "hsv"
                else:
                    spec[param.name] = choices[0]
            elif param.annotation in (str, "str", "str | Path"):
                spec[param.name] = ""
            elif "tuple" in str(param.annotation):
                # ponytail: `resize.size` is the only required tuple parameter;
                # a real pair so the stage builds, matching the `display` sink's
                # shipped default.
                spec[param.name] = [640, 480]
            else:
                spec[param.name] = 0
    spec.fa.set_flow_style()
    return spec


# --------------------------------------------------------------------------- #
# YAML, comments and all
# --------------------------------------------------------------------------- #


def _yaml() -> YAML:
    yaml = YAML()  # round-trip mode: comments, key order and quoting survive
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    # Wide enough never to re-wrap a line the author wrote, which would otherwise
    # show up as a diff on every save.
    yaml.width = 4096
    return yaml


def load(path: str | Path) -> CommentedMap:
    """Read a config, keeping everything a save has to put back."""
    with open(path, encoding="utf-8") as handle:
        return _yaml().load(handle)


def save(path: str | Path, cfg: CommentedMap) -> None:
    """Write a config back. Untouched lines come out byte-for-byte unchanged."""
    with open(path, "w", encoding="utf-8") as handle:
        _yaml().dump(cfg, handle)


def _demo() -> None:
    """The three rules above, checked against the repo's own configs."""
    import io as io_module

    assert STATEFUL <= set(registered("stage"))
    assert PUBLISHES.keys() <= set(registered("stage"))

    # Every shipped stage is filed into exactly one family — added twice or not
    # at all is a bug in FAMILIES, not something the palette should hide. A
    # leading underscore (test_pipeline.py's "_test_tag") marks a registration
    # that exists only for a test's own process and was never meant to be filed
    # anywhere — the registry is process-global, so a full test run can leave
    # one behind by the time this runs.
    shipped = {name for name in registered("stage") if not name.startswith("_")}
    seen: set[str] = set()
    for name, members in FAMILIES:
        duplicate = seen & set(members)
        assert not duplicate, f"{name!r} repeats {duplicate}"
        seen |= set(members)
    assert seen == shipped, seen ^ shipped

    assert [p.name for p in params("stage", "gray")] == []
    assert [p.name for p in params("stage", "canny")] == ["lo", "hi"]

    # Live vs rebuild, on the four cases the editor actually trips over.
    assert rebuild_params("stage", {"type": "farneback"}) == frozenset()
    assert rebuild_params("stage", {"type": "mog2"}) == {
        "history",
        "var_threshold",
        "detect_shadows",
    }
    assert rebuild_params("stage", {"type": "frame_diff"}) == {"lag"}
    assert rebuild_params("stage", {"type": "static_mask"}) == {"threshold", "invert"}
    assert rebuild_params("stage", {"type": "nonsense"}) == frozenset()

    # A load/save round trip must not touch a single byte of a commented config.
    for config in sorted(Path("configs").glob("*.yaml")):
        original = config.read_text(encoding="utf-8")
        buffer = io_module.StringIO()
        _yaml().dump(load(config), buffer)
        assert buffer.getvalue() == original, f"{config} changed on round trip"

    assert new_spec("stage", "canny") == {"type": "canny"}
    assert new_spec("sink", "ffmpeg") == {"type": "ffmpeg", "path": ""}
    assert new_spec("stage", "colorspace") == {"type": "colorspace", "to": "hsv"}
    assert new_spec("stage", "resize") == {"type": "resize", "size": [640, 480]}
    print("spec ok")


if __name__ == "__main__":
    _demo()
