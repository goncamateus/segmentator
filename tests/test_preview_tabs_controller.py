"""Tests for :class:`~segmentator.gui.preview_tabs_controller.PreviewTabsController`.

Needs PyQt6 only because it borrows ``preview_key`` from
:mod:`segmentator.gui.worker`; nothing here touches a widget.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from segmentator.gui.document_controller import DocumentController
from segmentator.gui.preview_tabs_controller import PreviewTabsController


def test_tabs_with_nothing_selected_and_no_sinks_is_just_source():
    document = DocumentController({"stages": [], "sinks": []})
    controller = PreviewTabsController(document)

    assert controller.tabs(None) == [("source", "source")]


def test_tabs_lead_with_the_selected_stage():
    document = DocumentController({"stages": [{"type": "gray"}, {"type": "canny"}], "sinks": []})
    controller = PreviewTabsController(document)

    assert controller.tabs(1) == [("selected: canny", "#1"), ("source", "source")]


def test_tabs_ignore_an_out_of_range_selection():
    document = DocumentController({"stages": [{"type": "gray"}], "sinks": []})
    controller = PreviewTabsController(document)

    assert controller.tabs(5) == [("source", "source")]
    assert controller.tabs(-1) == [("source", "source")]


def test_tabs_include_one_entry_per_image_sink_using_its_input_or_default():
    document = DocumentController(
        {
            "stages": [],
            "sinks": [
                {"type": "display"},
                {"type": "ffmpeg", "input": "mask"},
                {"type": "csv"},  # not image-producing: excluded
            ],
        }
    )
    controller = PreviewTabsController(document)

    assert controller.tabs(None) == [
        ("source", "source"),
        ("display ← image", "image"),
        ("ffmpeg ← mask", "mask"),
    ]


def test_select_updates_current():
    document = DocumentController({"stages": [], "sinks": []})
    controller = PreviewTabsController(document)

    assert controller.current == "source"
    controller.select("mask")
    assert controller.current == "mask"
