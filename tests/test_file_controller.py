"""Tests for :class:`~segmentator.gui.file_controller.FileController`.

Qt-free on purpose, the same split ``tests/test_document_controller.py`` draws:
the controller is constructible from a document and a path, with no
``QApplication`` and no ``MainWindow`` — picking a path is
:class:`~segmentator.gui.window.MainWindow`'s job, not this one's.
"""

from pathlib import Path

from ruamel.yaml.comments import CommentedMap

from segmentator.gui import spec as spec_module
from segmentator.gui.document_controller import DocumentController
from segmentator.gui.file_controller import FileController

SOURCE = "source:\n  type: folder\n  path: .\n"


def test_open_loads_the_new_config_into_the_documents_cfg_in_place(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(SOURCE + "stages:\n  - {type: gray}\n", encoding="utf-8")
    b = tmp_path / "b.yaml"
    b.write_text(SOURCE + "stages:\n  - {type: canny}\n", encoding="utf-8")

    document = DocumentController(spec_module.load(a))
    controller = FileController(document, a)

    controller.open(b)

    assert controller.path == b
    assert controller.document is document  # same instance; cfg was swapped in place
    assert [s["type"] for s in document.specs("stage")] == ["canny"]


def test_save_writes_the_documents_current_cfg_to_the_controllers_path(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(SOURCE + "stages:\n  - {type: gray}\n", encoding="utf-8")
    document = DocumentController(spec_module.load(a))
    controller = FileController(document, a)

    document.specs("stage").append(spec_module.new_spec("stage", "canny"))
    controller.save()

    reloaded = spec_module.load(a)
    assert [s["type"] for s in reloaded["stages"]] == ["gray", "canny"]


def test_save_as_redirects_the_path_without_writing_by_itself(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(SOURCE + "stages: []\n", encoding="utf-8")
    document = DocumentController(spec_module.load(a))
    controller = FileController(document, a)

    b = tmp_path / "b.yaml"
    controller.save_as(b)

    assert controller.path == b
    assert not b.exists()  # save_as only redirects; the write is a separate call

    controller.save()
    assert b.exists()
    assert not (tmp_path / "a-unchanged-marker").exists()  # sanity: a.yaml untouched by the redirect


def test_run_command_names_the_batch_entry_point_for_the_current_path():
    document = DocumentController(CommentedMap(source={"type": "folder", "path": "."}))
    controller = FileController(document, Path("configs/baseline.yaml"))

    assert controller.run_command() == "uv run segmentator configs/baseline.yaml"
