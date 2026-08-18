"""The window: a stage list, a form generated from the signature, and a preview.

The config is a linear list of stages, so **the list is the graph** — drawn at
1:1 rather than as a straight line of boxes on a canvas. Where a config does
branch, it branches by *name*: a `select` stage or a sink reading a tap shows up
as a named reference, which this window makes clickable but does not draw as an
edge.

This class owns no image processing at all. Its whole job is:

    an edit          ->  update the CommentedMap  ->  hand fresh specs to the worker
    worker emitted   ->  paint it

Every edit goes into the ruamel document, never into a parallel model, so what
is saved is the file that was opened with the edits applied — comments included.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from segmentator.gui import spec as spec_module
from segmentator.gui.spec import CHOICES, STATEFUL, Param, params, rebuild_params
from segmentator.gui.worker import PreviewWorker, preview_key
from segmentator.pipeline import registered

REBUILD_STYLE = "color: #b44d12; font-weight: 600;"
SINK_IMAGE_TYPES = ("display", "ffmpeg", "image", "crops")


def move(seq: Any, src: int, dst: int) -> None:
    """Move one item of a ruamel sequence, taking its comment with it.

    ponytail: a standalone comment block *between* two items belongs to whichever
    item ruamel attached it to, so reordering can leave one behind. Reordering a
    commented stage is rare enough to not be worth a comment-reparenting pass.
    """
    comments = {index: seq.ca.items.get(index) for index in range(len(seq))}
    item = seq.pop(src)
    moved = comments.pop(src, None)
    order = [comments[index] for index in sorted(comments)]
    seq.insert(dst, item)
    order.insert(dst, moved)
    seq.ca.items.clear()
    for index, comment in enumerate(order):
        if comment is not None:
            seq.ca.items[index] = comment


class ParamForm(QWidget):
    """The parameter rows for one component, generated from its signature.

    A row writes its key into the spec only when it differs from the constructor
    default, and deletes it when it goes back — so a config stays as terse as the
    one that was opened instead of growing every default on first click.
    """

    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.kind = "stage"
        self.spec: dict[str, Any] | None = None
        self._loading = False
        self._layout = QFormLayout(self)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    def show_spec(self, kind: str, spec: dict[str, Any] | None) -> None:
        self._loading = True
        self.kind = kind
        self.spec = spec
        while self._layout.count():
            self._layout.takeAt(0).widget().deleteLater()
        if spec is None:
            self._loading = False
            return

        type_name = spec.get("type", "")
        header = QLabel(f"<b>{type_name}</b><br><span style='color:#7b8794'>"
                        f"generated from the signature</span>")
        self._layout.addRow(header)
        try:
            rows = params(kind, type_name)
        except KeyError as exc:
            self._layout.addRow(QLabel(str(exc)))
            self._loading = False
            return

        forced = rebuild_params(kind, spec)
        for param in rows:
            widget = self._widget(type_name, param, spec.get(param.name, param.default))
            label = QLabel(param.name)
            if param.name in forced:
                label.setStyleSheet(REBUILD_STYLE)
                label.setToolTip(
                    "Construction parameter: changing it rebuilds the stage, and a "
                    "stage that remembers frames starts over."
                )
            self._layout.addRow(label, widget)

        if kind == "stage":
            name = QLineEdit(str(spec.get("name", "")))
            name.setPlaceholderText("untapped")
            name.textChanged.connect(lambda text: self._write("name", text or None))
            self._layout.addRow(QLabel("name (tap)"), name)
        self._loading = False

    # --- one widget per parameter ------------------------------------------ #

    def _widget(self, type_name: str, param: Param, value: Any) -> QWidget:
        default = None if param.required else param.default
        choices = CHOICES.get((type_name, param.name))
        if isinstance(default, bool) or isinstance(value, bool):
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda state: self._write(param.name, state))
            return box
        if choices is not None or param.name == "input" or param.name == "draw_on":
            combo = QComboBox()
            combo.setEditable(param.name in ("input", "draw_on"))
            combo.addItems(list(choices or self._image_keys()))
            combo.setCurrentText("" if value is None else str(value))
            combo.currentTextChanged.connect(lambda text: self._write(param.name, text))
            return combo
        if isinstance(default, int) and not isinstance(default, bool):
            spin = QSpinBox()
            spin.setRange(-1_000_000, 1_000_000)
            spin.setValue(int(value if value is not None else 0))
            spin.valueChanged.connect(lambda number: self._write(param.name, number))
            return spin
        if isinstance(default, float):
            spin = QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(float(value if value is not None else 0.0))
            spin.valueChanged.connect(lambda number: self._write(param.name, number))
            return spin
        if isinstance(default, (tuple, list)):
            line = QLineEdit(", ".join(str(v) for v in (value or default)))
            line.setPlaceholderText("comma separated")
            line.textChanged.connect(lambda text: self._write(param.name, _as_list(text)))
            return line
        line = QLineEdit("" if value is None else str(value))
        line.textChanged.connect(lambda text: self._write(param.name, text))
        return line

    def _image_keys(self) -> tuple[str, ...]:
        window = self.window()
        return getattr(window, "image_keys", lambda: ("image", "source"))()

    def _write(self, key: str, value: Any) -> None:
        if self._loading or self.spec is None:
            return
        default = _default(self.kind, self.spec.get("type", ""), key)
        if value is None or (default is not inspect.Parameter.empty and value == default):
            self.spec.pop(key, None)
        else:
            self.spec[key] = value
        self.changed.emit()


def _as_list(text: str) -> list[int] | None:
    try:
        return [int(part) for part in text.split(",") if part.strip()] or None
    except ValueError:
        return None


def _default(kind: str, type_name: str, key: str) -> Any:
    if key == "name":
        return None
    for param in params(kind, type_name):
        if param.name == key:
            return param.default
    return inspect.Parameter.empty


class MainWindow(QMainWindow):
    """Stage list, parameter form, preview — and the transport under it."""

    def __init__(self, config: str | Path):
        super().__init__()
        self.path = Path(config)
        self.cfg = spec_module.load(self.path)
        self.worker: PreviewWorker | None = None
        self._images: dict[str, QPixmap] = {}
        self._current = "source"

        self.setWindowTitle(f"segmentator — {self.path.name}")
        self.resize(1280, 760)
        splitter = QSplitter()
        splitter.addWidget(self._lists())
        splitter.addWidget(self._form())
        splitter.addWidget(self._preview())
        splitter.setSizes([260, 260, 760])
        self.setCentralWidget(splitter)
        self._menus()
        self.statusBar().showMessage("ready")

        self.reload_lists()
        self.stage_list.setCurrentRow(0)
        self.start_worker()

    # --- construction ------------------------------------------------------- #

    def _lists(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("<b>Stages</b>"))
        self.stage_list = QListWidget()
        self.stage_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.stage_list.currentRowChanged.connect(self.on_stage_selected)
        self.stage_list.model().rowsMoved.connect(self.on_rows_moved)
        layout.addWidget(self.stage_list, 3)
        layout.addLayout(self._buttons("stage"))

        layout.addWidget(QLabel("<b>Sinks</b>"))
        self.sink_list = QListWidget()
        self.sink_list.currentRowChanged.connect(self.on_sink_selected)
        layout.addWidget(self.sink_list, 2)
        layout.addLayout(self._buttons("sink"))
        return panel

    def _buttons(self, kind: str) -> QHBoxLayout:
        row = QHBoxLayout()
        add = QPushButton("+")
        add.setToolTip(f"add a {kind}")
        add.clicked.connect(lambda: self.add(kind))
        remove = QPushButton("−")
        remove.setToolTip(f"remove the selected {kind}")
        remove.clicked.connect(lambda: self.remove(kind))
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        return row

    def _form(self) -> QWidget:
        self.form = ParamForm()
        self.form.changed.connect(self.on_edited)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.form)
        return area

    def _preview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tabs = QTabBar()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabs)

        self.view = QLabel("no frame yet")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(360, 240)
        self.view.setStyleSheet("background: #12161c; color: #7b8794;")
        layout.addWidget(self.view, 3)

        self.metrics = QTableWidget(0, 2)
        self.metrics.horizontalHeader().setVisible(False)
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics.setMaximumHeight(150)
        layout.addWidget(self.metrics, 1)

        transport = QHBoxLayout()
        for text, slot, tip in (
            ("◀◀", lambda: self.jump(-10), "back 10 frames"),
            ("◀", lambda: self.jump(-1), "previous frame"),
            ("▶", self.toggle_play, "play / pause"),
            ("▶▶", lambda: self.jump(1), "next frame"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.setFixedWidth(44)
            button.clicked.connect(slot)
            transport.addWidget(button)
            if text == "▶":
                self.play_button = button
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider)
        transport.addWidget(self.slider, 1)
        layout.addLayout(transport)
        return panel

    def _menus(self) -> None:
        def action(menu, text, slot, shortcut=None):
            item = QAction(text, self)
            item.triggered.connect(slot)
            if shortcut:
                item.setShortcut(QKeySequence(shortcut))
            menu.addAction(item)

        file_menu = self.menuBar().addMenu("&File")
        action(file_menu, "&Open…", self.open, "Ctrl+O")
        action(file_menu, "&Save", self.save, "Ctrl+S")
        action(file_menu, "Save &As…", self.save_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        action(file_menu, "&Copy run command", self.copy_command)
        action(file_menu, "&Quit", self.close, "Ctrl+Q")

        edit = self.menuBar().addMenu("&Pipeline")
        action(edit, "Add &stage…", lambda: self.add("stage"), "Ctrl+N")
        action(edit, "Add sin&k…", lambda: self.add("sink"))
        action(edit, "&Remove selected", self.remove_current, "Ctrl+D")
        edit.addSeparator()
        action(edit, "Move &up", lambda: self.shift(-1), "Ctrl+Up")
        action(edit, "Move &down", lambda: self.shift(1), "Ctrl+Down")

    # --- the document ------------------------------------------------------- #

    def specs(self, kind: str) -> Any:
        key = "stages" if kind == "stage" else "sinks"
        if key not in self.cfg:
            self.cfg[key] = []
        return self.cfg[key]

    def image_keys(self) -> tuple[str, ...]:
        """Everything a sink's ``input:`` or a ``draw_on:`` can currently resolve."""
        names = [s.get("name") for s in self.specs("stage") if s.get("name")]
        return ("image", "source", *names, "mask", "heat", "histogram")

    def reload_lists(self) -> None:
        for kind, widget in (("stage", self.stage_list), ("sink", self.sink_list)):
            row = widget.currentRow()
            widget.blockSignals(True)
            widget.clear()
            for position, entry in enumerate(self.specs(kind)):
                widget.addItem(QListWidgetItem(self._label(kind, position, entry)))
            widget.setCurrentRow(min(row, widget.count() - 1))
            widget.blockSignals(False)
        self.reload_tabs()

    def _label(self, kind: str, position: int, entry: dict[str, Any]) -> str:
        type_name = entry.get("type", "?")
        if kind == "sink":
            return f"{type_name}  ← {entry.get('input', self._sink_default(type_name))}"
        marks = ""
        if entry.get("name"):
            marks += f"   ⟨{entry['name']}⟩"
        if type_name in STATEFUL:
            marks += "  ●"
        return f"{position + 1}. {type_name}{marks}"

    def _sink_default(self, type_name: str) -> str:
        if type_name == "csv":
            return "rows"
        if type_name == "json":
            return "metrics"
        return "source" if type_name == "crops" else "image"

    # --- edits -------------------------------------------------------------- #

    def add(self, kind: str) -> None:
        names = registered(kind)
        name, ok = QInputDialog.getItem(self, f"Add {kind}", f"{kind}:", names, 0, False)
        if not ok:
            return
        widget = self.stage_list if kind == "stage" else self.sink_list
        at = widget.currentRow() + 1 if widget.currentRow() >= 0 else len(self.specs(kind))
        self.specs(kind).insert(at, spec_module.new_spec(kind, name))
        self.reload_lists()
        widget.setCurrentRow(at)
        self.push()

    def remove(self, kind: str) -> None:
        widget = self.stage_list if kind == "stage" else self.sink_list
        row = widget.currentRow()
        if row < 0:
            return
        del self.specs(kind)[row]
        self.reload_lists()
        self.push()

    def remove_current(self) -> None:
        """Remove from whichever list has the focus. The menu item cannot see one."""
        self.remove("sink" if self.sink_list.hasFocus() else "stage")

    def shift(self, delta: int) -> None:
        row = self.stage_list.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < len(self.specs("stage")):
            return
        move(self.specs("stage"), row, target)
        self.reload_lists()
        self.stage_list.setCurrentRow(target)
        self.push()

    def on_rows_moved(self, _parent, start: int, _end: int, _dest, row: int) -> None:
        """A drag inside the list. Qt reports the insert point *before* the removal."""
        move(self.specs("stage"), start, row - 1 if row > start else row)
        self.reload_lists()
        self.push()

    def on_stage_selected(self, row: int) -> None:
        entries = self.specs("stage")
        self.form.show_spec("stage", entries[row] if 0 <= row < len(entries) else None)
        if 0 <= row < len(entries):
            self.sink_list.setCurrentRow(-1)
            self.reload_tabs()
            self.tabs.setCurrentIndex(0)

    def on_sink_selected(self, row: int) -> None:
        entries = self.specs("sink")
        if 0 <= row < len(entries):
            self.form.show_spec("sink", entries[row])
            key = entries[row].get("input", self._sink_default(entries[row].get("type", "")))
            index = next(
                (i for i in range(self.tabs.count()) if self.tabs.tabData(i) == key), None
            )
            if index is not None:
                self.tabs.setCurrentIndex(index)

    def on_edited(self) -> None:
        self.reload_lists()
        self.push()

    # --- preview tabs -------------------------------------------------------- #

    def reload_tabs(self) -> None:
        wanted = [("source", "source")]
        row = self.stage_list.currentRow()
        if row >= 0:
            entry = self.specs("stage")[row]
            wanted.insert(0, (f"selected: {entry.get('type', '?')}", preview_key(row)))
        for entry in self.specs("sink"):
            type_name = entry.get("type", "?")
            if type_name not in SINK_IMAGE_TYPES:
                continue
            key = entry.get("input", self._sink_default(type_name))
            wanted.append((f"{type_name} ← {key}", key))

        keys = [key for _, key in wanted]
        if [self.tabs.tabData(i) for i in range(self.tabs.count())] == keys:
            return
        self.tabs.blockSignals(True)
        while self.tabs.count():
            self.tabs.removeTab(0)
        for label, key in wanted:
            self.tabs.setTabData(self.tabs.addTab(label), key)
        index = keys.index(self._current) if self._current in keys else 0
        self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(False)
        self._current = keys[index] if keys else "source"
        self.push_wanted()

    def on_tab_changed(self, index: int) -> None:
        key = self.tabs.tabData(index)
        if key:
            self._current = key
            self.push_wanted()
            self.repaint_view()

    def push_wanted(self) -> None:
        if self.worker is not None:
            self.worker.wanted = (self._current,)

    # --- worker -------------------------------------------------------------- #

    def start_worker(self) -> None:
        self.stop_worker()
        try:
            worker = PreviewWorker(self.cfg)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.critical(self, "source", str(exc))
            return
        worker.images_ready.connect(self.on_images)
        worker.measured.connect(self.on_measured)
        worker.position.connect(self.on_position)
        worker.status.connect(self.statusBar().showMessage)
        worker.failed.connect(lambda text: QMessageBox.critical(self, "preview", text))
        self.worker = worker
        self.push_wanted()
        worker.start()

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def push(self) -> None:
        """Hand the worker a fresh snapshot of the specs. Never mutate in place."""
        if self.worker is None:
            return
        self.worker.source_spec = dict(self.cfg["source"])
        self.worker.specs = tuple(dict(entry) for entry in self.specs("stage"))

    def toggle_play(self) -> None:
        if self.worker is None:
            return
        self.worker.playing = not self.worker.playing
        self.play_button.setText("❙❙" if self.worker.playing else "▶")

    def jump(self, frames: int) -> None:
        if self.worker is not None:
            self.worker.step(frames)
            self.play_button.setText("▶")

    def on_slider(self, value: int) -> None:
        if self.worker is not None:
            self.worker.playing = False
            self.play_button.setText("▶")
            self.worker.seek(value)

    # --- painting ------------------------------------------------------------ #

    def on_images(self, images: dict) -> None:
        self._images = {key: QPixmap.fromImage(image) for key, image in images.items()}
        self.repaint_view()

    def repaint_view(self) -> None:
        pixmap = self._images.get(self._current)
        if pixmap is None:
            self.view.setText(f"nothing resolves {self._current!r} on this frame")
            return
        self.view.setPixmap(
            pixmap.scaled(
                self.view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def on_measured(self, _index: int, metrics: dict, rows: dict) -> None:
        entries = list(metrics.items()) + [(f"{kind} rows", len(r)) for kind, r in rows.items()]
        self.metrics.setRowCount(len(entries))
        for row, (key, value) in enumerate(entries):
            self.metrics.setItem(row, 0, QTableWidgetItem(str(key)))
            self.metrics.setItem(row, 1, QTableWidgetItem(str(value)))
        self.metrics.resizeColumnsToContents()

    def on_position(self, index: int, count: int) -> None:
        self.slider.setMaximum(max(0, count - 1))
        if not self.slider.isSliderDown():
            self.slider.setValue(index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        self.repaint_view()

    # --- files ---------------------------------------------------------------- #

    def open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open config", "configs", "YAML (*.yaml *.yml)")
        if not path:
            return
        self.stop_worker()
        self.path = Path(path)
        self.cfg = spec_module.load(self.path)
        self.setWindowTitle(f"segmentator — {self.path.name}")
        self.reload_lists()
        self.start_worker()

    def save(self) -> None:
        spec_module.save(self.path, self.cfg)
        self.statusBar().showMessage(f"saved {self.path}", 4000)

    def save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save config", str(self.path), "YAML (*.yaml)")
        if not path:
            return
        self.path = Path(path)
        self.setWindowTitle(f"segmentator — {self.path.name}")
        self.save()

    def copy_command(self) -> None:
        """The batch run this window deliberately does not do."""
        command = f"uv run segmentator {self.path}"
        QGuiApplication.clipboard().setText(command)
        self.statusBar().showMessage(f"copied: {command}", 6000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.stop_worker()
        super().closeEvent(event)
