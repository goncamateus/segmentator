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

from PyQt6.QtCore import QRectF, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QKeySequence,
    QPainter,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from segmentator.gui import spec as spec_module
from segmentator.gui import style
from segmentator.gui.document_controller import DocumentController
from segmentator.gui.edit_ops_controller import EditOpsController
from segmentator.gui.file_controller import FileController
from segmentator.gui.paint_controller import PaintController
from segmentator.gui.playback_controller import PlaybackController
from segmentator.gui.preview_tabs_controller import PreviewTabsController
from segmentator.gui.spec import (
    CHOICES,
    FAMILIES,
    Param,
    channel_label,
    params,
    rebuild_params,
)
from segmentator.gui.optimize_worker import OptimizeWorker
from segmentator.gui.style import PALETTE, label_style
from segmentator.optimize import Finding, check, observable
from segmentator.pipeline import registered

ROW_HEIGHT = 22
# Rows shown in one AddDialog family card before it scrolls instead of growing.
# A fixed cap, not "tallest card in this grid row": a 7-member family sitting
# next to a 2-member one would otherwise dictate the whole row's height, and
# every card is meant to look like the same kind of thing regardless of how
# many stages happen to be filed under it.
CARD_ROWS = 3

# Rows shown in the sink card before it scrolls. Sinks are a single card, not
# a grid row sharing height with siblings, so it can hold far more than a
# stage family card before scrolling is worth it.
SINK_LIST_ROWS = 20


class RowDelegate(QStyledItemDelegate):
    """Paints one stage or sink row the way `docs/assets/gui-window.svg` draws it.

    The decorations are shapes rather than characters — a rounded pill for a
    `name:` tap, an amber dot for a stage that carries state between frames —
    because both are things you scan the list for while tuning, and `⟨flow⟩ ●`
    in the middle of a label is not something you can scan.
    """

    def paint(self, painter: QPainter, option, index) -> None:
        entry = index.data(Qt.ItemDataRole.UserRole) or {}
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(3, 1, -3, -1)

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.setPen(QColor(PALETTE["accent"]))
            painter.setBrush(QColor(PALETTE["accent_fill"]))
            painter.drawRoundedRect(QRectF(rect), 4, 4)
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(PALETTE["ground"]))
            painter.drawRoundedRect(QRectF(rect), 4, 4)

        font = QFont(option.font)
        font.setBold(selected)
        painter.setFont(font)
        if selected:
            colour = PALETTE["ink"]
        elif entry.get("kind") == "sink":
            colour = PALETTE["green"]
        else:
            colour = PALETTE["body"]
        text_rect = rect.adjusted(9, 0, -9, 0)
        metrics = QFontMetrics(font)
        name = entry.get("name")

        # Reserve the decorations first and elide the label into what is left:
        # a narrow list must drop characters off the type name, never paint a
        # pill on top of it.
        pill_width = metrics.horizontalAdvance(name) + 14 if name else 0
        reserved = (18 if entry.get("stateful") else 0) + (pill_width + 10 if name else 0)
        painter.setPen(QColor(colour))
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            metrics.elidedText(
                text, Qt.TextElideMode.ElideRight, max(0, text_rect.width() - reserved)
            ),
        )

        right = text_rect.right()
        if entry.get("stateful"):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(PALETTE["amber"]))
            painter.drawEllipse(QRectF(right - 8, rect.center().y() - 3.5, 7, 7))
            right -= 18

        if name:
            pill = QRectF(right - pill_width, rect.center().y() - 8, pill_width, 16)
            painter.setPen(QColor(PALETTE["amber"]))
            painter.setBrush(QColor(PALETTE["amber_fill"]))
            painter.drawRoundedRect(pill, 8, 8)
            painter.setPen(QColor(PALETTE["amber_deep"]))
            painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), name)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt name
        return QSize(super().sizeHint(option, index).width(), ROW_HEIGHT)


class AddDialog(QDialog):
    """The stage/sink palette: sectioned cards instead of one long combo.

    ``QInputDialog.getItem`` used to hold this — one combo box with all 43
    stage names in registration order, whose dropdown was a single column
    that ran off the bottom of the screen. Wide instead of tall fixes that
    directly: laid out as a grid of family cards (`FAMILIES`, the same
    grouping `docs/stages.md` and `docs/assets/stage-families.svg` use), 43
    names fit in three short rows of cards rather than one 43-row list.

    Each family gets its own small `QListWidget` — rather than one list with
    non-selectable header rows — because `RowDelegate` already knows how to
    paint a stateful stage's amber dot, and reusing it here means the dialog
    and the main lists agree on what that dot means without a second
    painter. Selecting a row in one card clears the others, so the dialog
    still behaves like a single choice.
    """

    def __init__(self, parent: QWidget, kind: str) -> None:
        super().__init__(parent)
        self.kind = kind
        self.chosen: str | None = None
        self.setWindowTitle(f"Add {kind}")
        # Half the main window's default width (1280), which is what turns
        # "one tall column" into "a few short rows of cards".
        self.setFixedWidth(640)

        layout = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(f"filter {kind}s…")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        # `body` is what QScrollArea stretches to fill the viewport once there
        # are only a couple of rows of cards left to show — a filter down to
        # three cards leaves most of the dialog's height unclaimed. The grid
        # goes inside `self._body`'s own child, `self._grid_host`, with an
        # `addStretch()` after it: a QGridLayout's own `setAlignment` only
        # takes effect when it is nested inside another layout, not when it is
        # installed directly on a widget, so it does nothing for `body` here
        # and the slack would otherwise fall to whichever child of a card is
        # not held at a fixed size — the heading label, stretched tall.
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        body_layout.addWidget(self._grid_host)
        body_layout.addStretch(1)
        area.setWidget(self._body)
        layout.addWidget(area, 1)

        # Sinks are few enough (6) that sectioning them would be one card
        # anyway; FAMILIES only covers stages.
        groups = FAMILIES if kind == "stage" else (("Sinks", tuple(registered("sink"))),)
        self._columns = 4 if kind == "stage" else 1
        self._lists: list[QListWidget] = []
        self._cards: list[QWidget] = []
        max_rows = CARD_ROWS if kind == "stage" else SINK_LIST_ROWS
        for title, members in groups:
            card, listw = self._card(title, members, max_rows)
            self._lists.append(listw)
            self._cards.append(card)

        footer = QHBoxLayout()
        if kind == "stage":
            # The same legend the catalogue figure carries, for the same reason:
            # the amber dot is the one mark in here that means something, and
            # nothing else on the dialog says what.
            legend = QLabel(
                f"<span style='color:{PALETTE['amber']}'>&#9679;</span>"
                f" <span style='color:{PALETTE['muted']}'>carries state between frames</span>"
            )
            footer.addWidget(legend)
        footer.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)

        self.resize(640, 520)
        self.filter.setFocus()
        self._relayout(self._cards)  # initial packing, same path a filter uses

    def _card(self, title: str, members: tuple[str, ...], max_rows: int) -> tuple[QWidget, QListWidget]:
        card = QFrame()
        card.setObjectName("card")
        # White panel, 1px border, and a coloured bar across the top edge — the
        # catalogue figure's card, so the palette and `docs/stages.md` group the
        # families into the same four colours.
        card.setStyleSheet(style.card_sheet(title))
        card_layout = QVBoxLayout(card)
        # Tight: a mono `brightness_contrast` needs every pixel of a card that
        # is one quarter of 640 wide, and the delegate reserves its own padding
        # inside this again.
        card_layout.setContentsMargins(4, 4, 4, 4)
        heading = QLabel(title)
        heading.setObjectName("section")
        heading.setStyleSheet(style.family_heading_style(title))
        # Fixed, not left to its sizeHint: a QGridLayout equalises every cell in
        # a row to the tallest one, and a heading is the one child in a card
        # that isn't already held to a fixed height — left free, it is what
        # absorbs that slack, stretching a short family's title down away from
        # its list.
        heading.setFixedHeight(ROW_HEIGHT)
        card_layout.addWidget(heading)

        listw = QListWidget()
        listw.setItemDelegate(RowDelegate(listw))
        listw.setFrameShape(QFrame.Shape.NoFrame)
        # Monospace, as the catalogue sets them: these are `type:` values to be
        # typed into a config, not prose, and the figure already reads them that
        # way. Sized off the app font so it tracks the 9pt the rest of the
        # editor uses rather than whatever the platform's fixed font defaults to.
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        # The hint is not belt-and-braces. On macOS that call hands back a private
        # `.AppleSystemUIFontMonospaced`, which normal family matching cannot resolve,
        # and an unresolved family falls back to a *proportional* font — the names
        # would quietly render as prose. A style hint is what the fallback consults,
        # so it picks a fixed-pitch face instead of Helvetica.
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setFixedPitch(True)
        # A point under the headings, which is the proportion the figure draws
        # them at — and monospace is wide enough that the point buys back most
        # of a name like `brightness_contrast`.
        mono.setPointSize(max(7, self.font().pointSize() - 1))
        listw.setFont(mono)
        # Horizontal stays off regardless of row count: the base
        # sizeHint().width() this delegate inherits reflects the un-elided
        # text, wider than the card, even though the delegate elides at paint
        # time to whatever width it is actually given — left at the default
        # "as needed", that mismatch shows a bar that has nothing to scroll to.
        listw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Vertical is "as needed": capped at max_rows, a family with more
        # members scrolls inside its own card rather than growing past it.
        listw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        listw.setFixedHeight(min(len(members), max_rows) * ROW_HEIGHT + 4)
        for name in members:
            item = QListWidgetItem(name)
            # Four columns in 640px leaves each card ~150px wide, and the
            # delegate elides a name like `brightness_contrast` to fit it — the
            # tooltip is what makes the elided name still readable on hover.
            item.setToolTip(name)
            marks = {
                "kind": self.kind,
                "type": name,
                "name": None,
                "stateful": self.kind == "stage" and spec_module.is_stateful(name),
            }
            item.setData(Qt.ItemDataRole.UserRole, marks)
            listw.addItem(item)
        listw.itemClicked.connect(lambda item, source=listw: self._choose(source, item.text()))
        listw.itemDoubleClicked.connect(lambda item: self._accept_choice(item.text()))
        card_layout.addWidget(listw)
        # Pinning the heading's own height (above) stops it from stretching,
        # but with every child in this layout fixed-size, Qt has nowhere
        # obvious to put a shorter card's leftover height and spreads it across
        # the margins and inter-widget spacing instead — nudging the heading
        # down along with everything below it. An explicit stretch item claims
        # that leftover space for itself, so heading and list stay pinned to
        # the top with zero slack between them, and any blank space a shorter
        # card ends up with lands after the list, where it is inert.
        card_layout.addStretch(1)
        return card, listw

    def _choose(self, source: QListWidget, name: str) -> None:
        # Only one row selected across every card — clearing the others is
        # what makes twelve independent lists read as one choice.
        for other in self._lists:
            if other is not source:
                other.blockSignals(True)
                other.setCurrentRow(-1)
                other.blockSignals(False)
        self.chosen = name
        self.ok_button.setEnabled(True)

    def _accept_choice(self, name: str) -> None:
        self.chosen = name
        self.accept()

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        visible: list[QWidget] = []
        for card, listw in zip(self._cards, self._lists):
            matched = False
            for row in range(listw.count()):
                item = listw.item(row)
                hit = needle in item.text().lower()
                listw.setRowHidden(row, not hit)
                matched = matched or hit
            card.setVisible(matched)
            if matched:
                visible.append(card)
        self._relayout(visible)

    def _relayout(self, visible_cards: list[QWidget]) -> None:
        """Repack the grid to only the visible cards, in family order.

        Rebuilds the `QGridLayout` from scratch rather than clearing and
        re-filling the existing one: `QGridLayout.takeAt` releases a widget's
        cell but the layout's row/column count — and the size it reserves for
        them — never shrinks back down, a limitation Qt does not work around.
        Reused, a filter down to three cards would still visually reserve the
        blank rows the other nine used to occupy.

        Two steps, in this order and not the other way round: first drain
        every item out with `takeAt` — which detaches a card from the layout
        without deleting it, leaving it parented to `self._grid_host` — and
        only once the old grid holds nothing do we hand it to a throwaway
        widget to be garbage-collected. Orphaning it while it still held the
        cards deletes them along with it: a `QLayout` owns whatever widgets
        are in it, and that ownership cascades.
        """
        while self._grid.count():
            self._grid.takeAt(0)
        QWidget().setLayout(self._grid)  # now-empty; safe to let go of
        self._grid = QGridLayout(self._grid_host)
        for position, card in enumerate(visible_cards):
            self._grid.addWidget(card, position // self._columns, position % self._columns)

    @staticmethod
    def get_type(parent: QWidget, kind: str) -> str | None:
        """The chosen ``type:`` name, or ``None`` if the dialog was cancelled."""
        dialog = AddDialog(parent, kind)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.chosen
        return None


class OptimizeDialog(QDialog):
    """Review what the optimizer found, and pick what to keep.

    Nothing is applied without going through here. The editor has no undo stack,
    and half of what this dialog lists rests on a finite sample of frames rather
    than a proof — so the checkbox is the safety, and what each row has to say
    for itself is the point of the column that spells out the basis.

    Pure deletions start ticked. A fusion does not: swapping two named stages for
    a 256-entry table is a straighter chain to execute and a muddier one to read,
    and that trade is the user's to make, not this dialog's.
    """

    BASIS_TEXT = {
        "identity": "provable — the parameters make it a passthrough",
        "dataflow": "provable — nothing downstream reads it",
        "exhaustive": "provable — checked on all 256 input values",
        "sampled": "sampled — no counterexample found",
    }

    def __init__(self, parent: QWidget, findings: list[Finding]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optimize")
        self.setFixedWidth(720)
        self._boxes: list[tuple[QCheckBox, Finding]] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(findings)} change(s) that leave every sink output identical:"))

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        rows = QVBoxLayout(body)
        for finding in findings:
            box = QCheckBox(self._title(finding))
            box.setChecked(finding.basis != "exhaustive")
            rows.addWidget(box)
            note = QLabel(f"      {self.BASIS_TEXT.get(finding.basis, finding.basis)} · {finding.detail}")
            note.setStyleSheet(f"color: {PALETTE['muted']};")
            rows.addWidget(note)
            self._boxes.append((box, finding))
        rows.addStretch(1)
        area.setWidget(body)
        layout.addWidget(area, 1)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {PALETTE['amber_ink']}; font-weight: 600;")
        layout.addWidget(self.warning)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(720, min(560, 140 + 52 * len(findings)))

    @staticmethod
    def _title(finding: Finding) -> str:
        where = ", ".join(str(position + 1) for position in finding.positions)
        saved = f"  −{finding.saved_ms:.1f} ms" if finding.saved_ms >= 0.05 else ""
        return f"{finding.label}   (stage {where}){saved}"

    def selected(self) -> list[Finding]:
        return [finding for box, finding in self._boxes if box.isChecked()]

    def warn(self, text: str) -> None:
        """Say why a selection was refused, without closing the dialog."""
        self.warning.setText(text)


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
        self.position: int | None = None
        self._loading = False
        self._layout = QFormLayout(self)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    def show_spec(self, kind: str, spec: dict[str, Any] | None, position: int | None = None) -> None:
        self._loading = True
        self.kind = kind
        self.spec = spec
        self.position = position
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
            label = QLabel(channel_label(type_name, param.name, spec) or param.name)
            if param.name in forced:
                label.setStyleSheet(label_style())
                # The stylesheet tints the field to match the label, so the row
                # reads as one warning rather than two.
                widget.setProperty("rebuild", True)
                widget.setToolTip(
                    "Construction parameter: changing it rebuilds the stage, and a "
                    "stage that remembers frames starts over."
                )
                label.setToolTip(widget.toolTip())
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
        if choices is not None or param.name in ("input", "draw_on", "mask"):
            combo = QComboBox()
            combo.setEditable(param.name in ("input", "draw_on", "mask"))
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
        upto = self.position if self.kind == "stage" else None
        keys = getattr(window, "image_keys", None)
        return keys(upto) if keys is not None else ("image", "source")

    def _write(self, key: str, value: Any) -> None:
        if self._loading or self.spec is None:
            return
        default = _default(self.kind, self.spec.get("type", ""), key)
        if value is None or (default is not inspect.Parameter.empty and value == default):
            self.spec.pop(key, None)
        else:
            self.spec[key] = value
        # `space` decides the channel letters ch0..ch2 are labelled with —
        # those labels are drawn once in show_spec, so a value that feeds
        # another param's label needs the form rebuilt to pick it up.
        if key == "space" and spec_module.channel_params(self.spec.get("type", "")):
            self.show_spec(self.kind, self.spec, self.position)
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
        path = Path(config)
        self.document = DocumentController(spec_module.load(path))
        self.files = FileController(self.document, path)
        self.edit_ops = EditOpsController(self.document)
        self.preview_tabs = PreviewTabsController(self.document)
        self.playback = PlaybackController(self.document)
        self.paint = PaintController()
        self._optimizer: OptimizeWorker | None = None
        self.theme = str(QSettings("segmentator", "gui").value("theme", "light"))
        self._scale = 1.0

        self.setWindowTitle(f"segmentator — {self.files.path.name}")
        self.resize(1280, 760)
        splitter = QSplitter()
        splitter.addWidget(self._lists())
        splitter.addWidget(self._form())
        splitter.addWidget(self._preview())
        splitter.setSizes([240, 250, 450])
        # Only the preview grows: the list and the form have a width their
        # contents need, and stealing it to widen a frame helps nobody.
        for index, stretch in enumerate((0, 0, 1)):
            splitter.setStretchFactor(index, stretch)
        self.setCentralWidget(splitter)
        self._menus()
        self._theme_button()
        self.statusBar().showMessage("ready")

        self.reload_lists()
        self.stage_list.setCurrentRow(0)
        self.start_worker()

    # --- construction ------------------------------------------------------- #

    def _lists(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(200)
        layout = QVBoxLayout(panel)

        layout.addWidget(self._section("Stages"))
        self.stage_list = QListWidget()
        self.stage_list.setItemDelegate(RowDelegate(self))
        self.stage_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.stage_list.currentRowChanged.connect(self.on_stage_selected)
        self.stage_list.model().rowsMoved.connect(self.on_rows_moved)
        layout.addWidget(self.stage_list, 3)
        layout.addLayout(self._buttons("stage"))

        layout.addWidget(self._section("Sinks"))
        self.sink_list = QListWidget()
        self.sink_list.setItemDelegate(RowDelegate(self))
        self.sink_list.currentRowChanged.connect(self.on_sink_selected)
        layout.addWidget(self.sink_list, 2)
        layout.addLayout(self._buttons("sink"))
        return panel

    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def _buttons(self, kind: str) -> QHBoxLayout:
        row = QHBoxLayout()
        add = QPushButton("+")
        add.setToolTip(f"add a {kind}")
        add.setFixedSize(30, 22)
        add.clicked.connect(lambda: self.add(kind))
        remove = QPushButton("−")
        remove.setToolTip(f"remove the selected {kind}")
        remove.setFixedSize(30, 22)
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
        area.setMinimumWidth(250)
        return area

    def _preview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tabs = QTabBar()
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        # Elide rather than grow scroll arrows: the tab strip is a set of things
        # to look at, and one you cannot see is worse than one that is shortened.
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabs)

        self.view = QLabel("no frame yet")
        self.view.setObjectName("preview")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(360, 240)
        layout.addWidget(self.view, 3)

        self.metrics = QTableWidget(0, 2)
        self.metrics.horizontalHeader().setVisible(False)
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.metrics.setShowGrid(False)
        self.metrics.setMaximumHeight(150)
        layout.addWidget(self.metrics, 1)

        transport = QHBoxLayout()
        for text, slot, tip in (
            ("◀◀", lambda: self.jump(-1), "previous frame"),
            ("▶", self.toggle_play, "play / pause"),
            ("▶▶", lambda: self.jump(1), "next frame"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.setFixedSize(36, 22)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(slot)
            transport.addWidget(button)
            if text == "▶":
                self.play_button = button
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider)
        transport.addWidget(self.slider, 1)
        layout.addLayout(transport)

        QShortcut(QKeySequence(","), self, activated=lambda: self.jump(-1))
        QShortcut(QKeySequence("."), self, activated=lambda: self.jump(1))
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)
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
        edit.addSeparator()
        action(edit, "&Optimize…", self.optimize)

    def _theme_button(self) -> None:
        """Day/night, in the status bar.

        Not the menu bar's corner: macOS renders the menu bar natively, and
        QMenuBar corner widgets are silently dropped there — only Windows/Linux
        show them.
        """
        self.theme_button = QToolButton()
        self.theme_button.setAutoRaise(True)
        self.theme_button.setObjectName("theme")
        self.theme_button.clicked.connect(self.toggle_theme)
        self.statusBar().addPermanentWidget(self.theme_button)
        self._show_theme()

    def _show_theme(self) -> None:
        """The glyph is the theme you are in, not the one you would get."""
        night = self.theme == "dark"
        self.theme_button.setText("☾" if night else "☀")
        self.theme_button.setToolTip(f"{'night' if night else 'day'} — click for the other")

    def toggle_theme(self) -> None:
        self.set_theme("light" if self.theme == "dark" else "dark")

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        QSettings("segmentator", "gui").setValue("theme", theme)
        self._show_theme()
        self._restyle()

    def _scale_text(self) -> None:
        """Point size follows window width: the figure's 9pt is tuned for 1280px."""
        scale = round(max(0.75, min(1.75, self.width() / 1280)), 2)
        if scale != self._scale:
            self._scale = scale
            self._restyle()

    def _restyle(self) -> None:
        """Restyle everything, including what was painted before the switch."""
        style.apply(QApplication.instance(), self.theme, self._scale)
        # Three things hold a colour of their own rather than reading the sheet:
        # the delegate paints from the live palette and only needs a repaint, the
        # form's amber labels carry an inline style, and the metrics cells were
        # given an explicit foreground when they were last written.
        for widget in (self.stage_list, self.sink_list):
            widget.viewport().update()
        self.form.show_spec(self.form.kind, self.form.spec)
        self.on_measured(*self.paint.measured)

    # --- the document ------------------------------------------------------- #
    #
    # All of this — the spec-list CRUD, label/marks/default-sink lookups — is
    # :class:`~segmentator.gui.document_controller.DocumentController`'s job.
    # What is left here is Qt: turning the controller's plain-data answers into
    # widget state, and the handful of thin pass-throughs (`cfg`, `specs`,
    # `image_keys`) other classes in this module already call by those names.

    @property
    def cfg(self) -> Any:
        return self.document.cfg

    def specs(self, kind: str) -> Any:
        return self.document.specs(kind)

    def image_keys(self, upto: int | None = None) -> tuple[str, ...]:
        return self.document.image_keys(upto)

    def reload_lists(self) -> None:
        for kind, widget in (("stage", self.stage_list), ("sink", self.sink_list)):
            row = widget.currentRow()
            widget.blockSignals(True)
            widget.clear()
            for position, entry in enumerate(self.document.specs(kind)):
                item = QListWidgetItem(self.document.label(kind, position, entry))
                item.setData(Qt.ItemDataRole.UserRole, self.document.marks(kind, entry))
                widget.addItem(item)
            widget.setCurrentRow(min(row, widget.count() - 1))
            widget.blockSignals(False)
        self.reload_tabs()

    # --- edits -------------------------------------------------------------- #

    def add(self, kind: str) -> None:
        name = AddDialog.get_type(self, kind)
        if name is None:
            return
        widget = self.stage_list if kind == "stage" else self.sink_list
        at = widget.currentRow() + 1 if widget.currentRow() >= 0 else len(self.specs(kind))
        at = self.edit_ops.add(kind, name, at)
        self.reload_lists()
        self.select(kind, at)
        self.push()

    def remove(self, kind: str) -> None:
        widget = self.stage_list if kind == "stage" else self.sink_list
        row = widget.currentRow()
        if row < 0:
            return
        next_row = self.edit_ops.remove(kind, row)
        self.reload_lists()
        self.select(kind, next_row)
        self.push()

    def remove_current(self) -> None:
        """Remove from whichever list has the focus. The menu item cannot see one."""
        self.remove("sink" if self.sink_list.hasFocus() else "stage")

    def select(self, kind: str, row: int) -> None:
        """Select a row and make the form follow it.

        Via -1 on purpose: `reload_lists` restores the old index with signals
        blocked, so selecting the same number again emits nothing and would leave
        the form describing the stage that used to be there.
        """
        widget = self.stage_list if kind == "stage" else self.sink_list
        widget.setCurrentRow(-1)
        widget.setCurrentRow(row)
        if row < 0:
            # The list was just emptied. Both calls above land on -1, so
            # `currentRowChanged` never fires and the form would otherwise keep
            # showing — and writing into — the spec dict that was just removed.
            self.form.show_spec(kind, None)

    def shift(self, delta: int) -> None:
        row = self.stage_list.currentRow()
        target = self.edit_ops.shift("stage", row, delta)
        if target is None:
            return
        self.reload_lists()
        self.select("stage", target)
        self.push()

    def on_rows_moved(self, _parent, start: int, _end: int, _dest, row: int) -> None:
        """A drag inside the list. Qt reports the insert point *before* the removal."""
        self.edit_ops.move("stage", start, row - 1 if row > start else row)
        self.reload_lists()
        self.push()

    def on_stage_selected(self, row: int) -> None:
        entries = self.specs("stage")
        self.form.show_spec("stage", entries[row] if 0 <= row < len(entries) else None, row)
        if 0 <= row < len(entries):
            self.sink_list.setCurrentRow(-1)
            self.reload_tabs()
            self.tabs.setCurrentIndex(0)

    def on_sink_selected(self, row: int) -> None:
        entries = self.specs("sink")
        if 0 <= row < len(entries):
            self.form.show_spec("sink", entries[row])
            key = entries[row].get("input", self.document.sink_default(entries[row].get("type", "")))
            index = next(
                (i for i in range(self.tabs.count()) if self.tabs.tabData(i) == key), None
            )
            if index is not None:
                self.tabs.setCurrentIndex(index)

    def on_edited(self) -> None:
        self.reload_lists()
        self.push()

    # --- optimize ------------------------------------------------------------ #

    def optimize(self) -> None:
        """Analyse the chain and offer to remove what makes no difference."""
        if not len(self.specs("stage")):
            self.statusBar().showMessage("nothing to optimize: no stages", 4000)
            return
        if self._optimizer is not None:
            return  # one at a time; the dialog below is modal anyway

        worker = OptimizeWorker(self.cfg)
        # A busy range (0, 0) until the search reports a candidate count — the
        # frame sampling in front of it has no progress to report and, on a 1080p
        # source, is a second or two of decoding on its own.
        waiting = QProgressDialog("Analysing the chain…", "Cancel", 0, 0, self)
        waiting.setWindowTitle("Optimize")
        waiting.setWindowModality(Qt.WindowModality.WindowModal)
        waiting.setMinimumDuration(0)
        waiting.canceled.connect(worker.cancel)

        def on_progress(done: int, total: int, label: str) -> None:
            waiting.setMaximum(total)
            waiting.setValue(done)
            waiting.setLabelText(f"Trying: {label}" if total else label)

        worker.progress.connect(on_progress)
        worker.done.connect(lambda findings: self.on_optimized(waiting, findings))
        worker.failed.connect(lambda text: self.on_optimize_failed(waiting, text))
        self._optimizer = worker
        worker.start()

    def on_optimize_failed(self, waiting: QProgressDialog, text: str) -> None:
        waiting.close()
        self._optimizer = None
        QMessageBox.critical(self, "optimize", text)

    def on_optimized(self, waiting: QProgressDialog, findings: list[Finding]) -> None:
        """Review the findings, re-check whatever was ticked, and apply it."""
        waiting.close()
        worker, self._optimizer = self._optimizer, None
        if worker is None:
            return
        try:
            if not findings:
                self.statusBar().showMessage("optimize: nothing to simplify", 5000)
                return
            specs = [dict(entry) for entry in self.specs("stage")]
            observed = observable([dict(entry) for entry in self.specs("sink")])
            dialog = OptimizeDialog(self, findings)
            while dialog.exec() == QDialog.DialogCode.Accepted:
                chosen = dialog.selected()
                if not chosen:
                    return
                # Each finding was judged on its own, so a selection has to be
                # judged again: two stages can each be removable alone and not
                # together — drop both greys from `gray, gray, threshold` and the
                # threshold is suddenly looking at a colour frame.
                if check(specs, chosen, worker.frames, observed):
                    self.apply_optimizations(chosen)
                    saved = sum(f.saved_ms for f in chosen)
                    self.statusBar().showMessage(
                        f"optimize: applied {len(chosen)} change(s), about {saved:.0f} ms/frame", 6000
                    )
                    return
                dialog.warn(
                    "Those changes are not equivalent when taken together, though each one is "
                    "on its own. Untick one and try again."
                )
        finally:
            worker.release()

    def apply_optimizations(self, findings: list[Finding]) -> None:
        """Rewrite the stage list in the ruamel document, bottom-up.

        Row by row rather than replacing the sequence: ruamel hangs each comment
        off its item's index, and assigning a fresh list drops every one of them,
        which would show up as the whole file rewritten on the next save.
        """
        for finding in sorted(findings, key=lambda f: f.positions[0], reverse=True):
            first, last = finding.positions[0], finding.positions[-1]
            for position in range(last, first - 1, -1):
                self.document.delete("stage", position)
            for offset, replacement in enumerate(finding.replacement):
                self.document.insert("stage", first + offset, spec_module.as_spec(replacement))
        self.reload_lists()
        self.select("stage", min(findings[0].positions[0], len(self.specs("stage")) - 1))
        self.push()

    # --- preview tabs -------------------------------------------------------- #

    def reload_tabs(self) -> None:
        row = self.stage_list.currentRow()
        wanted = self.preview_tabs.tabs(row if row >= 0 else None)

        keys = [key for _, key in wanted]
        if [self.tabs.tabData(i) for i in range(self.tabs.count())] == keys:
            return
        self.tabs.blockSignals(True)
        while self.tabs.count():
            self.tabs.removeTab(0)
        for label, key in wanted:
            self.tabs.setTabData(self.tabs.addTab(label), key)
        index = keys.index(self.preview_tabs.current) if self.preview_tabs.current in keys else 0
        self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(False)
        self.preview_tabs.select(keys[index] if keys else "source")
        self.push_wanted()

    def on_tab_changed(self, index: int) -> None:
        key = self.tabs.tabData(index)
        if key:
            self.preview_tabs.select(key)
            self.push_wanted()
            self.repaint_view()

    def push_wanted(self) -> None:
        self.playback.set_wanted((self.preview_tabs.current,))

    # --- worker -------------------------------------------------------------- #

    def start_worker(self) -> None:
        try:
            worker = self.playback.build()
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.critical(self, "source", str(exc))
            return
        worker.images_ready.connect(self.on_images)
        worker.measured.connect(self.on_measured)
        worker.position.connect(self.on_position)
        worker.status.connect(self.statusBar().showMessage)
        worker.failed.connect(lambda text: QMessageBox.critical(self, "preview", text))
        self.push_wanted()
        self.playback.launch()

    def stop_worker(self) -> None:
        self.playback.stop()

    def push(self) -> None:
        self.playback.push()

    def toggle_play(self) -> None:
        if self.playback.worker is None:
            return
        playing = self.playback.toggle_play()
        self.play_button.setText("❙❙" if playing else "▶")

    def jump(self, frames: int) -> None:
        if self.playback.worker is not None:
            self.playback.jump(frames)
            self.play_button.setText("▶")

    def on_slider(self, value: int) -> None:
        if self.playback.worker is not None:
            self.playback.seek(value)
            self.play_button.setText("▶")

    # --- painting ------------------------------------------------------------ #

    def on_images(self, images: dict) -> None:
        self.paint.receive_images(images)
        self.repaint_view()

    def repaint_view(self) -> None:
        pixmap = self.paint.pixmap_for(self.preview_tabs.current)
        if pixmap is None:
            self.view.setText(f"nothing resolves {self.preview_tabs.current!r} on this frame")
            return
        self.view.setPixmap(
            pixmap.scaled(
                self.view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def on_measured(self, index: int, metrics: dict, rows: dict) -> None:
        self.paint.receive_measurement(index, metrics, rows)
        entries = self.paint.metric_rows()
        self.metrics.setRowCount(len(entries))
        for row, (key, value) in enumerate(entries):
            name = QTableWidgetItem(str(key))
            name.setForeground(QColor(PALETTE["muted"]))
            number = QTableWidgetItem(str(value))
            number.setForeground(QColor(PALETTE["ink"]))
            number.setTextAlignment(
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            )
            self.metrics.setItem(row, 0, name)
            self.metrics.setItem(row, 1, number)
        self.metrics.resizeColumnsToContents()
        self.metrics.horizontalHeader().setStretchLastSection(True)

    def on_position(self, index: int, count: int) -> None:
        self.slider.setMaximum(max(0, count - 1))
        if not self.slider.isSliderDown():
            self.slider.setValue(index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        self.repaint_view()
        self._scale_text()

    # --- files ---------------------------------------------------------------- #

    def open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open config", "configs", "YAML (*.yaml *.yml)")
        if not path:
            return
        self.stop_worker()
        self.files.open(path)
        self.setWindowTitle(f"segmentator — {self.files.path.name}")
        self.reload_lists()
        self.start_worker()

    def save(self) -> None:
        self.files.save()
        self.statusBar().showMessage(f"saved {self.files.path}", 4000)

    def save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save config", str(self.files.path), "YAML (*.yaml)")
        if not path:
            return
        self.files.save_as(path)
        self.setWindowTitle(f"segmentator — {self.files.path.name}")
        self.save()

    def copy_command(self) -> None:
        """The batch run this window deliberately does not do."""
        command = self.files.run_command()
        QGuiApplication.clipboard().setText(command)
        self.statusBar().showMessage(f"copied: {command}", 6000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.stop_worker()
        super().closeEvent(event)
