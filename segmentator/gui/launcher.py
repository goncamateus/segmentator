"""The window `segmentator-gui` opens with when it is given no config.

What it replaces is a bare `QFileDialog` parented to nothing: the first thing a
double-clicked bundle showed was a file browser, and cancelling it ended the
process. `docs/assets/gui-launcher.svg` is the figure, and — as everywhere else
here — the figure is what the window renders rather than a sketch of it.

Two ways in, and they are not symmetrical. **Open** is the old picker, kept. **New**
is the path that did not exist at all: a config is the unit of work in this
project, and until now there was no way to get a first one except to write the
YAML by hand. So New asks for the thing a config cannot be written without — a
video — and derives the rest.

:func:`starter_config` is deliberately free of Qt. It is the half with a claim to
check (a config that previews without an edit), and a test should not need a
`QApplication` to check it.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from segmentator.gui import style

VIDEO_FILTER = "Video (*.mp4 *.avi *.mov *.mkv);;All files (*)"
CONFIG_FILTER = "YAML (*.yaml *.yml)"

# Half the main window's 1280, the same reasoning `AddDialog` uses for its width;
# the height then holds the proportion the figure draws.
WIDTH, HEIGHT = 640, 500
LOGO = 256
# The corner the platforms round the icon to anyway. The source PNG is a full-bleed
# square with its own near-black ground, which on the day palette lands as a black
# box; rounded, it reads as the app icon it is.
LOGO_RADIUS = 0.22

# No stages: a source and a `display` sink over it is already a valid chain, so
# the editor comes up with a live preview rather than an error box, and the
# first stage the user adds is a real choice instead of a placeholder to
# delete. Comments and all: a config the editor writes should read like the
# ones shipped in `configs/`.
STARTER = """\
# {stem} — a new pipeline. Add stages and sinks from the Pipeline menu.
name: {name}

source:
  type: video
  path: {path}

stages: []

sinks:
  - {{type: display, size: [640, 480]}}
"""


def icon_path() -> Path:
    """The app icon, wherever this is running from.

    `importlib.resources` rather than a path relative to `__file__` or to the
    working directory: the same call resolves in a checkout, in an installed
    wheel, and inside the PyInstaller bundle, which starts with `cwd` at `/`.
    """
    return Path(str(files("segmentator.gui") / "assets" / "icon.png"))


def app_version() -> str:
    """The packaged version, or an empty string if there is no metadata to read."""
    try:
        return version("segmentator")
    except PackageNotFoundError:
        return ""


def starter_config(video: Path) -> Path:
    """Write a first config for `video` and return where it landed.

    Never overwrites: a name already taken gets a `-2`, `-3` suffix instead. The
    config is the user's document, and the one thing worse than not creating it
    is destroying the one they already had.
    """
    video = Path(video)
    # ponytail: `configs/` when it is already there — a dev checkout run from the
    # repo root — and beside the video otherwise, because a frozen bundle runs
    # with cwd at `/` and a relative `configs/` would land somewhere unwritable
    # and unfindable. A real preference, remembered between runs, is the upgrade.
    # Resolved, never relative: the path is handed to `MainWindow`, which keeps it
    # for every later Save, and a relative one would follow the working directory.
    folder = Path("configs").resolve() if Path("configs").is_dir() else video.parent
    stem = video.stem
    path = folder / f"{stem}.yaml"
    count = 1
    while path.exists():
        count += 1
        path = folder / f"{stem}-{count}.yaml"
    # `json.dumps` for both scalars: a double-quoted JSON string is also a valid
    # YAML one, and it escapes the colons and backslashes an interpolated
    # filename would otherwise hand straight to the parser. The name needs it as
    # much as the path does — `foo: bar.mp4` is a legal filename.
    body = STARTER.format(stem=stem, name=json.dumps(stem), path=json.dumps(str(video)))
    path.write_text(body, encoding="utf-8")
    return path


class LauncherWindow(QDialog):
    """Logo and version on one side, the two ways in on the other."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("segmentator")
        # The one window in the app with a fixed size: it has no content that can
        # grow, and the figure's proportion is the whole of its layout.
        self.setFixedSize(WIDTH, HEIGHT)
        self.picked: str | None = None

        layout = QHBoxLayout(self)
        layout.addWidget(self._brand(), 3)
        layout.addWidget(self._rule())
        layout.addWidget(self._actions(), 2)

    # --- construction ------------------------------------------------------- #

    def _brand(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)

        logo = QLabel()
        tile = self._logo_tile()
        if tile is not None:
            logo.setPixmap(tile)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        column.addStretch(1)
        column.addWidget(logo)
        column.addStretch(1)

        self.version_label = QLabel(f"version: {app_version()}")
        self.version_label.setObjectName("hint")  # `{muted}`, per the sheet
        self.version_label.setVisible(bool(app_version()))
        column.addWidget(self.version_label)
        return panel

    def _logo_tile(self) -> QPixmap | None:
        """The icon at `LOGO` points, rounded, and sharp on a HiDPI screen.

        Drawn at `LOGO × devicePixelRatio` device pixels with the ratio stamped on
        the result: the source is 1024² and has the detail to spend, so scaling to
        logical points alone would throw away half of it on a Retina display.
        """
        icon = QPixmap(str(icon_path()))
        if icon.isNull():
            return None
        ratio = self.devicePixelRatioF()
        side = round(LOGO * ratio)
        scaled = icon.scaled(
            side,
            side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        tile = QPixmap(scaled.size())
        tile.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tile)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        corner = scaled.width() * LOGO_RADIUS
        path = QPainterPath()
        path.addRoundedRect(QRectF(scaled.rect()), corner, corner)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, scaled)
        painter.end()

        tile.setDevicePixelRatio(ratio)
        return tile

    def _rule(self) -> QFrame:
        """The divider. A styled 1px frame, not `QFrame.Shape.VLine`.

        Fusion paints a VLine as an etched two-tone groove out of the desktop
        palette, which is a different grey from the `{rule}` every other seam in
        the editor is drawn in.
        """
        line = QFrame()
        line.setFixedWidth(1)
        # Resolved at construction, which is safe for the reason `style.card_sheet`
        # documents: this window is modal and short-lived, so no theme switch can
        # happen underneath it.
        line.setStyleSheet(f"background: {style.PALETTE['rule']};")
        return line

    def _actions(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setSpacing(16)  # the gap the figure draws — half a button apart
        column.addStretch(1)
        for text, name in (("New Project", "new"), ("Open Project", "open")):
            button = QPushButton(text)
            button.setFixedSize(160, 30)
            button.setObjectName(name)
            button.clicked.connect(lambda _, choice=name: self._pick(choice))
            column.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)
        column.addStretch(1)
        return panel

    # --- the two ways in ------------------------------------------------------ #

    def _pick(self, choice: str) -> None:
        self.picked = choice
        self.accept()


def choose(parent: QWidget | None = None) -> Path | None:
    """Run the launcher and return the config to open, or `None` to quit.

    Loops rather than returning on a cancelled file dialog: backing out of the
    video picker is "I did not mean that", not "close the application".
    """
    while True:
        window = LauncherWindow(parent)
        if window.exec() != QDialog.DialogCode.Accepted:
            return None
        if window.picked == "new":
            video, _ = QFileDialog.getOpenFileName(parent, "Video for the new project", "", VIDEO_FILTER)
            if video:
                return starter_config(Path(video))
        else:
            config, _ = QFileDialog.getOpenFileName(parent, "Open config", "configs", CONFIG_FILTER)
            if config:
                return Path(config)
