"""The look, in one place: the palettes `docs/assets/gui-window.svg` draws.

The figure is the spec — a white window, `#f5f7fa` fields, `#cbd2d9` borders, a
blue selection, and amber for the two things that matter while tuning: a `name:`
tap, and a stage that remembers frames. The editor forces one of its own two
palettes rather than inheriting the desktop theme, so the marks that carry
meaning keep the colour they are documented with.

Two mechanisms, and both are needed:

* the **stylesheet** styles what the window builds itself;
* the **palette** covers what Qt builds for us — `QInputDialog`, `QFileDialog`,
  `QMessageBox`, tooltips — none of which the sheet's selectors reach, and all
  of which come back in the desktop theme without it.

`PALETTE` is the *current* one, mutated in place by :func:`apply` rather than
rebound, so anything holding a reference — `RowDelegate` paints straight out of
it — follows a theme switch without being told.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette

LIGHT = {
    "panel": "#ffffff",  # the window itself, and the list/table panels on it
    "ground": "#f5f7fa",  # input fills and buttons
    "border": "#cbd2d9",
    "rule": "#e4e7eb",
    "ink": "#1f2933",
    "head": "#3e4c59",
    "body": "#52606d",
    "muted": "#7b8794",
    "faint": "#9aa5b1",
    "accent": "#1f6feb",
    "accent_fill": "#e3f0ff",
    "accent_deep": "#d3e4fb",
    "amber": "#f0b429",
    "amber_fill": "#fdf3d8",
    "amber_ink": "#b44d12",
    "amber_deep": "#5c3d00",
    "green": "#1f8a70",
    "canvas": "#12161c",  # behind the preview frame
}

# Night. Same keys, same roles, and deliberately the greys the documentation's
# code blocks already use. Two things do not simply invert: `accent` lightens,
# because #1f6feb on a dark panel is a hole rather than a highlight, and
# `amber_fill` becomes a dark tint that the amber border still reads against —
# the amber marks have to stay recognisably the same marks.
DARK = {
    "panel": "#1f2933",
    "ground": "#171e26",
    "border": "#3e4c59",
    "rule": "#2b3743",
    "ink": "#e8eef5",
    "head": "#cbd2d9",
    "body": "#9fb3c8",
    "muted": "#7b8794",
    "faint": "#616e7c",
    "accent": "#6ea8fe",
    "accent_fill": "#24344d",
    "accent_deep": "#2d4266",
    "amber": "#f0b429",
    "amber_fill": "#3a2d10",
    "amber_ink": "#f0b429",
    "amber_deep": "#f7d774",
    "green": "#4fd1a5",
    "canvas": "#0b0e12",
}

THEMES = {"light": LIGHT, "dark": DARK}

# The live palette. Mutated, never rebound — see the module docstring.
PALETTE = dict(LIGHT)

MONO = "ui-monospace, SFMono-Regular, Menlo, DejaVu Sans Mono, monospace"

TEMPLATE = """
QMainWindow, QDialog, QWidget {{ background: {panel}; color: {head}; }}
QLabel {{ background: transparent; color: {body}; }}
QLabel#section {{ color: {head}; font-weight: 600; }}
QLabel#hint {{ color: {muted}; }}
QLabel#preview {{ background: {canvas}; color: {muted}; border-radius: 4px; }}

QListWidget, QTableWidget {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 5px;
    outline: 0;
}}
QTableWidget::item {{ padding: 1px 6px; }}

QPushButton {{
    background: {ground};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 2px 8px;
    color: {head};
}}
QPushButton:hover {{ background: {accent_fill}; border-color: {accent}; }}
QPushButton:pressed {{ background: {accent_deep}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {ground};
    border: 1px solid {border};
    border-radius: 3px;
    padding: 1px 6px;
    color: {ink};
    selection-background-color: {accent_fill};
    selection-color: {ink};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {accent}; }}
/* A construction parameter: changing it rebuilds the stage. Same amber the
   label uses, so the row reads as one warning rather than two. */
QLineEdit[rebuild="true"], QSpinBox[rebuild="true"],
QDoubleSpinBox[rebuild="true"], QComboBox[rebuild="true"] {{
    background: {amber_fill};
    border-color: {amber};
}}
/* The drop-down box is left unstyled on purpose: Fusion paints the chevron
   inside it, and a QSS border triangle renders as a grey square instead. */
/* No spin arrows: the figure has plain fields, and typing or scrolling is how
   these get set anyway. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: 0; }}
QComboBox QAbstractItemView {{
    background: {panel};
    border: 1px solid {border};
    selection-background-color: {accent_fill};
    selection-color: {ink};
}}
QCheckBox {{ background: transparent; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {border};
    border-radius: 2px;
    background: {panel};
}}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}

QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: {rule};
    color: {body};
    padding: 3px 9px;
    margin-right: 4px;
    border-radius: 3px;
}}
QTabBar::tab:selected {{ background: {accent}; color: {panel}; }}
QTabBar::tab:!selected:hover {{ background: {accent_fill}; color: {ink}; }}

QSlider::groove:horizontal {{ height: 6px; background: {rule}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {accent};
    width: 10px;
    margin: -3px 0;
    border-radius: 5px;
}}

QStatusBar {{ background: {panel}; color: {muted}; font-family: {mono};
              border-top: 1px solid {rule}; }}
QStatusBar::item {{ border: 0; }}
QStatusBar QSizeGrip {{ width: 0; height: 0; }}

QMenuBar {{ background: {panel}; color: {body}; border-bottom: 1px solid {rule}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {accent_fill}; color: {ink}; }}
QMenu {{ background: {panel}; border: 1px solid {border}; }}
QMenu::item {{ padding: 4px 24px 4px 20px; }}
QMenu::item:selected {{ background: {accent_fill}; color: {ink}; }}

QToolButton#theme {{
    background: transparent;
    border: 0;
    color: {muted};
    font-size: 13pt;
    padding: 0 10px 2px 10px;
}}
QToolButton#theme:hover {{ color: {amber}; }}

QSplitter::handle {{ background: {rule}; }}
QScrollArea {{ border: 0; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {panel}; border: 0; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {border}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {ink}; color: {panel}; border: 0; padding: 4px; }}
"""


def sheet(colours: dict[str, str]) -> str:
    """The stylesheet for one palette."""
    return TEMPLATE.format(mono=MONO, **colours)


def palette(colours: dict[str, str]) -> QPalette:
    """A QPalette, for everything the stylesheet's selectors cannot reach."""
    roles = {
        QPalette.ColorRole.Window: "panel",
        QPalette.ColorRole.WindowText: "head",
        QPalette.ColorRole.Base: "panel",
        QPalette.ColorRole.AlternateBase: "ground",
        QPalette.ColorRole.Text: "ink",
        QPalette.ColorRole.Button: "ground",
        QPalette.ColorRole.ButtonText: "head",
        QPalette.ColorRole.ToolTipBase: "ink",
        QPalette.ColorRole.ToolTipText: "panel",
        QPalette.ColorRole.Highlight: "accent_fill",
        QPalette.ColorRole.HighlightedText: "ink",
        QPalette.ColorRole.PlaceholderText: "faint",
    }
    built = QPalette()
    for role, key in roles.items():
        built.setColor(role, QColor(colours[key]))
    return built


def apply(app, theme: str = "light") -> None:
    """Style one QApplication. Call it before the window is built.

    Raises:
        KeyError: on an unknown theme name, listing the known ones.
    """
    if theme not in THEMES:
        raise KeyError(f"unknown theme {theme!r}; known: {sorted(THEMES)}")
    colours = THEMES[theme]
    PALETTE.clear()
    PALETTE.update(colours)
    # Fusion rather than the platform style: a native style paints its own
    # backgrounds for some primitives and the sheet then only half applies.
    app.setStyle("Fusion")
    # 9pt, which is the proportion the figure draws — Qt's default here is a
    # size or two larger and the tab strip stops fitting. Points, not pixels, so
    # a HiDPI desktop still scales it.
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    app.setPalette(palette(colours))
    app.setStyleSheet(sheet(colours))


def label_style() -> str:
    """The amber label of a construction parameter, in the current theme."""
    return f"color: {PALETTE['amber_ink']}; font-weight: 600;"


def _demo() -> None:
    """Both themes must define exactly the same roles, or a switch leaves a hole."""
    assert LIGHT.keys() == DARK.keys(), LIGHT.keys() ^ DARK.keys()
    for name, colours in THEMES.items():
        assert sheet(colours), name
        assert all(value.startswith("#") for value in colours.values()), name
        # Built, not merely formatted: a role naming a key that is not there
        # raises here rather than three themes later.
        assert palette(colours).color(QPalette.ColorRole.Window).isValid(), name
    print("style ok")


if __name__ == "__main__":
    _demo()
