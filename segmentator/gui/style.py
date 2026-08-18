"""The look, in one place: the palette `docs/assets/gui-window.svg` draws.

The figure is the spec — a white window, `#f5f7fa` fields, `#cbd2d9` borders, a
blue selection, and amber for the two things that matter while tuning: a `name:`
tap, and a stage that remembers frames. The editor forces it rather than
inheriting the desktop theme, so it looks the same on every machine and the same
as the documentation.

Two mechanisms, and both are needed:

* the **stylesheet** styles what the window builds itself;
* the **palette** covers what Qt builds for us — `QInputDialog`, `QFileDialog`,
  `QMessageBox`, tooltips — none of which the sheet's selectors reach, and all
  of which come back dark on a dark desktop without it.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette

PALETTE = {
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

MONO = "ui-monospace, SFMono-Regular, Menlo, DejaVu Sans Mono, monospace"

SHEET = """
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

QSplitter::handle {{ background: {rule}; }}
QScrollArea {{ border: 0; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {panel}; border: 0; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {border}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {ink}; color: {panel}; border: 0; padding: 4px; }}
""".format(mono=MONO, **PALETTE)


def palette() -> QPalette:
    """The light palette, for everything the stylesheet's selectors cannot reach."""
    colours = {
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
    for role, key in colours.items():
        built.setColor(role, QColor(PALETTE[key]))
    return built


def apply(app) -> None:
    """Style one QApplication. Call it before the window is built."""
    # Fusion rather than the platform style: a native style paints its own
    # backgrounds for some primitives and the sheet then only half applies.
    app.setStyle("Fusion")
    # 9pt, which is the proportion the figure draws — Qt's default here is a
    # size or two larger and the tab strip stops fitting. Points, not pixels, so
    # a HiDPI desktop still scales it.
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    app.setPalette(palette())
    app.setStyleSheet(SHEET)
