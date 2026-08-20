# -*- mode: python ; coding: utf-8 -*-
"""One build recipe for the editor, shared by Linux and macOS.

Packages `segmentator-gui` only — the CLI (`segmentator`) is a `uv run segmentator`
tool, not something a double-clickable app has a use for.

onedir rather than onefile: onefile would unpack cv2 and scikit-image into a temp
directory on every launch, which costs seconds of cold start and buys nothing once an
installer is wrapping the output anyway.

`hiddenimports` is deliberately empty and `excludes` deliberately short.
pyinstaller-hooks-contrib already knows how to collect PyQt6 and scikit-image — whose
`lazy_loader` stubs are data files rather than imports — and the PyQt6 hook prunes Qt
down to the modules actually imported. Entries added here without a failing smoke test
to point at are cargo cult.
"""

import sys
from importlib.metadata import version

from PyInstaller.utils.hooks import copy_metadata

# The build env has the package installed, so the version has one source: pyproject.toml.
VERSION = version("segmentator")

a = Analysis(
    ["segmentator/gui/main.py"],
    pathex=[],
    binaries=[],
    # The icon, because the launcher window draws it, and the dist-info beside
    # it, because the launcher also *reads* the version — and
    # `importlib.metadata.version` raises `PackageNotFoundError` in a frozen
    # bundle that carries the code without the metadata.
    datas=[
        ("segmentator/gui/assets/icon.png", "segmentator/gui/assets"),
        *copy_metadata("segmentator"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="segmentator-gui",
    debug=False,
    bootloader_ignore_signals=False,
    # UPX mangles Qt's shared libraries often enough that the size win is not worth the
    # class of bug it introduces, and stripping breaks some binary wheels the same way.
    strip=False,
    upx=False,
    console=False,
    # PyInstaller only consumes an icon here on Windows; macOS takes it from BUNDLE
    # below, and the Linux AppImage takes it from the .desktop file instead.
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="segmentator-gui",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Segmentator.app",
        # Built by packaging/macos/build-dmg.sh, which is also what invokes this spec.
        icon="build/icon.icns",
        bundle_identifier="io.github.goncamateus.segmentator",
        version=VERSION,
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleShortVersionString": VERSION,
        },
    )
