# GUI extra depends on full `opencv-python`, not `-headless`

The `gui` extra pulls in plain `opencv-python` rather than `opencv-python-headless`
— the opposite of goncanalyser's own choice, and for the opposite reason:
`DisplaySink` needs the full wheel to open an OpenCV window, even during a batch
(non-GUI) run. That collides with PyQt6's bundled Qt plugins, so
`segmentator/gui/main.py` pops `QT_QPA_PLATFORM_PLUGIN_PATH` before PyQt6 loads to
avoid the clash.

Both halves of this are easy to "fix" in ways that silently break the other:
swapping to `-headless` kills `DisplaySink`'s window; dropping the env-var
workaround (e.g. during a dev-setup cleanup) breaks Qt plugin loading the moment
both packages are present. The workaround has to ship inside the frozen PyInstaller
bundle too, not just a dev-only setup step, since the same collision exists there.
