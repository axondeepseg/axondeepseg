"""Shared setup for the standalone GUI tests.

Unlike the napari plugin tests (test/ads_napari/), these are plain PyQt5 widgets
with no vispy/OpenGL involved, so they can also run on a headless Linux CI runner
once Qt's 'offscreen' platform plugin is selected.

That selection is deliberately narrow. pytest-qt builds a single QApplication for
the whole session, so whichever test package runs first fixes the platform plugin
for every Qt test after it — forcing 'offscreen' unconditionally would drag the
napari tests onto a platform that gives them no GL context. Only fill it in when
there is genuinely no display to talk to, and let setdefault yield to anyone who
set the variable themselves.
"""
import os
import sys

_NO_DISPLAY = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")

if sys.platform.startswith("linux") and _NO_DISPLAY:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
