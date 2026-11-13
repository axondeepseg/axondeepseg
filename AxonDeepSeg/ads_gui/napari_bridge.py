"""Shares a single napari viewer across the app's 'Open in napari' entry points
(the main window's button and the preview dialog's button) so repeated clicks
reuse one window instead of stacking new ones.
"""
_viewer = None


def get_viewer():
    """Return the shared napari.Viewer, creating one if needed.

    If the previous viewer's window was closed, its Qt widgets are gone even
    though the Python object lingers — any attribute access on it then raises
    RuntimeError("wrapped C/C++ object ... has been deleted"), which we treat
    as "make a new one".
    """
    global _viewer
    import napari

    if _viewer is not None:
        try:
            _viewer.window._qt_window.isVisible()
        except RuntimeError:
            _viewer = None

    if _viewer is None:
        _viewer = napari.Viewer()
    else:
        # Bring the existing window to front rather than leaving the click looking like a no-op.
        window = _viewer.window._qt_window
        window.raise_()
        window.activateWindow()
    return _viewer
