"""Utilities for wrapping tqdm progress bars across GUI integrations."""

from typing import Callable, Optional


class TqdmProgressWrapper:
    """Wrap a tqdm-compatible object for progress callbacks and cancellation.

    This wrapper is intentionally generic so both the napari plugin and the
    standalone GUI backend can reuse the same logic. It supports both:
    - context-manager usage via ``with tqdm(...) as pbar:``
    - direct ``update()`` calls

    The wrapped object delegates attribute access to the inner tqdm object.
    """

    def __init__(
        self,
        inner,
        progress_callback: Callable[[int, int, str], None],
        is_cancel_requested: Callable[[], bool],
        description: str = "Applying model",
    ):
        self._inner = inner
        self._progress_callback = progress_callback
        self._is_cancel_requested = is_cancel_requested
        self._description = description

    def _emit_progress(self, current: int, total: Optional[int], description: Optional[str] = None):
        if total is None:
            return
        self._progress_callback(current, total, description or self._description)

    def _check_cancelled(self):
        if self._is_cancel_requested():
            raise InterruptedError("Segmentation cancelled by user")

    def __iter__(self):
        total = getattr(self._inner, "total", None)
        if total is not None:
            self._emit_progress(0, total)
        for i, item in enumerate(self._inner):
            self._check_cancelled()
            yield item
            if total is not None:
                self._emit_progress(i + 1, total)

    def update(self, n=1):
        self._check_cancelled()
        result = self._inner.update(n)
        total = getattr(self._inner, "total", None)
        current = getattr(self._inner, "n", None)
        if total is not None and current is not None:
            self._emit_progress(current, total)
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        self._inner.__enter__()
        total = getattr(self._inner, "total", None)
        if total is not None:
            self._emit_progress(0, total)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return self._inner.__exit__(exc_type, exc, exc_tb)
