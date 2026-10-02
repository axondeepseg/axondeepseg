import pytest

from AxonDeepSeg.tqdm_utils import TqdmProgressWrapper


class DummySignal:
    def __init__(self):
        self.calls = []

    def emit(self, current, total, desc):
        self.calls.append((current, total, desc))


class DummyTqdm:
    def __init__(self, total=3):
        self.total = total
        self.n = 0
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.exited = True
        return False

    def update(self, n=1):
        self.n += n
        return None


class TestTqdmWrapper:
    @pytest.mark.unit
    def test_tqdm_progress_wrapper_supports_context_manager_and_updates(self):
        signal = DummySignal()
        inner = DummyTqdm(total=3)
        wrapper = TqdmProgressWrapper(inner, signal.emit, lambda: False, "Testing")

        with wrapper as wrapped:
            assert wrapped is wrapper
            assert inner.entered is True
            assert signal.calls == [(0, 3, "Testing")]

            wrapped.update(2)

        assert inner.exited is True
        assert signal.calls[-1] == (2, 3, "Testing")
