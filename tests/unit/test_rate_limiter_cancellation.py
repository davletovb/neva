import threading

import pytest

from neva.utils.exceptions import RateLimiterCancelledError
from neva.utils.safety import RateLimiter


def test_cancellation_exception_remains_compatible():
    from concurrent.futures import CancelledError

    assert issubclass(RateLimiterCancelledError, CancelledError)
    assert issubclass(RateLimiterCancelledError, Exception)


def test_cancellation_checked_after_lock_entry_preserves_token():
    cancel = threading.Event()
    limiter = RateLimiter(rate=1, per=60)

    class CancellingLock:
        def __enter__(self):
            cancel.set()

        def __exit__(self, *args):
            return False

    limiter._lock = CancellingLock()
    with pytest.raises(RateLimiterCancelledError):
        limiter.acquire(cancel_event=cancel)
    assert limiter._allowance == 1


def test_real_token_wait_can_be_cancelled():
    waiting = threading.Event()
    outcome = []

    class NotifyingEvent(threading.Event):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    cancel = NotifyingEvent()
    limiter = RateLimiter(rate=1, per=60)
    limiter.acquire()

    def acquire():
        try:
            limiter.acquire(cancel_event=cancel)
        except RateLimiterCancelledError:
            outcome.append("cancelled")

    worker = threading.Thread(target=acquire, daemon=True)
    worker.start()
    try:
        assert waiting.wait(timeout=2)
        # Waiting must not monopolize the shared limiter lock.
        acquired = limiter._lock.acquire(timeout=1)
        if acquired:
            limiter._lock.release()
        assert acquired
    finally:
        cancel.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert outcome == ["cancelled"]


def test_cancelled_waiter_leaves_limiter_healthy_for_subsequent_callers(monkeypatch):
    from neva.utils import safety

    now = [0.0]
    waits = []

    class CancellingEvent(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            self.set()
            return True

    monkeypatch.setattr(safety.time, "monotonic", lambda: now[0])
    limiter = RateLimiter(rate=1, per=60)
    limiter.acquire()
    cancel = CancellingEvent()
    with pytest.raises(RateLimiterCancelledError):
        limiter.acquire(cancel_event=cancel)
    assert waits == [60.0]
    assert limiter._allowance == 0

    # A different caller can use the next token after refill.
    now[0] = 60.0
    other_caller = threading.Event()
    assert limiter.acquire(cancel_event=other_caller) is None
    assert limiter._allowance == 0


def test_unset_event_waits_for_refill(monkeypatch):
    from neva.utils import safety

    now = [0.0]
    waits = []
    monkeypatch.setattr(safety.time, "monotonic", lambda: now[0])

    class AdvancingEvent(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            now[0] += timeout
            return False

    limiter = RateLimiter(rate=1, per=10)
    cancel = AdvancingEvent()
    limiter.acquire(cancel_event=cancel)
    limiter.acquire(cancel_event=cancel)
    assert waits == [10.0]
    assert limiter._allowance == 0


def test_pre_cancelled_acquire_preserves_available_token():
    limiter = RateLimiter(rate=1, per=60)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RateLimiterCancelledError):
        limiter.acquire(cancel_event=cancel)
    # Cancellation must not consume the initial token.
    assert limiter._allowance == 1


def test_waiting_acquire_is_cancelled_without_consuming_token(monkeypatch):
    from neva.utils import safety

    now = [0.0]
    monkeypatch.setattr(safety.time, "monotonic", lambda: now[0])

    class CancellingEvent(threading.Event):
        def wait(self, timeout=None):
            self.set()
            return True

    cancel = CancellingEvent()
    limiter = RateLimiter(rate=1, per=60)
    limiter.acquire()
    # Bound the old implementation's sleep so RED cannot hang.
    monkeypatch.setattr(safety.time, "sleep", lambda seconds: now.__setitem__(0, 60.0))
    with pytest.raises(RateLimiterCancelledError, match="cancelled"):
        limiter.acquire(cancel_event=cancel)
    assert limiter._allowance == 0
