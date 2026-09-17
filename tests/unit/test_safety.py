import threading

import pytest

from neva.utils import safety
from neva.utils.exceptions import (
    CircuitBreakerConfigurationError,
    CircuitOpenError,
    PromptValidationError,
)


def test_prompt_validator_rejects_forbidden_pattern() -> None:
    validator = safety.PromptValidator()
    with pytest.raises(PromptValidationError):
        validator.validate("DROP table users;")


def test_prompt_validator_allows_ordinary_shutdown_language() -> None:
    validator = safety.PromptValidator()
    assert "shutdown" in validator.validate("Please shutdown the rover after the survey.")


def test_prompt_validator_sanitises_control_characters() -> None:
    validator = safety.PromptValidator()
    result = validator.validate("hello\x07world")
    assert result == "helloworld"


def test_rate_limiter_honours_rate(monkeypatch) -> None:
    limiter = safety.RateLimiter(rate=2, per=1)
    timestamps = [0.0]
    sleeps = []

    def fake_monotonic() -> float:
        return timestamps[0]

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)
        timestamps[0] += duration

    monkeypatch.setattr(safety.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(safety.time, "sleep", fake_sleep)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleeps, "rate limiter should require sleeping once allowance exhausted"
    assert timestamps[0] > 0


def test_rate_limiter_sleeps_outside_lock(monkeypatch) -> None:
    timestamps = [0.0]
    sleeping = threading.Event()
    lock_released = threading.Event()

    def fake_monotonic() -> float:
        return timestamps[0]

    def fake_sleep(duration: float) -> None:
        sleeping.set()
        assert lock_released.wait(timeout=1.0)
        timestamps[0] += duration

    monkeypatch.setattr(safety.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(safety.time, "sleep", fake_sleep)

    limiter = safety.RateLimiter(rate=1, per=1)
    limiter.acquire()

    waiter = threading.Thread(target=limiter.acquire)
    waiter.start()
    assert sleeping.wait(timeout=1.0)
    acquired = limiter._lock.acquire(timeout=0.2)
    try:
        assert acquired, "sleep should happen outside the limiter lock"
    finally:
        lock_released.set()
        if acquired:
            limiter._lock.release()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()


def test_circuit_breaker_rejects_invalid_config() -> None:
    with pytest.raises(CircuitBreakerConfigurationError):
        safety.CircuitBreaker(failure_threshold=0)
    with pytest.raises(CircuitBreakerConfigurationError):
        safety.CircuitBreaker(cooldown=-1)


def test_circuit_breaker_opens_and_probes(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(safety.time, "monotonic", lambda: now[0])
    breaker = safety.CircuitBreaker(failure_threshold=2, cooldown=10.0)

    breaker.allow()
    breaker.record_failure()
    breaker.allow()
    breaker.record_failure()
    with pytest.raises(CircuitOpenError, match="circuit open"):
        breaker.allow()

    now[0] += 10.0
    breaker.allow()
    with pytest.raises(CircuitOpenError, match="probe"):
        breaker.allow()
    breaker.record_rejected()
    with pytest.raises(CircuitOpenError, match="retry after"):
        breaker.allow()
    now[0] += 10.0
    breaker.allow()
    breaker.record_success()
    breaker.allow()
    breaker.allow()
