import pytest

import pytest

from neva.utils import safety
from neva.utils.exceptions import PromptValidationError


def test_prompt_validator_rejects_forbidden_pattern() -> None:
    validator = safety.PromptValidator()
    with pytest.raises(PromptValidationError):
        validator.validate("DROP table users;")


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
