"""Safety and validation helpers for user provided prompts."""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from neva.utils.exceptions import (
    CircuitBreakerConfigurationError,
    CircuitOpenError,
    PromptValidationError,
    RateLimiterConfigurationError,
)

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_input(text: str) -> str:
    """Remove control characters that can break terminal logs or JSON."""
    return CONTROL_CHARS_RE.sub("", text)


@dataclass
class PromptValidator:
    """Validate prompts before sending them to external LLM APIs."""

    max_length: int = 4000
    forbidden_patterns: Iterable[str] = field(
        default_factory=lambda: [r"<script\b", r"drop\s+table"]
    )

    def __post_init__(self) -> None:
        self._compiled_patterns: list[re.Pattern[str]] = [
            re.compile(pattern, flags=re.IGNORECASE) for pattern in self.forbidden_patterns
        ]

    def validate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be a string")
        prompt = sanitize_input(prompt)
        if len(prompt) > self.max_length:
            raise PromptValidationError("prompt exceeds maximum allowed length")
        for pattern in self._compiled_patterns:
            if pattern.search(prompt):
                raise PromptValidationError("prompt contains forbidden content")
        return prompt


class RateLimiter:
    """Simple token bucket rate limiter to guard API usage."""

    def __init__(self, rate: int, per: float = 60.0) -> None:
        if rate <= 0:
            raise RateLimiterConfigurationError("rate must be positive")
        if per <= 0:
            raise RateLimiterConfigurationError("per must be positive")
        self._rate = rate
        self._per = per
        self._allowance = float(rate)
        self._last_check = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available.

        Sleep happens outside the lock so callers sharing this limiter are not
        serialized for the full wait. Limits apply to this instance only; pass
        the same limiter into multiple agents to share a provider budget.
        """

        while True:
            sleep_time = 0.0
            with self._lock:
                current = time.monotonic()
                time_passed = current - self._last_check
                self._last_check = current
                self._allowance += time_passed * (self._rate / self._per)
                if self._allowance > self._rate:
                    self._allowance = float(self._rate)
                if self._allowance >= 1.0:
                    self._allowance -= 1.0
                    return
                sleep_time = (1.0 - self._allowance) * (self._per / self._rate)
            time.sleep(sleep_time)


class CircuitBreaker:
    """Fail fast after consecutive retryable failures; probe after a cooldown.

    Pass the same instance to multiple agents to share a provider circuit.
    Limits apply to this instance only.
    """

    def __init__(self, failure_threshold: int = 5, cooldown: float = 30.0) -> None:
        if failure_threshold <= 0:
            raise CircuitBreakerConfigurationError("failure_threshold must be positive")
        if cooldown < 0:
            raise CircuitBreakerConfigurationError("cooldown must be non-negative")
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def allow(self) -> None:
        """Raise ``CircuitOpenError`` when the circuit is open and cooling down."""

        with self._lock:
            if self._opened_at is None:
                return
            remaining = self._cooldown - (time.monotonic() - self._opened_at)
            if remaining > 0:
                raise CircuitOpenError(f"circuit open; retry after {remaining:.1f}s")
            if self._probe_in_flight:
                raise CircuitOpenError("circuit open; a recovery probe is already in flight")
            self._probe_in_flight = True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._probe_in_flight = False
            if self._opened_at is not None or self._failures >= self._failure_threshold:
                self._opened_at = time.monotonic()
