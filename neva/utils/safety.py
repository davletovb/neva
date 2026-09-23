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
    RateLimiterCancelledError,
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
    """FIFO token-bucket limiter with cooperative cancellation.

    The limiter is thread-safe and loop-agnostic. Waiters are admitted in FIFO
    order. Token sleeps happen outside the mutex, and normal threading.Lock
    acquisition is polled so a cancellation event can interrupt a waiter even
    while another thread briefly owns the lock.

    This class is process-local. GPTAgent uses ProviderResourceCoordinator by
    default for automatic provider/account sharing and optional SQLite-backed
    cross-process coordination.
    """

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
        self._next_ticket = 0
        self._waiters = []  # type: list[tuple[int, threading.Event]]

    def _lock_enter(self, cancel_event: Optional[threading.Event]) -> bool:
        acquire = getattr(self._lock, "acquire", None)
        if not callable(acquire):
            self._lock.__enter__()
            return False
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RateLimiterCancelledError("Rate limiter acquisition cancelled")
            try:
                acquired = acquire(timeout=0.05)
            except TypeError:
                acquire()
                acquired = True
            if acquired:
                return True

    def _lock_exit(self, acquired_normally: bool) -> None:
        if acquired_normally:
            self._lock.release()
        else:
            self._lock.__exit__(None, None, None)

    def _remove_waiter(self, ticket: int) -> None:
        for index, (candidate, _) in enumerate(self._waiters):
            if candidate == ticket:
                self._waiters.pop(index)
                break
        if self._waiters:
            self._waiters[0][1].set()

    def acquire(self, *, cancel_event: Optional[threading.Event] = None) -> None:
        """Block until one token is available, preserving FIFO waiter order."""

        if cancel_event is not None and cancel_event.is_set():
            raise RateLimiterCancelledError("Rate limiter acquisition cancelled")

        acquired_normally = self._lock_enter(cancel_event)
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RateLimiterCancelledError("Rate limiter acquisition cancelled")
            ticket = self._next_ticket
            self._next_ticket += 1
            wake = threading.Event()
            self._waiters.append((ticket, wake))
            if self._waiters[0][0] == ticket:
                wake.set()
        finally:
            self._lock_exit(acquired_normally)

        admitted = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise RateLimiterCancelledError("Rate limiter acquisition cancelled")

                sleep_time = 0.05
                is_head = False
                acquired_normally = self._lock_enter(cancel_event)
                try:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RateLimiterCancelledError(
                            "Rate limiter acquisition cancelled"
                        )
                    is_head = bool(self._waiters and self._waiters[0][0] == ticket)
                    if is_head:
                        current = time.monotonic()
                        time_passed = current - self._last_check
                        self._last_check = current
                        self._allowance += time_passed * (self._rate / self._per)
                        if self._allowance > self._rate:
                            self._allowance = float(self._rate)
                        if self._allowance >= 1.0:
                            self._allowance -= 1.0
                            self._waiters.pop(0)
                            admitted = True
                            if self._waiters:
                                self._waiters[0][1].set()
                            return
                        sleep_time = (1.0 - self._allowance) * (
                            self._per / self._rate
                        )
                        wake.clear()
                finally:
                    self._lock_exit(acquired_normally)

                if is_head:
                    if cancel_event is None:
                        time.sleep(sleep_time)
                    elif cancel_event.wait(sleep_time):
                        continue
                elif cancel_event is None:
                    wake.wait(0.05)
                elif cancel_event.wait(0.05):
                    continue
        finally:
            if not admitted:
                acquired_normally = self._lock_enter(None)
                try:
                    self._remove_waiter(ticket)
                finally:
                    self._lock_exit(acquired_normally)


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

    def record_rejected(self) -> None:
        """Release an in-flight probe without counting provider downtime.

        A half-open probe that fails for a non-retryable reason (auth, config)
        must not stay marked in-flight, or later calls can never probe again.
        The circuit stays open and a new probe waits for the cooldown.
        """

        with self._lock:
            if not self._probe_in_flight:
                return
            self._probe_in_flight = False
            self._opened_at = time.monotonic()
