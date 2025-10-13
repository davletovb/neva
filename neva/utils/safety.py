"""Safety and validation helpers for user provided prompts."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List

from neva.utils.exceptions import (
    PromptValidationError,
    RateLimiterConfigurationError,
)


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060]")
WHITESPACE_NORMALISER_RE = re.compile(r"\s+")
DEFAULT_PROMPT_INJECTION_PATTERNS: Iterable[str] = (
    r"ignore\s+all\s+previous\s+instructions",
    r"forget\s+(?:the\s+)?previous\s+(?:steps|instructions)",
    r"(?<![\w-])sys(?:tem)?\s*:\s*",
    r"developer\s+mode\s+is\s+now\s+enabled",
    r"(?<![\w-])reset\s+the\s+conversation",
)


def sanitize_input(text: str) -> str:
    """Normalise user provided text prior to validation.

    The sanitisation step removes terminal control characters, zero-width glyphs that can
    be used to obfuscate prompt injection attempts, and collapses consecutive whitespace.
    The resulting string is safe to log and suitable for subsequent validation.
    """

    cleaned = CONTROL_CHARS_RE.sub("", text)
    cleaned = ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = WHITESPACE_NORMALISER_RE.sub(" ", cleaned)
    return cleaned.strip()


@dataclass
class PromptValidator:
    """Validate prompts before sending them to external LLM APIs."""

    max_length: int = 4000
    forbidden_patterns: Iterable[str] = field(
        default_factory=lambda: [
            r"<script>",
            r"drop\s+table",
            r"\bshutdown\b",
            *DEFAULT_PROMPT_INJECTION_PATTERNS,
        ]
    )

    def __post_init__(self) -> None:
        self._compiled_patterns: List[re.Pattern[str]] = [
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
        with self._lock:
            current = time.monotonic()
            time_passed = current - self._last_check
            self._last_check = current
            self._allowance += time_passed * (self._rate / self._per)
            if self._allowance > self._rate:
                self._allowance = float(self._rate)
            if self._allowance < 1.0:
                sleep_time = (1.0 - self._allowance) * (self._per / self._rate)
                time.sleep(sleep_time)
                self._allowance = 0.0
                self._last_check = time.monotonic()
            else:
                self._allowance -= 1.0

