"""Automatic retry/escalation policy and observable recovery counters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Type


@dataclass(frozen=True)
class RecoveryPolicy:
    """Opt-in retry/backoff and final escalation policy.

    max_retries=0 preserves the historical single-attempt behavior.
    escalation may inherit the existing environment/per-agent error policy,
    or force raise/return after retries are exhausted.
    """

    max_retries: int = 0
    backoff: float = 0.0
    backoff_multiplier: float = 2.0
    escalation: str = "inherit"
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if (
            isinstance(self.backoff, bool)
            or not isinstance(self.backoff, (int, float))
            or not math.isfinite(self.backoff)
            or self.backoff < 0
        ):
            raise ValueError("backoff must be a finite non-negative number")
        if (
            isinstance(self.backoff_multiplier, bool)
            or not isinstance(self.backoff_multiplier, (int, float))
            or not math.isfinite(self.backoff_multiplier)
            or self.backoff_multiplier < 1
        ):
            raise ValueError("backoff_multiplier must be finite and at least 1")
        if self.escalation not in {"inherit", "raise", "return"}:
            raise ValueError("escalation must be 'inherit', 'raise', or 'return'")
        if not self.retry_on or not all(
            isinstance(exc_type, type) and issubclass(exc_type, BaseException)
            for exc_type in self.retry_on
        ):
            raise ValueError("retry_on must contain exception types")

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        return attempt < self.max_attempts and isinstance(exc, self.retry_on)

    def delay_for(self, attempt: int) -> float:
        """Return seconds to wait after failed attempt before retrying."""

        if attempt <= 0:
            raise ValueError("attempt must be positive")
        return float(self.backoff) * float(self.backoff_multiplier) ** (attempt - 1)


@dataclass
class RecoveryState:
    """Observable counters for automatic recovery activity in one environment."""

    failures_seen: int = 0
    retries_attempted: int = 0
    recoveries_succeeded: int = 0
    retries_exhausted: int = 0
    escalations: int = 0
    failure_records_written: int = 0
    last_action: Optional[str] = None
    last_error_type: Optional[str] = None
    last_error_message: Optional[str] = None
    last_agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["RecoveryPolicy", "RecoveryState"]
