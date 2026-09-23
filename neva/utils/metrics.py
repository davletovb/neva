"""Light-weight metrics helpers for observability and evaluation."""

from __future__ import annotations

import math
import statistics
import threading
import tracemalloc
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

from neva.utils.exceptions import SpendBudgetConfigurationError, SpendBudgetExceededError


def estimate_token_count(text: str) -> int:
    """Coarse token estimation that works without backend specific tooling."""

    if not text:
        return 0
    return max(1, len(text.split()))


_estimate_token_count = estimate_token_count


@dataclass
class TokenUsageTracker:
    """Track prompt/response token usage for cost attribution.

    When a provider reports usage for a call, pass it via ``usage``; otherwise
    counts are estimated from whitespace and flagged in ``estimated_calls``.
    """

    records: List[Tuple[int, int]] = field(default_factory=list)
    estimated_calls: int = 0

    def record(
        self,
        prompt: str,
        response: str,
        *,
        usage: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        if usage is not None:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            response_tokens = int(
                usage.get("completion_tokens") or usage.get("response_tokens") or 0
            )
            if not prompt_tokens and not response_tokens:
                prompt_tokens = _estimate_token_count(prompt)
                response_tokens = _estimate_token_count(response)
                self.estimated_calls += 1
        else:
            prompt_tokens = _estimate_token_count(prompt)
            response_tokens = _estimate_token_count(response)
            self.estimated_calls += 1
        self.records.append((prompt_tokens, response_tokens))
        return prompt_tokens, response_tokens

    def total_tokens(self) -> int:
        return sum(prompt + response for prompt, response in self.records)


@dataclass
class CostTracker:
    """Estimate the monetary cost of LLM usage based on token counts.

    Default table is USD per 1k tokens for Neva's built-in models (list
    prices as of 2026-09). Override ``pricing_per_1k_tokens`` for billing.
    Split input/output dicts are used for current models; legacy flat rates
    remain for ``gpt-3.5-turbo`` and ``gpt-4``.
    """

    pricing_per_1k_tokens: Dict[str, Any] = field(
        default_factory=lambda: {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "grok-4.5": {"input": 0.002, "output": 0.006},
            "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        }
    )
    usage: Dict[str, int] = field(default_factory=dict)
    split_usage: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def add_usage(
        self,
        model: str,
        tokens: int,
        *,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
    ) -> None:
        self.usage[model] = self.usage.get(model, 0) + tokens
        if prompt_tokens is not None or response_tokens is not None:
            previous = self.split_usage.get(model, (0, 0))
            self.split_usage[model] = (
                previous[0] + (prompt_tokens or 0),
                previous[1] + (response_tokens or 0),
            )

    def unpriced_models(self) -> List[str]:
        """Models that were used but have no pricing entry."""

        return sorted(model for model in self.usage if model not in self.pricing_per_1k_tokens)

    def cost_for(
        self,
        model: str,
        *,
        prompt_tokens: int,
        response_tokens: int,
    ) -> Optional[float]:
        """Return the isolated estimated cost for one model call."""

        price = self.pricing_per_1k_tokens.get(model)
        if price is None:
            return None
        if isinstance(price, dict):
            return (prompt_tokens / 1000.0) * price.get("input", 0.0) + (
                response_tokens / 1000.0
            ) * price.get("output", 0.0)
        return ((prompt_tokens + response_tokens) / 1000.0) * price

    def total_cost(self) -> Optional[float]:
        """Return estimated cost, or ``None`` when pricing is incomplete.

        A model is priced either by a flat per-1k rate or by input/output
        rates; mixing priced and unpriced models yields ``None`` so callers
        never mistake unknown pricing for a free run.
        """

        cost = 0.0
        for model, tokens in self.usage.items():
            prompt_tokens, response_tokens = self.split_usage.get(model, (0, 0))
            if tokens and prompt_tokens + response_tokens not in {0, tokens}:
                return None
            if prompt_tokens + response_tokens == 0:
                prompt_tokens, response_tokens = tokens, 0
            call_cost = self.cost_for(
                model,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
            if call_cost is None:
                return None
            cost += call_cost
        return cost


@dataclass
class ResponseTimeTracker:
    """Context manager that measures response latency."""

    durations: List[float] = field(default_factory=list)

    @contextmanager
    def track(self) -> Generator[None, None, None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.durations.append(perf_counter() - start)

    def latest(self) -> float:
        return self.durations[-1] if self.durations else 0.0

    def average(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0


def profile_memory_usage(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[int, int]:
    """Run ``func`` and return (current, peak) memory usage in KiB."""

    tracemalloc.start()
    try:
        func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        return current // 1024, peak // 1024
    finally:
        tracemalloc.stop()


def batch_prompt_summary(prompts: Iterable[str]) -> Dict[str, int]:
    """Return simple descriptive statistics for a batch of prompts."""

    lengths = [len(prompt) for prompt in prompts]
    token_estimates = [_estimate_token_count(prompt) for prompt in prompts]
    if not lengths:
        return {"count": 0, "avg_length": 0, "avg_tokens": 0}
    return {
        "count": len(lengths),
        "avg_length": int(statistics.mean(lengths)),
        "avg_tokens": int(statistics.mean(token_estimates)),
    }


@dataclass(frozen=True)
class SpendReservation:
    """Opaque reservation against a :class:`SpendBudget`."""

    token: str
    amount: float


class SpendBudget:
    """Thread-safe hard ceiling on estimated monetary spend.

    Pass the same instance to multiple agents to share one budget (per
    instance only; no account- or process-wide coordination). Amounts come
    from :class:`CostTracker` estimates, so this bounds *estimated* spend:
    unpriced models yield ``None`` costs that cannot be accounted for, and
    estimates may differ from live billing. This is a guardrail against
    runaway runs, not a billing system.
    """

    def __init__(self, max_cost: float) -> None:
        if (
            isinstance(max_cost, bool)
            or not isinstance(max_cost, (int, float))
            or not math.isfinite(max_cost)
            or max_cost <= 0
        ):
            raise SpendBudgetConfigurationError("max_cost must be a finite positive number")
        self._max_cost = float(max_cost)
        self._spent = 0.0
        self._reservations: Dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def reserved(self) -> float:
        with self._lock:
            return sum(self._reservations.values())

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(
                0.0,
                self._max_cost - self._spent - sum(self._reservations.values()),
            )

    def check(self, required_cost: float = 0.0) -> None:
        """Raise when completed + reserved + required spend cannot fit."""

        self._validate_cost(required_cost)
        with self._lock:
            committed = self._spent + sum(self._reservations.values())
            exhausted = (
                committed >= self._max_cost - 1e-9
                if required_cost == 0
                else committed + required_cost > self._max_cost + 1e-9
            )
            if exhausted:
                raise SpendBudgetExceededError(
                    "spend budget exhausted or reserved "
                    f"({committed:.6f} + {required_cost:.6f} > {self._max_cost:.6f})"
                )

    @staticmethod
    def _validate_cost(cost: float) -> None:
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise SpendBudgetConfigurationError("cost must be a finite, non-negative number")

    def reserve(self, cost: float) -> SpendReservation:
        """Atomically reserve estimated worst-case spend before a provider call."""

        self._validate_cost(cost)
        token = uuid.uuid4().hex
        with self._lock:
            committed = self._spent + sum(self._reservations.values())
            if committed + cost > self._max_cost + 1e-9:
                raise SpendBudgetExceededError(
                    f"reservation of {cost:.6f} exceeds the remaining budget"
                )
            self._reservations[token] = float(cost)
        return SpendReservation(token=token, amount=float(cost))

    def release(self, reservation: SpendReservation) -> None:
        """Release an unused reservation after a call fails before billing."""

        if not isinstance(reservation, SpendReservation):
            raise TypeError("reservation must be a SpendReservation")
        with self._lock:
            self._reservations.pop(reservation.token, None)

    def settle(self, reservation: SpendReservation, actual_cost: float) -> None:
        """Replace one reservation with the call's measured/estimated cost."""

        if not isinstance(reservation, SpendReservation):
            raise TypeError("reservation must be a SpendReservation")
        self._validate_cost(actual_cost)
        with self._lock:
            reserved = self._reservations.pop(reservation.token, None)
            if reserved is None:
                raise SpendBudgetConfigurationError(
                    "spend reservation is unknown or already settled"
                )
            committed = self._spent + sum(self._reservations.values())
            if committed + actual_cost > self._max_cost + 1e-9:
                self._spent = self._max_cost
                raise SpendBudgetExceededError(
                    "provider call exceeded the remaining budget after settlement"
                )
            self._spent += float(actual_cost)

    def reconcile(self, actual_spent: float) -> None:
        """Replace completed estimates with an authoritative billing total.

        In-flight reservations remain reserved. This is intended for provider-
        specific billing integrations that can supply an account-level total.
        """

        self._validate_cost(actual_spent)
        with self._lock:
            self._spent = float(actual_spent)
            if self._spent + sum(self._reservations.values()) > self._max_cost + 1e-9:
                raise SpendBudgetExceededError(
                    "authoritative spend plus reservations exceeds the budget"
                )

    def consume(self, cost: float) -> None:
        """Record ``cost`` of spend, refusing amounts that exceed the budget.

        ``cost`` must be a finite, non-negative number (bools excluded);
        anything else raises :class:`SpendBudgetConfigurationError` rather
        than corrupting the budget. ``check()`` before a call is
        best-effort: concurrent consumers can each pass the pre-check, so
        ``consume`` itself caps the total at ``max_cost`` (within a 1e-9
        floating-point tolerance). When ``cost`` would overflow the budget,
        spend is clamped to the ceiling and :class:`SpendBudgetExceededError`
        is raised: the call that produced this cost already happened, so it
        counts as fully spent and every later ``check()`` refuses.
        """

        self._validate_cost(cost)
        with self._lock:
            if self._spent + sum(self._reservations.values()) + cost > self._max_cost + 1e-9:
                self._spent = self._max_cost
                raise SpendBudgetExceededError(
                    f"spend of {cost:.6f} exceeds the remaining budget; " "budget marked exhausted"
                )
            self._spent += cost
