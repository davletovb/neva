"""Light-weight metrics helpers for observability and evaluation."""

from __future__ import annotations

import statistics
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Generator, Iterable, List, Tuple


def _estimate_token_count(text: str) -> int:
    """Coarse token estimation that works without backend specific tooling."""

    if not text:
        return 0
    return max(1, len(text.split()))


@dataclass
class TokenUsageTracker:
    """Track prompt/response token usage for cost attribution."""

    records: List[Tuple[int, int]] = field(default_factory=list)

    def record(self, prompt: str, response: str) -> Tuple[int, int]:
        prompt_tokens = _estimate_token_count(prompt)
        response_tokens = _estimate_token_count(response)
        self.records.append((prompt_tokens, response_tokens))
        return prompt_tokens, response_tokens

    def total_tokens(self) -> int:
        return sum(prompt + response for prompt, response in self.records)


@dataclass
class CostTracker:
    """Estimate the monetary cost of LLM usage based on token counts."""

    pricing_per_1k_tokens: Dict[str, float] = field(
        default_factory=lambda: {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
        }
    )
    usage: Dict[str, int] = field(default_factory=dict)

    def add_usage(self, model: str, tokens: int) -> None:
        self.usage[model] = self.usage.get(model, 0) + tokens

    def total_cost(self) -> float:
        cost = 0.0
        for model, tokens in self.usage.items():
            price = self.pricing_per_1k_tokens.get(model)
            if price is None:
                continue
            cost += (tokens / 1000.0) * price
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
