import pytest

from neva.utils.metrics import (
    CostTracker,
    ResponseTimeTracker,
    TokenUsageTracker,
    batch_prompt_summary,
)


def test_token_usage_tracker_accumulates_counts() -> None:
    tracker = TokenUsageTracker()
    tracker.record("hello world", "response here")
    tracker.record("more words", "and more output")
    assert tracker.total_tokens() > 0


def test_total_cost_returns_none_for_unpriced_models() -> None:
    tracker = CostTracker()
    tracker.add_usage("totally-new-model", 1000)
    assert tracker.total_cost() is None
    assert tracker.unpriced_models() == ["totally-new-model"]


def test_total_cost_supports_input_output_pricing() -> None:
    tracker = CostTracker(pricing_per_1k_tokens={"dual": {"input": 0.001, "output": 0.003}})
    tracker.add_usage("dual", 2000, prompt_tokens=1000, response_tokens=1000)
    assert tracker.total_cost() == pytest.approx(0.001 + 0.003)
    tracker.add_usage("dual", 1000)  # no input/output split recorded
    assert tracker.total_cost() is None  # split-priced totals need a split
    assert tracker.unpriced_models() == []


def test_token_usage_tracker_prefers_provider_usage() -> None:
    tracker = TokenUsageTracker()
    prompt_tokens, response_tokens = tracker.record(
        "one two three", "four five", usage={"prompt_tokens": 11, "completion_tokens": 7}
    )
    assert (prompt_tokens, response_tokens) == (11, 7)
    assert tracker.estimated_calls == 0
    estimate_prompt, estimate_response = tracker.record("one two", "three")
    assert (estimate_prompt, estimate_response) == (2, 1)
    assert tracker.estimated_calls == 1


def test_cost_tracker_estimates_cost() -> None:
    tracker = CostTracker()
    tracker.add_usage("gpt-3.5-turbo", 1000)
    assert tracker.total_cost() == tracker.pricing_per_1k_tokens["gpt-3.5-turbo"]


def test_default_prices_cover_builtin_models() -> None:
    tracker = CostTracker()
    tracker.add_usage("gpt-4o-mini", 2000, prompt_tokens=1000, response_tokens=1000)
    assert tracker.total_cost() == pytest.approx(0.00015 + 0.0006)
    grok = CostTracker()
    grok.add_usage("grok-4.5", 2000, prompt_tokens=1000, response_tokens=1000)
    assert grok.total_cost() == pytest.approx(0.002 + 0.006)
    assert "gemini-1.5-flash" in tracker.pricing_per_1k_tokens
    assert "claude-3-5-sonnet-latest" in tracker.pricing_per_1k_tokens


def test_response_time_tracker_records_duration() -> None:
    tracker = ResponseTimeTracker()
    with tracker.track():
        pass
    assert tracker.latest() >= 0.0


def test_batch_prompt_summary_handles_multiple_prompts() -> None:
    summary = batch_prompt_summary(["hi there", "general kenobi"])
    assert summary["count"] == 2
