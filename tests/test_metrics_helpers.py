from metrics import CostTracker, ResponseTimeTracker, TokenUsageTracker, batch_prompt_summary


def test_token_usage_tracker_accumulates_counts() -> None:
    tracker = TokenUsageTracker()
    tracker.record("hello world", "response here")
    tracker.record("more words", "and more output")
    assert tracker.total_tokens() > 0


def test_cost_tracker_estimates_cost() -> None:
    tracker = CostTracker()
    tracker.add_usage("gpt-3.5-turbo", 1000)
    assert tracker.total_cost() == tracker.pricing_per_1k_tokens["gpt-3.5-turbo"]


def test_response_time_tracker_records_duration() -> None:
    tracker = ResponseTimeTracker()
    with tracker.track():
        pass
    assert tracker.latest() >= 0.0


def test_batch_prompt_summary_handles_multiple_prompts() -> None:
    summary = batch_prompt_summary(["hi there", "general kenobi"])
    assert summary["count"] == 2
