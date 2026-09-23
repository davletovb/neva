import threading

import pytest
import requests

from neva.utils.exceptions import (
    ConfigurationError,
    SpendBudgetConfigurationError,
    SpendBudgetExceededError,
)
from neva.utils.metrics import CostTracker, SpendBudget


def test_consume_then_check_enforces_limit():
    budget = SpendBudget(max_cost=1.0)
    budget.check()
    budget.consume(0.6)
    budget.check()
    budget.consume(0.4)
    with pytest.raises(SpendBudgetExceededError):
        budget.check()


def test_consume_above_budget_raises_immediately():
    budget = SpendBudget(max_cost=0.5)
    with pytest.raises(SpendBudgetExceededError):
        budget.consume(0.6)
    assert budget.spent == pytest.approx(0.5)


@pytest.mark.parametrize("limit", [0, -1.0, "1.0", True, float("nan"), float("inf")])
def test_invalid_budget_rejected(limit):
    with pytest.raises(SpendBudgetConfigurationError):
        SpendBudget(max_cost=limit)


@pytest.mark.parametrize(
    "cost", [float("nan"), float("inf"), -float("inf"), -0.5, True, "0.1", None]
)
def test_invalid_cost_rejected_without_corrupting_budget(cost):
    budget = SpendBudget(max_cost=1.0)
    with pytest.raises(SpendBudgetConfigurationError):
        budget.consume(cost)
    assert budget.spent == 0.0
    budget.check()  # still enforcing normally


def test_remaining_and_spent_track_exactly():
    budget = SpendBudget(max_cost=2.0)
    assert budget.spent == 0.0
    assert budget.remaining == pytest.approx(2.0)
    budget.consume(0.25)
    assert budget.spent == pytest.approx(0.25)
    assert budget.remaining == pytest.approx(1.75)


def test_reservations_reduce_remaining_and_settle_actual_cost():
    budget = SpendBudget(max_cost=1.0)
    reservation = budget.reserve(0.6)

    assert budget.reserved == pytest.approx(0.6)
    assert budget.remaining == pytest.approx(0.4)
    with pytest.raises(SpendBudgetExceededError):
        budget.reserve(0.5)

    budget.settle(reservation, 0.25)
    assert budget.reserved == 0.0
    assert budget.spent == pytest.approx(0.25)
    assert budget.remaining == pytest.approx(0.75)


def test_release_and_reconcile_preserve_inflight_reservations():
    budget = SpendBudget(max_cost=2.0)
    reservation = budget.reserve(0.5)
    budget.reconcile(0.75)

    assert budget.spent == pytest.approx(0.75)
    assert budget.reserved == pytest.approx(0.5)
    assert budget.remaining == pytest.approx(0.75)

    budget.release(reservation)
    assert budget.remaining == pytest.approx(1.25)


def test_concurrent_consumers_never_exceed_budget():
    budget = SpendBudget(max_cost=1.0)
    increments = 0.01
    granted = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        while True:
            try:
                budget.consume(increments)
            except SpendBudgetExceededError:
                return
            with lock:
                granted.append(increments)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    total = sum(granted)
    assert total == pytest.approx(1.0)
    assert budget.remaining == pytest.approx(0.0, abs=1e-9)


def test_tracker_cost_integration_with_budget():
    tracker = CostTracker()
    tracker.add_usage("gpt-4o-mini", 2000, prompt_tokens=1000, response_tokens=1000)
    cost = tracker.total_cost()
    assert cost is not None
    budget = SpendBudget(max_cost=cost / 2)
    with pytest.raises(SpendBudgetExceededError):
        budget.consume(cost)


# ---------------------------------------------------------------------------
# GPTAgent integration
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def json(self):
        return self._payload


def _patch_provider(monkeypatch, *, usage=None, calls=None):
    def fake_post(url, headers=None, json=None, timeout=None):
        if calls is not None:
            calls.append(url)
        payload = {"choices": [{"message": {"content": "priced reply"}}]}
        if usage is not None:
            payload["usage"] = usage
        return _FakeResponse(payload)

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)


def test_gpt_agent_consumes_estimated_cost(monkeypatch):
    from neva.agents.gpt import GPTAgent

    _patch_provider(monkeypatch, usage={"prompt_tokens": 1000, "completion_tokens": 1000})
    budget = SpendBudget(max_cost=1.0)
    agent = GPTAgent(api_key="x", provider="openai", max_retries=0, spend_budget=budget)
    assert "priced reply" in agent.respond("ping")
    # gpt-4o-mini: 0.00015 in + 0.0006 out per 1k tokens.
    assert budget.spent == pytest.approx(0.00075)


def test_gpt_agent_refuses_call_when_budget_exhausted(monkeypatch):
    from neva.agents.gpt import GPTAgent

    calls = []
    _patch_provider(monkeypatch, usage={"prompt_tokens": 10, "completion_tokens": 10}, calls=calls)
    budget = SpendBudget(max_cost=0.001)
    budget.consume(0.001)
    agent = GPTAgent(api_key="x", provider="openai", max_retries=0, spend_budget=budget)
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("ping")
    assert calls == []  # refused before any provider request


def test_gpt_agent_budget_overflow_marks_exhausted(monkeypatch):
    from neva.agents.gpt import GPTAgent

    _patch_provider(monkeypatch, usage={"prompt_tokens": 1000, "completion_tokens": 1000})
    # Reservation fits the coarse prompt estimate, but provider-reported input
    # usage is much larger and the settled actual cost crosses the ceiling.
    budget = SpendBudget(max_cost=0.00065)
    agent = GPTAgent(api_key="x", provider="openai", max_retries=0, spend_budget=budget)
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("ping")
    assert budget.remaining == 0.0
    with pytest.raises(SpendBudgetExceededError):
        budget.check()


def test_gpt_agent_unpriced_model_refused_before_provider_call(monkeypatch):
    from neva.agents.gpt import GPTAgent

    calls = []
    _patch_provider(monkeypatch, usage={"prompt_tokens": 5, "completion_tokens": 5}, calls=calls)
    budget = SpendBudget(max_cost=1.0)
    agent = GPTAgent(
        api_key="x",
        provider="openai",
        model="mystery-model",
        max_retries=0,
        spend_budget=budget,
    )
    with pytest.raises(ConfigurationError, match="no pricing entry"):
        agent.respond("ping")
    assert calls == []  # refused before any provider request
    assert budget.spent == 0.0


def test_gpt_agent_without_budget_is_unaffected_for_unpriced_models(monkeypatch):
    from neva.agents.gpt import GPTAgent

    _patch_provider(monkeypatch, usage={"prompt_tokens": 5, "completion_tokens": 5})
    agent = GPTAgent(
        api_key="x",
        provider="openai",
        model="mystery-model",
        max_retries=0,
    )
    assert "priced reply" in agent.respond("ping")


def test_gpt_agent_nan_pricing_raises_configuration_error(monkeypatch):
    from neva.agents.gpt import GPTAgent

    calls = []
    _patch_provider(
        monkeypatch, usage={"prompt_tokens": 1000, "completion_tokens": 1000}, calls=calls
    )
    tracker = CostTracker(
        pricing_per_1k_tokens={"gpt-4o-mini": {"input": float("nan"), "output": 0.0}}
    )
    budget = SpendBudget(max_cost=1.0)
    agent = GPTAgent(
        api_key="x",
        provider="openai",
        max_retries=0,
        cost_tracker=tracker,
        spend_budget=budget,
    )
    with pytest.raises(ConfigurationError, match="non-finite"):
        agent.respond("ping")
    assert calls == []  # invalid pricing is rejected before provider admission
    assert budget.spent == 0.0


def test_shared_budget_enforces_common_ceiling_across_agents(monkeypatch):
    from neva.agents.gpt import GPTAgent

    calls = []
    _patch_provider(
        monkeypatch,
        usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        calls=calls,
    )
    budget = SpendBudget(max_cost=0.0008)  # one turn costs ~0.00075
    first = GPTAgent(api_key="x", provider="openai", max_retries=0, spend_budget=budget)
    second = GPTAgent(api_key="x", provider="openai", max_retries=0, spend_budget=budget)
    assert "priced reply" in first.respond("ping")
    before = len(calls)
    with pytest.raises(SpendBudgetExceededError):
        second.respond("ping")  # worst-case reservation is refused pre-call
    assert len(calls) == before
    assert 0.0 < budget.remaining < 0.0001
    third = GPTAgent(api_key="x", provider="openai", max_retries=0, spend_budget=budget)
    with pytest.raises(SpendBudgetExceededError):
        third.respond("ping")
    assert len(calls) == before  # exhausted budget refuses before any request


def test_spend_budget_rejects_injected_backend_and_bad_types():
    from neva.agents.gpt import GPTAgent

    with pytest.raises(ConfigurationError, match="built-in provider backend"):
        GPTAgent(
            api_key="x",
            llm_backend=lambda prompt: "static",
            spend_budget=SpendBudget(max_cost=1.0),
        )
    with pytest.raises(ConfigurationError, match="SpendBudget instance"):
        GPTAgent(api_key="x", spend_budget=object())


# ---------------------------------------------------------------------------
# Circuit-breaker probe lifecycle on budget exits
# ---------------------------------------------------------------------------


def _half_open_breaker():
    from neva.utils.safety import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=1, cooldown=0.0)
    breaker.record_failure()  # opens the circuit; next allow() admits a probe
    return breaker


def test_budget_refusal_releases_half_open_probe(monkeypatch):
    from neva.agents.gpt import GPTAgent

    breaker = _half_open_breaker()
    budget = SpendBudget(max_cost=1.0)
    budget.consume(1.0)  # exhausted
    agent = GPTAgent(
        api_key="x",
        provider="openai",
        max_retries=0,
        circuit_breaker=breaker,
        spend_budget=budget,
    )
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("ping")
    # The refused probe must be released: the next attempt is admitted and
    # refused for budget again, not stuck as "probe already in flight".
    calls = []
    _patch_provider(monkeypatch, usage={"prompt_tokens": 1, "completion_tokens": 1}, calls=calls)
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("ping")
    assert calls == []


def test_budget_overflow_after_success_completes_half_open_probe(monkeypatch):
    from neva.agents.gpt import GPTAgent

    _patch_provider(monkeypatch, usage={"prompt_tokens": 1000, "completion_tokens": 1000})
    breaker = _half_open_breaker()
    budget = SpendBudget(max_cost=0.00065)
    agent = GPTAgent(
        api_key="x",
        provider="openai",
        max_retries=0,
        circuit_breaker=breaker,
        spend_budget=budget,
    )
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("ping")
    # The provider call succeeded, so the probe must be completed: the next
    # attempt is admitted (and refused for budget) instead of finding the
    # probe permanently in flight.
    calls = []
    _patch_provider(monkeypatch, usage={"prompt_tokens": 1, "completion_tokens": 1}, calls=calls)
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("ping")
    assert calls == []
