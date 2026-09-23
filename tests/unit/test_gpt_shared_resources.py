import asyncio
import threading
import time

import pytest

from neva.agents.gpt import GPTAgent
from neva.utils.exceptions import SpendBudgetExceededError
from neva.utils.provider_resources import _clear_provider_resource_registry_for_tests


class _FakeResponse:
    def __init__(self, text="ok", *, usage=None):
        self._payload = {"choices": [{"message": {"content": text}}]}
        if usage is not None:
            self._payload["usage"] = usage
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_provider_registry():
    _clear_provider_resource_registry_for_tests()
    yield
    _clear_provider_resource_registry_for_tests()


def test_same_account_agents_share_resources_but_different_keys_do_not():
    first = GPTAgent(
        api_key="same",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=2,
    )
    second = GPTAgent(
        api_key="same",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=2,
    )
    other = GPTAgent(
        api_key="different",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=2,
    )

    assert first._provider_resources is second._provider_resources
    assert first._provider_resources is not other._provider_resources


def test_shared_concurrency_limit_serializes_same_account_calls(monkeypatch):
    entered = []
    first_entered = threading.Event()
    release_first = threading.Event()
    lock = threading.Lock()

    def fake_post(url, headers=None, json=None, timeout=None):
        with lock:
            entered.append(json["messages"][-1]["content"])
            position = len(entered)
        if position == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        return _FakeResponse(
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    first = GPTAgent(
        api_key="shared",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=1,
        max_retries=0,
    )
    second = GPTAgent(
        api_key="shared",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=1,
        max_retries=0,
    )

    outcomes = []

    def call(agent, text):
        outcomes.append(agent.respond(text))

    a = threading.Thread(target=call, args=(first, "first"))
    b = threading.Thread(target=call, args=(second, "second"))
    a.start()
    assert first_entered.wait(timeout=1)
    b.start()
    time.sleep(0.1)
    assert len(entered) == 1
    release_first.set()
    a.join(timeout=2)
    b.join(timeout=2)

    assert not a.is_alive()
    assert not b.is_alive()
    assert len(entered) == 2
    assert sorted(outcomes) == ["ok", "ok"]


def test_shared_provider_spend_limit_reserves_before_second_call(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return _FakeResponse(
            usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        )

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    first = GPTAgent(
        api_key="shared-spend",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=2,
        provider_spend_limit=0.0008,
        max_retries=0,
    )
    second = GPTAgent(
        api_key="shared-spend",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=2,
        provider_spend_limit=0.0008,
        max_retries=0,
    )

    assert first.respond("first") == "ok"
    before = len(calls)
    with pytest.raises(SpendBudgetExceededError):
        second.respond("second")

    assert len(calls) == before
    assert first._provider_resources is second._provider_resources
    assert first._provider_resources.spent == pytest.approx(0.00075)


def test_billing_reconciler_updates_shared_spend_before_admission(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return _FakeResponse(usage={"prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(
        api_key="billing",
        provider="openai",
        provider_rate=None,
        provider_spend_limit=1.0,
        billing_reconciler=lambda: 0.9999,
        max_retries=0,
    )

    with pytest.raises(SpendBudgetExceededError):
        agent.respond("blocked by authoritative spend")

    assert calls == []
    assert agent._provider_resources.spent == pytest.approx(0.9999)


def test_async_cancellation_removes_waiting_provider_admission(monkeypatch):
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["messages"][-1]["content"])
        if len(calls) == 1:
            first_entered.set()
            assert release_first.wait(timeout=3)
        return _FakeResponse(usage={"prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    first = GPTAgent(
        api_key="cancel",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=1,
        max_retries=0,
    )
    second = GPTAgent(
        api_key="cancel",
        provider="openai",
        provider_rate=None,
        max_provider_concurrency=1,
        max_retries=0,
    )

    async def scenario():
        running = asyncio.create_task(first.areceive("first"))
        while not first_entered.is_set():
            await asyncio.sleep(0.01)

        waiting = asyncio.create_task(second.areceive("second"))
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release_first.set()
        assert await running == "ok"

        await asyncio.sleep(0.1)
        assert len(calls) == 1

    asyncio.run(scenario())


def test_legacy_explicit_rate_limiter_disables_shared_rate_but_keeps_concurrency():
    from neva.utils.safety import RateLimiter

    limiter = RateLimiter(rate=1, per=60)
    agent = GPTAgent(
        api_key="legacy",
        provider="openai",
        rate_limiter=limiter,
        max_provider_concurrency=3,
    )

    assert agent._rate_limiter is limiter
    assert agent._provider_resources.rate is None
    assert agent._provider_resources.max_concurrency == 3


def test_async_cancel_during_inflight_call_settles_once_and_caches(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        entered.set()
        assert release.wait(timeout=3)
        return _FakeResponse(
            text="paid reply",
            usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        )

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(
        api_key="cancel-inflight",
        provider="openai",
        provider_rate=None,
        provider_spend_limit=1.0,
        max_retries=0,
    )

    async def scenario():
        task = asyncio.create_task(agent.arespond("inflight"))
        while not entered.is_set():
            await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        release.set()
        for _ in range(200):
            resources = agent._provider_resources
            if resources.spent > 0 and resources.reserved == 0:
                break
            await asyncio.sleep(0.01)

        assert agent._provider_resources.spent == pytest.approx(0.00075)
        assert agent._provider_resources.reserved == 0.0

    asyncio.run(scenario())

    before = len(calls)
    assert agent.respond("inflight") == "paid reply"
    assert len(calls) == before
    assert agent._provider_resources.spent == pytest.approx(0.00075)
