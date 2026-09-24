"""Opt-in model token envelopes, including actual provider request framing."""

import pytest
import requests

from neva.agents.gpt import _RAW_PROMPT_MODE, GPTAgent
from neva.utils.context_budget import ModelContextBudget, openai_chat_counter
from neva.utils.exceptions import ConfigurationError, SpendBudgetExceededError
from neva.utils.metrics import CostTracker, SpendBudget


def chat_tokens(request):
    assert isinstance(request, list)
    return 3 + sum(3 + len(message["role"]) + len(message["content"]) for message in request)


def profile(*, provider="openai", model="gpt-4o-mini", limit=34, counter=chat_tokens):
    return ModelContextBudget(provider, model, limit, counter, "test-counter-v1")


@pytest.mark.parametrize("limit", [0, True, 1.5, "12"])
def test_budget_requires_positive_integer_limit(limit):
    with pytest.raises(ConfigurationError, match="max_tokens"):
        profile(limit=limit)


def test_counter_identity_and_callable_are_required():
    with pytest.raises(ConfigurationError, match="counter_id"):
        ModelContextBudget("openai", "gpt-4o-mini", 100, chat_tokens, "")
    with pytest.raises(ConfigurationError, match="count_request_tokens"):
        ModelContextBudget("openai", "gpt-4o-mini", 100, None, "v1")


@pytest.mark.parametrize("invalid", [-1, 0, 1.5, True, "2"])
def test_counter_fails_closed_on_invalid_result(invalid):
    agent = GPTAgent(context_budget=profile(counter=lambda _: invalid), max_output_tokens=1)
    with pytest.raises(ConfigurationError, match="counter must return"):
        agent._history_window("ping")


def test_counter_failure_fails_closed():
    def broken(request):
        raise ValueError("tokenizer unavailable")

    agent = GPTAgent(context_budget=profile(counter=broken), max_output_tokens=1)
    with pytest.raises(ConfigurationError, match="token counter failed"):
        agent._history_window("ping")


def test_openai_counter_uses_selected_model_encoding_and_message_framing(monkeypatch):
    seen = []

    class Encoding:
        @staticmethod
        def encode(text, *, disallowed_special):
            assert disallowed_special == ()
            return list(text)

    class Tiktoken:
        @staticmethod
        def encoding_for_model(model):
            seen.append(model)
            return Encoding()

    monkeypatch.setattr("neva.utils.context_budget.importlib.import_module", lambda _: Tiktoken)
    counter = openai_chat_counter("gpt-4o-mini")
    assert seen == ["gpt-4o-mini"]
    assert counter([{"role": "user", "content": "hello"}]) == 3 + 3 + 4 + 5
    with pytest.raises(ConfigurationError, match="role/content"):
        counter("hello")
    with pytest.raises(ConfigurationError, match="does not support"):
        openai_chat_counter("unknown-model")
    with pytest.raises(ConfigurationError, match="counter provider/model"):
        ModelContextBudget("gemini", "gpt-4o-mini", 100, counter, "openai-counter")


def test_special_token_looking_text_in_history_is_counted_as_ordinary_text(monkeypatch):
    class Encoding:
        @staticmethod
        def encode(text, *, disallowed_special):
            assert disallowed_special == ()
            return list(text)

    class Tiktoken:
        @staticmethod
        def encoding_for_model(model):
            return Encoding()

    monkeypatch.setattr("neva.utils.context_budget.importlib.import_module", lambda _: Tiktoken)
    agent = GPTAgent(
        name="Scout",
        llm_backend=lambda prompt: "ok",
        context_budget=ModelContextBudget(
            "openai", "gpt-4o-mini", 300, openai_chat_counter("gpt-4o-mini"), "test-v1"
        ),
        max_output_tokens=2,
    )
    agent._remember("user", "Pasted docs mention <|endoftext|> here")
    for message in ("first", "second"):
        assert agent._history_window(message)[0].message.startswith("Pasted docs")
        assert agent.respond(message) == "ok"


def test_declared_tiktoken_release_maps_the_documented_models():
    """The tiktoken release we declare must know the models the counter accepts.

    Every other counter test replaces the tiktoken import with a fake, so the
    declared floor is otherwise never exercised: with tiktoken 0.6.0 (the first
    pin) ``encoding_for_model("gpt-4o-mini")`` raises KeyError and
    ``openai_chat_counter`` fails with ConfigurationError.
    """

    tiktoken = pytest.importorskip("tiktoken")

    assert tiktoken.encoding_name_for_model("gpt-4o-mini") == "o200k_base"

    try:
        counter = openai_chat_counter("gpt-4o-mini")
    except ConfigurationError as exc:  # pragma: no cover - needs a reachable encoding download.
        pytest.skip(f"tiktoken encoding unavailable in this environment: {exc}")
    assert counter([{"role": "user", "content": "hello world"}]) > 0


def test_openai_counter_requires_optional_tokenizer(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr("neva.utils.context_budget.importlib.import_module", missing)
    with pytest.raises(ConfigurationError, match="requires tiktoken"):
        openai_chat_counter("gpt-4o-mini")


def test_encoding_cache_network_failure_is_configuration_error(monkeypatch):
    class Tiktoken:
        @staticmethod
        def encoding_for_model(model):
            raise requests.exceptions.ProxyError("offline")

    monkeypatch.setattr("neva.utils.context_budget.importlib.import_module", lambda _: Tiktoken)
    with pytest.raises(ConfigurationError, match="encoding cache, and network"):
        openai_chat_counter("gpt-4o-mini")


def test_profile_must_match_model_and_reserve_output_tokens():
    with pytest.raises(ConfigurationError, match="provider/model"):
        GPTAgent(provider="grok", context_budget=profile(), max_output_tokens=1)
    with pytest.raises(ConfigurationError, match="provider/model"):
        GPTAgent(model="different", context_budget=profile(), max_output_tokens=1)
    with pytest.raises(ConfigurationError, match="max_output_tokens"):
        GPTAgent(context_budget=profile(limit=8), max_output_tokens=8)
    with pytest.raises(ConfigurationError, match="max_output_tokens"):
        GPTAgent(context_budget=profile(), max_output_tokens=True)
    assert profile(provider="OpenAI").provider == "openai"
    GPTAgent(context_budget=profile(provider="OpenAI"), max_output_tokens=1)


def test_history_trimming_includes_roles_overhead_and_output_reservation():
    agent = GPTAgent(name="Scout", context_budget=profile(limit=34), max_output_tokens=5)
    agent._remember("user", "oldold")
    agent._remember("Scout", "ack")
    agent._remember("user", "newest")
    assert [turn.message for turn in agent._history_window("now")] == ["newest"]
    request = agent._chat_messages("now")
    assert [item["content"] for item in request] == ["newest", "now"]
    assert chat_tokens(request) + 5 <= 34


def test_current_request_that_cannot_fit_rejects_before_http_or_cache(monkeypatch):
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: pytest.fail("unexpected HTTP request"),
    )
    agent = GPTAgent(
        api_key="test", context_budget=profile(limit=20), max_output_tokens=5, max_retries=0
    )
    with pytest.raises(ConfigurationError, match=r"input tokens \+ 5 reserved output tokens"):
        agent.respond("too long")
    with pytest.raises(ConfigurationError, match="context budget"):
        agent.stream_response("too long")


def test_provider_payload_matches_budgeted_request(monkeypatch):
    seen = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, *, headers, json, timeout):
        seen.append(json)
        return Response()

    monkeypatch.setattr("neva.agents.gpt.requests.post", post)
    agent = GPTAgent(
        api_key="test",
        name="Scout",
        context_budget=profile(limit=110),
        max_output_tokens=5,
        max_retries=0,
    )
    agent._remember("user", "oldold")
    agent._remember("Scout", "ack")
    agent._remember("user", "newest")
    assert agent.respond("now") == "ok"
    assert seen[0]["messages"][0]["content"] == "newest"
    assert seen[0]["messages"][1]["content"].endswith(" now")
    assert chat_tokens(seen[0]["messages"]) + seen[0]["max_tokens"] <= 110


def test_gemini_counter_receives_flattened_provider_request():
    observed = []

    def count(request):
        assert isinstance(request, str)
        observed.append(request)
        return len(request)

    agent = GPTAgent(
        provider="gemini",
        name="Scout",
        context_budget=profile(
            provider="gemini", model="gemini-1.5-flash", limit=55, counter=count
        ),
        max_output_tokens=3,
    )
    agent._remember("user", "22222222")
    agent._remember("Scout", "ack")
    assert agent._prompt_with_history("3333") == (
        "Conversation so far:\nuser: 22222222\nScout: ack\n\n3333"
    )
    assert observed[-1] == agent._request_text("3333")


def test_raw_model_mode_excludes_recorded_history_but_checks_current_request():
    agent = GPTAgent(name="Scout", context_budget=profile(limit=20), max_output_tokens=1)
    agent._remember("user", "oldold")
    token = _RAW_PROMPT_MODE.set(True)
    try:
        assert agent._history_window("now") == []
        with pytest.raises(ConfigurationError, match="context budget"):
            agent._history_window("far too long")
    finally:
        _RAW_PROMPT_MODE.reset(token)


def test_cache_identity_changes_with_budget_and_model_change_rejects():
    first = GPTAgent(context_budget=profile(limit=34), max_output_tokens=1)
    second = GPTAgent(context_budget=profile(limit=35), max_output_tokens=1)
    assert first._scoped_key("ping") != second._scoped_key("ping")
    first.model = "changed"
    with pytest.raises(ConfigurationError, match="no longer matches"):
        first._scoped_key("ping")


def test_long_history_counts_logarithmically_many_requests(monkeypatch):
    calls = []

    def counting(request):
        calls.append(len(request))
        return chat_tokens(request)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("neva.agents.gpt.requests.post", lambda *args, **kwargs: Response())
    agent = GPTAgent(
        api_key="test",
        name="Scout",
        max_retries=0,
        max_context_chars=1_000_000,
        context_budget=profile(limit=1_000_000, counter=counting),
        max_output_tokens=1,
    )
    for index in range(800):
        agent._remember("user", str(index))
    assert agent.respond("now") == "ok"
    assert len(calls) < 100
    assert sum(calls) < 80_000


def test_spend_preflight_reserves_model_count_including_message_overhead(monkeypatch):
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: pytest.fail("unexpected HTTP request"),
    )
    cost = CostTracker()
    agent = GPTAgent(
        api_key="test",
        name="Scout",
        context_budget=profile(limit=100),
        max_output_tokens=5,
        max_retries=0,
        spend_budget=SpendBudget(0.01),
        cost_tracker=cost,
    )
    request = agent._chat_messages("now")
    assert agent._reservation_cost("now") == pytest.approx(
        cost.cost_for("gpt-4o-mini", prompt_tokens=chat_tokens(request), response_tokens=5)
    )
    agent._spend_budget = SpendBudget(0.000001)
    with pytest.raises(SpendBudgetExceededError):
        agent.respond("now")


def test_reproducibility_manifest_includes_counter_identity():
    from neva.environments import BasicEnvironment
    from neva.schedulers import RoundRobinScheduler
    from neva.utils.reproducibility import create_run_manifest

    agent = GPTAgent(llm_backend=lambda prompt: "ok", context_budget=profile(), max_output_tokens=1)
    env = BasicEnvironment("test", "context budget", RoundRobinScheduler())
    env.register_agent(agent)
    manifest = create_run_manifest(env, seed=1)
    assert manifest.to_dict()["agents"][0]["generation"]["context_budget"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 34,
        "counter_id": "test-counter-v1",
    }
