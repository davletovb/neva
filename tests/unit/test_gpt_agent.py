import pytest
import requests

from neva.agents.gpt import (
    GPTAgent,
    _chat_completions_url,
    _extract_chat_content,
    _provider_usage_from_gemini,
)
from neva.utils.exceptions import BackendError, CircuitOpenError, ConfigurationError
from neva.utils.metrics import TokenUsageTracker
from neva.utils.safety import CircuitBreaker


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def json(self):
        return self._payload


def test_default_model_follows_provider():
    grok = GPTAgent(provider="grok", llm_backend=lambda prompt: prompt)
    openai_agent = GPTAgent(provider="openai", llm_backend=lambda prompt: prompt)
    assert grok.model == "grok-4.5"
    assert openai_agent.model == "gpt-4o-mini"


def test_respond_uses_injected_backend():
    agent = GPTAgent(name="Scout", llm_backend=lambda prompt: "ack")
    assert agent.respond("hello") == "ack"


def test_missing_api_key_raises_configuration_error():
    agent = GPTAgent(name="Scout")
    agent.set_cache(None)
    with pytest.raises(ConfigurationError):
        agent.respond("hello")


def test_grok_uses_chat_completions_endpoint(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse({"choices": [{"message": {"content": "Grok here."}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=0)
    assert agent.respond("ping").endswith("Grok here.")
    assert calls[0]["url"] == "https://api.x.ai/v1/chat/completions"
    assert calls[0]["json"]["max_tokens"] == 1024
    assert calls[0]["json"]["model"] == "grok-4.5"


def test_openai_uses_chat_completions_endpoint(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == "https://api.openai.com/v1/chat/completions"
        return _FakeResponse({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(api_key="sk-test", provider="openai", name="Scout", max_retries=0)
    assert "hi" in agent.respond("ping")


def test_chat_completions_url_appends_path():
    assert _chat_completions_url(None, "https://api.x.ai/v1/chat/completions").endswith(
        "chat/completions"
    )
    assert (
        _chat_completions_url("https://api.x.ai/v1", "https://example.invalid")
        == "https://api.x.ai/v1/chat/completions"
    )
    assert (
        _chat_completions_url("https://api.x.ai/v1/chat/completions", "https://example.invalid")
        == "https://api.x.ai/v1/chat/completions"
    )


def test_extract_chat_content_handles_list_blocks():
    payload = {
        "choices": [
            {"message": {"content": [{"type": "text", "text": "Hello"}, {"text": " world"}]}}
        ]
    }
    assert _extract_chat_content(payload) == "Hello world"


def test_empty_provider_response_raises(monkeypatch):
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: _FakeResponse({"choices": [{"message": {"content": ""}}]}),
    )
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=0)
    with pytest.raises(BackendError):
        agent.respond("ping")


def test_unsupported_provider_raises():
    agent = GPTAgent(api_key="k", provider="unknown", name="Scout", max_retries=0)
    with pytest.raises(ConfigurationError):
        agent._invoke_provider("hello")


def test_backend_uses_internal_cache(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return _FakeResponse({"choices": [{"message": {"content": "cached"}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=0)
    backend = agent._default_backend()
    assert backend("hello") == "cached"
    assert backend("hello") == "cached"
    assert len(calls) == 1


def test_retries_then_succeeds(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("boom")
        return _FakeResponse({"choices": [{"message": {"content": "recovered"}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _seconds: None)
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=1)
    assert "recovered" in agent.respond("ping")
    assert len(calls) == 2


def test_extract_chat_content_fallbacks():
    assert _extract_chat_content({"choices": [{"text": "plain"}]}) == "plain"
    assert _extract_chat_content({"content": "top-level"}) == "top-level"
    assert _extract_chat_content({"choices": [{"message": {"content": [{"value": "v"}]}}]}) == "v"
    assert _extract_chat_content({"choices": [{"message": {"content": ["a", "b"]}}]}) == "ab"
    assert _extract_chat_content({}) is None


def test_retry_exhausted_raises(monkeypatch):
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _seconds: None)
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=0)
    with pytest.raises(BackendError):
        agent.respond("ping")


def test_extra_headers_are_sent(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert headers["X-Test"] == "1"
        assert url == "https://api.x.ai/v1/chat/completions"
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(
        api_key="xai-test",
        provider="grok",
        name="Scout",
        max_retries=0,
        extra_headers={"X-Test": "1"},
        api_base="https://api.x.ai/v1",
    )
    assert "ok" in agent.respond("ping")


def test_anthropic_requires_package(monkeypatch):
    def _missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr("neva.agents.gpt.importlib.import_module", _missing)
    agent = GPTAgent(api_key="k", provider="anthropic", name="Scout", max_retries=0)
    with pytest.raises(ConfigurationError):
        agent._invoke_anthropic("hello")


def test_gemini_requires_package(monkeypatch):
    def _missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr("neva.agents.gpt.importlib.import_module", _missing)
    agent = GPTAgent(api_key="k", provider="gemini", name="Scout", max_retries=0)
    with pytest.raises(ConfigurationError):
        agent._invoke_gemini("hello")


def test_anthropic_success_and_empty(monkeypatch):
    class _Block:
        def __init__(self, text):
            self.text = text

    class _Messages:
        def __init__(self, blocks):
            self._blocks = blocks

        def create(self, **kwargs):
            del kwargs
            return type("Resp", (), {"content": self._blocks})()

    class _Client:
        _blocks: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = _Messages(type(self)._blocks)

    def _install(blocks):
        class _Bound(_Client):
            _blocks = blocks

        class _Mod:
            Anthropic = _Bound

        monkeypatch.setattr(
            "neva.agents.gpt.importlib.import_module",
            lambda name: _Mod if name == "anthropic" else (_ for _ in ()).throw(ImportError(name)),
        )

    agent = GPTAgent(
        api_key="k",
        provider="anthropic",
        name="Scout",
        max_retries=0,
        api_base="https://api.anthropic.test",
    )
    _install([_Block("claude-ok")])
    assert agent._invoke_anthropic("hello") == "claude-ok"

    _install([{"text": "dict-ok"}])
    assert agent._invoke_anthropic("hello") == "dict-ok"

    _install([])
    with pytest.raises(BackendError):
        agent._invoke_anthropic("hello")

    _install([_Block("")])
    with pytest.raises(BackendError):
        agent._invoke_anthropic("hello")


def test_anthropic_passes_request_timeout(monkeypatch):
    captured = {}
    client_options = {}

    class _Block:
        text = "ok"

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Resp", (), {"content": [_Block()]})()

    class _Client:
        def __init__(self, **kwargs):
            client_options.update(kwargs)
            self.messages = _Messages()

    class _Mod:
        Anthropic = _Client

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _Mod if name == "anthropic" else (_ for _ in ()).throw(ImportError(name)),
    )
    agent = GPTAgent(api_key="k", provider="anthropic", max_retries=0, request_timeout=12.5)
    assert agent._invoke_anthropic("hello") == "ok"
    assert captured["timeout"] == 12.5
    assert client_options["max_retries"] == 0
    assert captured["max_tokens"] == 1024


def test_gemini_passes_request_timeout(monkeypatch):
    captured = {}

    class _Model:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt, **kwargs):
            captured.update(kwargs)
            return type("Resp", (), {"text": "ok"})()

    class _GenAI:
        @staticmethod
        def configure(api_key):
            del api_key

        GenerativeModel = _Model

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _GenAI
        if name == "google.generativeai"
        else (_ for _ in ()).throw(ImportError(name)),
    )
    agent = GPTAgent(api_key="k", provider="gemini", max_retries=0, request_timeout=7.5)
    assert agent._invoke_gemini("hello") == "ok"
    assert captured["request_options"] == {"timeout": 7.5, "retry": None}
    assert captured["generation_config"] == {"max_output_tokens": 1024}


def test_gemini_success_and_candidates(monkeypatch):
    class _Model:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt, **kwargs):
            del prompt
            return type("Resp", (), {"text": "gemini-ok"})()

    class _GenAI:
        configured = []

        @staticmethod
        def configure(api_key):
            _GenAI.configured.append(api_key)

        GenerativeModel = _Model

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _GenAI
        if name == "google.generativeai"
        else (_ for _ in ()).throw(ImportError(name)),
    )
    agent = GPTAgent(api_key="k", provider="gemini", name="Scout", max_retries=0)
    assert agent._invoke_gemini("hello") == "gemini-ok"

    class _Part:
        text = "from-candidate"

    class _Candidate:
        content = type("C", (), {"parts": [_Part()]})()

    class _EmptyTextModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt, **kwargs):
            del prompt
            return type("Resp", (), {"text": "", "candidates": [_Candidate()]})()

    class _CandidateGenAI:
        @staticmethod
        def configure(api_key):
            del api_key

        GenerativeModel = _EmptyTextModel

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _CandidateGenAI
        if name == "google.generativeai"
        else (_ for _ in ()).throw(ImportError(name)),
    )
    assert agent._invoke_gemini("hello") == "from-candidate"


def test_http_error_is_retried_then_fails(monkeypatch):
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: _FakeResponse({}, status_code=500),
    )
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _seconds: None)
    agent = GPTAgent(api_key="xai-test", provider="xai", name="Scout", max_retries=0)
    with pytest.raises(BackendError):
        agent.respond("ping")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_errors_are_not_retried(monkeypatch, status):
    calls, sleeps = [], []

    def post(*args, **kwargs):
        calls.append(1)
        return _FakeResponse({}, status_code=status)

    monkeypatch.setattr("neva.agents.gpt.requests.post", post)
    monkeypatch.setattr("neva.agents.gpt.sleep", sleeps.append)
    agent = GPTAgent(api_key="test", max_retries=3)
    with pytest.raises(BackendError) as caught:
        agent.respond("hello")
    assert isinstance(caught.value.__cause__, requests.HTTPError)
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503, 529])
def test_transient_http_errors_respect_retry_budget(monkeypatch, status):
    calls, sleeps = [], []

    def post(*args, **kwargs):
        calls.append(1)
        return _FakeResponse({}, status_code=status)

    monkeypatch.setattr("neva.agents.gpt.requests.post", post)
    monkeypatch.setattr("neva.agents.gpt.sleep", sleeps.append)
    with pytest.raises(BackendError):
        GPTAgent(api_key="test", max_retries=2).respond("hello")
    assert len(calls) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("error", [ConfigurationError("bad config"), ValueError("bad json")])
def test_local_errors_are_not_retried(monkeypatch, error):
    calls, sleeps = [], []

    def invoke(prompt):
        calls.append(prompt)
        raise error

    agent = GPTAgent(api_key="test")
    monkeypatch.setattr(agent, "_invoke_provider", invoke)
    monkeypatch.setattr("neva.agents.gpt.sleep", sleeps.append)
    expected = ConfigurationError if isinstance(error, ConfigurationError) else BackendError
    with pytest.raises(expected):
        agent.respond("hello")
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("status, attempts", [(401, 1), (429, 3), (503, 3)])
@pytest.mark.parametrize("sdk", ["anthropic", "google"])
def test_sdk_errors_are_classified_without_importing_sdk(monkeypatch, status, attempts, sdk):
    class SDKError(Exception):
        pass

    error = SDKError("provider error")
    if sdk == "anthropic":
        error.status_code = status
    else:
        error.code = status
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        raise error

    agent = GPTAgent(api_key="test", max_retries=2)
    monkeypatch.setattr(agent, "_invoke_provider", invoke)
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _: None)
    with pytest.raises(BackendError):
        agent.respond("hello")
    assert len(calls) == attempts


def test_chat_completions_url_appends_to_custom_bases():
    assert (
        _chat_completions_url("https://proxy.example/openai", "https://example.invalid")
        == "https://proxy.example/openai/chat/completions"
    )
    assert (
        _chat_completions_url("https://proxy.example/openai/", "https://example.invalid")
        == "https://proxy.example/openai/chat/completions"
    )
    assert (
        _chat_completions_url(
            "https://proxy.example/openai/chat/completions/", "https://example.invalid"
        )
        == "https://proxy.example/openai/chat/completions"
    )


def test_chat_messages_without_history_is_single_user_turn():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout")
    assert agent._chat_messages("ping") == [{"role": "user", "content": "ping"}]


def test_chat_messages_include_recorded_turns():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout")
    agent.receive("hello", sender="user")
    messages = agent._chat_messages("next")
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1] == {"role": "assistant", "content": "ack"}
    assert messages[2] == {"role": "user", "content": "next"}


def test_chat_messages_prefix_other_speakers():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout")
    agent.receive("knock", sender="Innkeeper")
    messages = agent._chat_messages("reply")
    assert messages[0] == {"role": "user", "content": "Innkeeper: knock"}
    assert messages[1] == {"role": "assistant", "content": "ack"}


def test_prompt_with_history_flattens_turns():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout", provider="gemini")
    assert agent._prompt_with_history("now") == "now"
    agent.receive("hello", sender="user")
    flattened = agent._prompt_with_history("now")
    assert flattened.startswith("Conversation so far:")
    assert "user: hello" in flattened
    assert "Scout: ack" in flattened
    assert flattened.endswith("now")


def test_provider_payload_includes_conversation_history(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return _FakeResponse({"choices": [{"message": {"content": f"r{len(calls)}"}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=0)
    agent.receive("hello", sender="user")
    assert len(calls[0]["messages"]) == 1
    assert calls[0]["messages"][0]["role"] == "user"
    assert "hello" in calls[0]["messages"][0]["content"]

    agent.receive("follow up", sender="user")
    messages = calls[1]["messages"]
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1] == {"role": "assistant", "content": "r1"}
    assert messages[2]["role"] == "user"
    assert "follow up" in messages[2]["content"]


def test_cache_key_includes_conversation_history():
    calls = []

    def backend(prompt: str) -> str:
        calls.append(prompt)
        return f"r{len(calls)}"

    agent = GPTAgent(llm_backend=backend, name="Scout")
    agent.receive("hello", sender="user")
    agent.receive("hello", sender="user")
    assert len(calls) == 2


def test_max_context_chars_must_be_positive():
    with pytest.raises(ConfigurationError, match="max_context_chars"):
        GPTAgent(llm_backend=lambda prompt: "ack", max_context_chars=0)


def test_history_window_drops_oldest_turns():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout", max_context_chars=24)
    agent._remember("user", "11111111")
    agent._remember("Scout", "ack")
    agent._remember("user", "22222222")
    agent._remember("Scout", "ack")
    window = agent._history_window("3333")
    assert [turn.message for turn in window] == ["22222222", "ack"]
    messages = agent._chat_messages("3333")
    assert [item["content"] for item in messages] == ["22222222", "ack", "3333"]
    assert agent._request_text("3333") == "22222222\nack\n3333"


def test_history_window_drops_leading_assistant():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout", max_context_chars=10)
    agent._remember("user", "hello")
    agent._remember("Scout", "ack")
    assert agent._history_window("prompt") == []
    assert agent._chat_messages("prompt") == [{"role": "user", "content": "prompt"}]


def test_provider_payload_omits_turns_outside_the_window(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(
        api_key="xai-test",
        provider="grok",
        name="Scout",
        max_retries=0,
        max_context_chars=24,
    )
    agent._remember("user", "11111111")
    agent._remember("Scout", "ack")
    agent._remember("user", "22222222")
    agent._remember("Scout", "ack")
    agent._invoke_grok("3333")
    contents = [item["content"] for item in calls[0]["messages"]]
    assert contents == ["22222222", "ack", "3333"]


def test_oversized_current_prompt_raises():
    agent = GPTAgent(llm_backend=lambda prompt: "ack", name="Scout", max_context_chars=8)
    with pytest.raises(ConfigurationError, match="exceeds max_context_chars=8"):
        agent._history_window("123456789")
    with pytest.raises(ConfigurationError, match="exceeds max_context_chars=8"):
        agent.respond("123456789")


def test_gemini_serialized_history_respects_budget():
    agent = GPTAgent(
        llm_backend=lambda prompt: "ack",
        name="Scout",
        provider="gemini",
        max_context_chars=24,
    )
    agent._remember("user", "22222222")
    agent._remember("Scout", "ack")
    flattened = agent._prompt_with_history("3333")
    assert flattened == "3333"
    assert len(flattened) <= 24


def test_gemini_keeps_turns_that_fit_serialized_budget():
    agent = GPTAgent(
        llm_backend=lambda prompt: "ack",
        name="Scout",
        provider="gemini",
        max_context_chars=52,
    )
    agent._remember("user", "11111111")
    agent._remember("Scout", "ack")
    agent._remember("user", "22222222")
    agent._remember("Scout", "ack")
    flattened = agent._prompt_with_history("3333")
    assert flattened == ("Conversation so far:\nuser: 22222222\nScout: ack\n\n3333")
    assert len(flattened) == 52
    assert "user: 11111111" not in flattened
    assert agent._request_text("3333") == flattened


def test_gemini_invoke_sends_budgeted_text(monkeypatch):
    captured = []

    class _Model:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt, **kwargs):
            captured.append(prompt)
            return type("Resp", (), {"text": "ok"})()

    class _GenAI:
        @staticmethod
        def configure(api_key):
            del api_key

        GenerativeModel = _Model

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _GenAI
        if name == "google.generativeai"
        else (_ for _ in ()).throw(ImportError(name)),
    )
    tight = GPTAgent(
        api_key="k",
        provider="gemini",
        name="Scout",
        max_retries=0,
        max_context_chars=24,
    )
    tight._remember("user", "22222222")
    tight._remember("Scout", "ack")
    tight._invoke_gemini("3333")
    assert captured == ["3333"]

    captured.clear()
    roomy = GPTAgent(
        api_key="k",
        provider="gemini",
        name="Scout",
        max_retries=0,
        max_context_chars=52,
    )
    roomy._remember("user", "22222222")
    roomy._remember("Scout", "ack")
    roomy._invoke_gemini("3333")
    assert len(captured[0]) == 52
    assert captured[0] == roomy._request_text("3333")


def test_gemini_token_tracker_counts_flattened_history(monkeypatch):
    tracker = TokenUsageTracker()
    captured = []

    class _Model:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt, **kwargs):
            captured.append(prompt)
            return type("Resp", (), {"text": "ok"})()

    class _GenAI:
        @staticmethod
        def configure(api_key):
            del api_key

        GenerativeModel = _Model

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _GenAI
        if name == "google.generativeai"
        else (_ for _ in ()).throw(ImportError(name)),
    )
    agent = GPTAgent(
        api_key="k",
        provider="gemini",
        name="Scout",
        max_retries=0,
        token_tracker=tracker,
    )
    agent.receive("hello there friend", sender="user")
    agent.receive("follow up question", sender="user")
    assert "Conversation so far:" in captured[1]
    assert tracker.records[1][0] > tracker.records[0][0]


def test_gemini_uses_provider_usage_metadata(monkeypatch):
    tracker = TokenUsageTracker()

    class _Usage:
        prompt_token_count = 21
        candidates_token_count = 4

    class _Model:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt, **kwargs):
            return type("Resp", (), {"text": "ok", "usage_metadata": _Usage()})()

    class _GenAI:
        @staticmethod
        def configure(api_key):
            del api_key

        GenerativeModel = _Model

    monkeypatch.setattr(
        "neva.agents.gpt.importlib.import_module",
        lambda name: _GenAI
        if name == "google.generativeai"
        else (_ for _ in ()).throw(ImportError(name)),
    )
    agent = GPTAgent(
        api_key="k",
        provider="gemini",
        name="Scout",
        max_retries=0,
        token_tracker=tracker,
    )
    assert "ok" in agent.respond("hello")
    assert tracker.records[-1] == (21, 4)
    assert tracker.estimated_calls == 0


def test_provider_usage_from_gemini_accepts_dicts_and_objects():
    assert _provider_usage_from_gemini({}) is None
    assert _provider_usage_from_gemini(
        {"usage_metadata": {"prompt_token_count": 2, "candidates_token_count": 1}}
    ) == {"prompt_tokens": 2, "completion_tokens": 1}


def test_grok_usage_is_priced(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
            }
        )

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(api_key="xai-test", provider="grok", name="Scout", max_retries=0)
    assert "ok" in agent.respond("ping")
    assert agent._cost_tracker.total_cost() == pytest.approx(0.008)


def test_circuit_opens_after_retryable_failures(monkeypatch):
    calls = []
    now = [0.0]

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return _FakeResponse({}, status_code=503)

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _seconds: None)
    monkeypatch.setattr("neva.utils.safety.time.monotonic", lambda: now[0])
    breaker = CircuitBreaker(failure_threshold=2, cooldown=30.0)
    agent = GPTAgent(
        api_key="xai-test",
        provider="grok",
        name="Scout",
        max_retries=0,
        circuit_breaker=breaker,
    )
    with pytest.raises(BackendError):
        agent.respond("a")
    with pytest.raises(BackendError):
        agent.respond("b")
    assert len(calls) == 2
    with pytest.raises(CircuitOpenError):
        agent.respond("c")
    assert len(calls) == 2
    now[0] += 30.0
    with pytest.raises(BackendError):
        agent.respond("d")
    assert len(calls) == 3


def test_permanent_errors_do_not_open_the_circuit(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return _FakeResponse({}, status_code=401)

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    breaker = CircuitBreaker(failure_threshold=1, cooldown=30.0)
    agent = GPTAgent(
        api_key="xai-test",
        provider="grok",
        name="Scout",
        max_retries=0,
        circuit_breaker=breaker,
    )
    with pytest.raises(BackendError):
        agent.respond("a")
    with pytest.raises(BackendError):
        agent.respond("b")
    assert len(calls) == 2


def test_circuit_counts_each_retryable_attempt(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return _FakeResponse({}, status_code=503)

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _seconds: None)
    breaker = CircuitBreaker(failure_threshold=2, cooldown=30.0)
    agent = GPTAgent(
        api_key="xai-test",
        provider="grok",
        name="Scout",
        max_retries=5,
        circuit_breaker=breaker,
    )
    with pytest.raises(CircuitOpenError):
        agent.respond("outage")
    assert len(calls) == 2


def test_half_open_probe_401_does_not_stick(monkeypatch):
    calls = []
    now = [0.0]
    statuses = [503, 503, 401, 401]

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return _FakeResponse({}, status_code=statuses[len(calls) - 1])

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    monkeypatch.setattr("neva.agents.gpt.sleep", lambda _seconds: None)
    monkeypatch.setattr("neva.utils.safety.time.monotonic", lambda: now[0])
    breaker = CircuitBreaker(failure_threshold=2, cooldown=30.0)
    agent = GPTAgent(
        api_key="xai-test",
        provider="grok",
        name="Scout",
        max_retries=0,
        circuit_breaker=breaker,
    )
    with pytest.raises(BackendError):
        agent.respond("a")
    with pytest.raises(BackendError):
        agent.respond("b")
    now[0] += 30.0
    with pytest.raises(BackendError):
        agent.respond("c")
    assert len(calls) == 3
    with pytest.raises(CircuitOpenError, match="retry after"):
        agent.respond("d")
    assert len(calls) == 3
    now[0] += 30.0
    with pytest.raises(BackendError):
        agent.respond("e")
    assert len(calls) == 4
