import pytest
import requests

from neva.agents.gpt import GPTAgent, _chat_completions_url, _extract_chat_content
from neva.utils.exceptions import BackendError, ConfigurationError


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

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


def test_gemini_success_and_candidates(monkeypatch):
    class _Model:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt):
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

        def generate_content(self, prompt):
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


def test_chat_completions_url_passthrough():
    assert (
        _chat_completions_url("https://proxy.example/custom", "https://example.invalid")
        == "https://proxy.example/custom"
    )
