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
