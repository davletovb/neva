"""Keep the live provider example opt-in without using external credentials."""

import pytest
import requests

from examples import live_provider_smoke as example


def test_default_is_scripted_and_never_calls_provider(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-unused")

    def no_network(*args, **kwargs):
        raise AssertionError("offline example attempted a provider call")

    monkeypatch.setattr("neva.agents.gpt.requests.post", no_network)
    assert example.main([]) == 0
    output = capsys.readouterr().out
    assert "[SCRIPTED]" in output
    assert "Split the task" in output
    assert "[GENERATED]" not in output


@pytest.mark.parametrize(
    "args, message",
    [
        (["--live"], "--live requires --max-spend-usd"),
        (["--max-spend-usd", "0.01"], "--max-spend-usd requires --live"),
        (["--live", "--max-spend-usd", "0"], "positive finite"),
        (["--live", "--max-spend-usd", "nan"], "positive finite"),
        (["--live", "--max-spend-usd", "inf"], "positive finite"),
    ],
)
def test_invalid_live_configuration_never_calls_provider(monkeypatch, capsys, args, message):
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: pytest.fail("unexpected provider call"),
    )
    with pytest.raises(SystemExit) as exc:
        example.main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_live_requires_credentials_before_provider_call(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: pytest.fail("unexpected provider call"),
    )
    with pytest.raises(SystemExit) as exc:
        example.main(["--live", "--max-spend-usd", "0.01"])
    assert exc.value.code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_live_uses_bounded_builtin_provider_and_reports_estimate(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "example-test-secret")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "Check sources before summarizing."}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 6},
            }

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    assert example.main(["--live", "--max-spend-usd", "0.01"]) == 0
    assert len(calls) == 1
    url, headers, payload, timeout = calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer example-test-secret"
    assert payload["model"] == example.MODEL
    assert payload["max_tokens"] == example.MAX_OUTPUT_TOKENS
    assert timeout == example.REQUEST_TIMEOUT_SECONDS
    output = capsys.readouterr().out
    assert "[GENERATED]" in output
    assert "Check sources before summarizing." in output
    assert "provider usage" in output
    assert "example-test-secret" not in output


def test_insufficient_estimated_budget_prevents_request(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "example-test-secret")
    monkeypatch.setattr(
        "neva.agents.gpt.requests.post",
        lambda *args, **kwargs: pytest.fail("unexpected provider call"),
    )
    with pytest.raises(SystemExit) as exc:
        example.main(["--live", "--max-spend-usd", "0.000001"])
    assert exc.value.code == 2
    assert "estimated prompt + maximum output cost" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["unauthorized", "timeout"])
def test_live_provider_failure_reports_hint_and_recorded_spend(monkeypatch, capsys, failure):
    monkeypatch.setenv("OPENAI_API_KEY", "example-test-secret")
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        if failure == "timeout":
            raise requests.Timeout("test request timed out")
        response = requests.Response()
        response.status_code = 401
        response.url = "https://api.openai.com/v1/chat/completions"
        return response

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    assert example.main(["--live", "--max-spend-usd", "0.01"]) == 1
    assert len(calls) == 1
    output = capsys.readouterr()
    assert "Live request failed" in output.err
    assert "OPENAI_API_KEY" in output.err
    assert "Recorded estimated spend so far: $0.000000" in output.err
    assert "Provider billing may still include" in output.err
    assert "example-test-secret" not in output.out + output.err
    assert "Traceback" not in output.err
