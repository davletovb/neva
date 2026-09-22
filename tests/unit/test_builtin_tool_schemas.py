import json
from types import SimpleNamespace

import pytest

from neva.agents import TransformerAgent
from neva.agents.base import TOOL_TEXT_ARGUMENT_KEYS, ToolCall
from neva.tools import MathTool, SummarizerTool, TranslatorTool, WikipediaTool


def make_agent():
    return TransformerAgent(name="agent", llm_backend=lambda prompt: "ok")


def builtin_tools():
    return [
        MathTool(),
        WikipediaTool(summary_sentences=1),
        SummarizerTool(summarizer_factory=lambda: lambda text: "summary"),
        TranslatorTool(translator_factory=lambda: lambda text: "translated"),
    ]


@pytest.mark.parametrize("tool", builtin_tools())
def test_builtins_declare_argument_schema(tool):
    assert tool.argument_schema is not None


@pytest.mark.parametrize("tool", builtin_tools())
@pytest.mark.parametrize("alias", TOOL_TEXT_ARGUMENT_KEYS)
def test_builtin_schema_preserves_supported_text_aliases(tool, alias):
    assert tool.argument_schema.validate({alias: "value"}) is None


@pytest.mark.parametrize(
    ("arguments", "accepted"),
    [
        ({}, False),
        ({"input": "value"}, True),
        ({"input": ""}, True),
        ({"task": "value"}, True),
        ({"query": "value"}, True),
        ({"text": "value"}, True),
        ({"payload": "value"}, True),
        ({"payload": 1}, False),
        ({"input": 123}, False),
        ({"input": 123, "task": "value"}, True),
        ({"input": 123, "task": 456}, False),
        ({"input": "value", "source": "unit-test"}, True),
        ({"source": "unit-test", "payload": "value"}, False),
        ({"left": "a", "right": "b"}, False),
        ({"query": "first", "text": "second"}, True),
        ({"input": None, "text": "value"}, True),
    ],
)
def test_builtin_schema_matches_tool_input_normalizer(arguments, accepted):
    agent = make_agent()
    schema = MathTool().argument_schema

    normalised = agent._normalise_tool_input(arguments)
    used_json_fallback = normalised == json.dumps(arguments, sort_keys=True)

    assert (schema.validate(arguments) is None) is accepted
    assert accepted is not used_json_fallback


@pytest.mark.parametrize("tool", builtin_tools())
def test_builtin_schema_rejects_payloads_that_cannot_normalise_to_text(tool):
    assert tool.argument_schema.validate({}) is not None
    assert tool.argument_schema.validate({"input": 123}) is not None
    assert tool.argument_schema.validate({"payload": 123}) is not None
    assert tool.argument_schema.validate({"left": "a", "right": "b"}) is not None


@pytest.mark.parametrize("tool", builtin_tools())
def test_builtin_schema_preserves_single_string_mapping_and_metadata(tool):
    assert tool.argument_schema.validate({"payload": "value"}) is None
    assert tool.argument_schema.validate({"input": "value", "source": "unit-test"}) is None


def test_builtin_schema_rejects_non_mapping_direct_validation():
    reason = MathTool().argument_schema.validate(["not", "a", "mapping"])
    assert reason is not None
    assert "mapping" in reason


def test_builtin_schema_rejects_invalid_call_before_tool_execution():
    calls = []

    def factory():
        calls.append("factory-called")
        return lambda text: "summary"

    tool = SummarizerTool(summarizer_factory=factory)
    agent = make_agent()
    agent.register_tool(tool)

    response = agent.call_tool(
        ToolCall(
            name="summarizer",
            arguments={"left": "ambiguous", "right": "should not execute"},
        )
    )

    assert not response.succeeded()
    assert "invalid arguments" in response.error
    assert calls == []


def test_builtin_call_tool_accepts_supported_payloads(monkeypatch):
    import neva.tools.wikipedia as wikipedia_module

    monkeypatch.setattr(
        wikipedia_module,
        "wikipedia",
        SimpleNamespace(summary=lambda task, sentences: f"wiki:{task}:{sentences}"),
    )

    cases = [
        (MathTool(), ToolCall(name="calculator", arguments="1 + 2"), "3.0"),
        (
            WikipediaTool(summary_sentences=1),
            ToolCall(name="wikipedia", arguments={"query": "Mars"}),
            "wiki:Mars:1",
        ),
        (
            SummarizerTool(summarizer_factory=lambda: lambda text: f"summary:{text}"),
            ToolCall(name="summarizer", arguments={"text": "long text"}),
            "summary:long text",
        ),
        (
            TranslatorTool(translator_factory=lambda: lambda text: f"translated:{text}"),
            ToolCall(
                name="translator",
                arguments={"input": "hello", "source": "unit-test"},
            ),
            "translated:hello",
        ),
    ]

    for tool, call, expected in cases:
        agent = make_agent()
        agent.register_tool(tool)
        response = agent.call_tool(call)
        assert response.succeeded(), response.error
        assert response.output == expected
