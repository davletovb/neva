import pytest

from neva.agents import TransformerAgent
from neva.agents.base import ToolCall
from neva.tools import MathTool, SummarizerTool, TranslatorTool, WikipediaTool


def make_agent():
    return TransformerAgent(name="agent", llm_backend=lambda prompt: "ok")


@pytest.mark.parametrize(
    "tool",
    [
        MathTool(),
        WikipediaTool(summary_sentences=1),
        SummarizerTool(summarizer_factory=lambda: lambda text: "summary"),
        TranslatorTool(translator_factory=lambda: lambda text: "translated"),
    ],
)
def test_builtins_declare_argument_schema(tool):
    assert tool.argument_schema is not None


@pytest.mark.parametrize(
    "tool",
    [
        MathTool(),
        WikipediaTool(summary_sentences=1),
        SummarizerTool(summarizer_factory=lambda: lambda text: "summary"),
        TranslatorTool(translator_factory=lambda: lambda text: "translated"),
    ],
)
@pytest.mark.parametrize("alias", ["input", "task", "query", "text"])
def test_builtin_schema_preserves_supported_text_aliases(tool, alias):
    assert tool.argument_schema.validate({alias: "value"}) is None


@pytest.mark.parametrize(
    "tool",
    [
        MathTool(),
        WikipediaTool(summary_sentences=1),
        SummarizerTool(summarizer_factory=lambda: lambda text: "summary"),
        TranslatorTool(translator_factory=lambda: lambda text: "translated"),
    ],
)
def test_builtin_schema_rejects_payloads_that_cannot_normalise_to_text(tool):
    assert tool.argument_schema.validate({"input": 123}) is not None
    assert tool.argument_schema.validate({"left": "a", "right": "b"}) is not None


@pytest.mark.parametrize(
    "tool",
    [
        MathTool(),
        WikipediaTool(summary_sentences=1),
        SummarizerTool(summarizer_factory=lambda: lambda text: "summary"),
        TranslatorTool(translator_factory=lambda: lambda text: "translated"),
    ],
)
def test_builtin_schema_preserves_single_string_mapping_and_metadata(tool):
    assert tool.argument_schema.validate({"payload": "value"}) is None
    assert tool.argument_schema.validate({"input": "value", "source": "unit-test"}) is None


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


def test_builtin_schema_keeps_raw_string_call_compatible():
    tool = MathTool()
    agent = make_agent()
    agent.register_tool(tool)

    response = agent.call_tool(ToolCall(name="calculator", arguments="1 + 2"))

    assert response.succeeded()
    assert response.output == "3.0"
