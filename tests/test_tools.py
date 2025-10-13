from types import SimpleNamespace

import pytest

from neva.tools import MathTool, SummarizerTool, TranslatorTool, WikipediaTool
from neva.utils.exceptions import MissingDependencyError, ToolExecutionError


def test_math_tool_evaluates_expression():
    tool = MathTool()
    assert tool.use("1 + 2 * 3") == "7.0"


def test_math_tool_rejects_invalid_expression():
    tool = MathTool()
    with pytest.raises(ToolExecutionError):
        tool.use("__import__('os').system('ls')")


def test_translator_tool_uses_injected_backend():
    def factory():
        return SimpleNamespace(translate=lambda text, dest=None: SimpleNamespace(text=f"{text}-{dest}"))

    tool = TranslatorTool(translator_factory=factory, target_language="fr")
    assert tool.use("hello") == "hello-fr"


def test_summarizer_tool_uses_injected_backend():
    tool = SummarizerTool(summarizer_factory=lambda: (lambda text: "summary"))
    assert tool.use("long text") == "summary"


def test_wikipedia_tool_missing_dependency_raises(monkeypatch):
    tool = WikipediaTool(summary_sentences=1)
    if tool.__module__ != "neva.tools":  # pragma: no cover - defensive guard
        pytest.skip("Unexpected module location")

    monkeypatch.setattr("neva.tools.wikipedia", None)
    with pytest.raises(MissingDependencyError) as exc:
        tool.use("Python (programming language)")

    assert "pip install wikipedia" in str(exc.value)
