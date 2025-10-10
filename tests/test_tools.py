import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools import MathTool, SummarizerTool, TranslatorTool, WikipediaTool


def test_math_tool_evaluates_expression():
    tool = MathTool()
    assert tool.use("1 + 2 * 3") == "7.0"


def test_math_tool_rejects_invalid_expression():
    tool = MathTool()
    with pytest.raises(RuntimeError):
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
    if tool.__module__ != "tools":  # pragma: no cover - defensive guard
        pytest.skip("Unexpected module location")

    monkeypatch.setattr("tools.wikipedia", None)
    with pytest.raises(RuntimeError):
        tool.use("Python (programming language)")
