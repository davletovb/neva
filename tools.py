"""Tool implementations used by Neva agents."""

from __future__ import annotations

import ast
import operator
from typing import Callable, Dict, Optional

import logging

from models import Tool

try:  # pragma: no cover - exercised when wikipedia is available.
    import wikipedia  # type: ignore
except Exception:  # pragma: no cover - fallback for optional dependency.
    wikipedia = None  # type: ignore


logger = logging.getLogger(__name__)


_ALLOWED_OPERATORS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


class MathTool(Tool):
    """Safely evaluate basic mathematical expressions."""

    def __init__(self) -> None:
        super().__init__(
            "calculator",
            "performs arithmetic expressions",
            capabilities=["math", "calculation"],
        )

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Num):  # pragma: no cover - python<3.8 fallback
            return float(node.n)  # type: ignore[attr-defined]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _ALLOWED_OPERATORS[op_type](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("Unsupported expression")

    def use(self, task: str) -> str:
        try:
            expression = ast.parse(task, mode="eval")
            result = self._eval_node(expression.body)
            return str(result)
        except Exception as exc:
            raise RuntimeError(f"Failed to use MathTool: {exc}")


class WikipediaTool(Tool):
    """Retrieve short summaries from Wikipedia."""

    def __init__(self, *, summary_sentences: int = 2) -> None:
        super().__init__(
            "wikipedia",
            "looks up concise encyclopedia summaries",
            capabilities=["search", "encyclopedia"],
        )
        self.summary_sentences = summary_sentences

    def use(self, task: str) -> str:
        if wikipedia is None:
            raise RuntimeError("The `wikipedia` package is not installed.")

        try:
            return wikipedia.summary(task, sentences=self.summary_sentences)
        except Exception as exc:
            raise RuntimeError(f"Failed to use WikipediaTool: {exc}")


class TranslatorTool(Tool):
    """Translate text using a configurable backend."""

    def __init__(
        self,
        *,
        translator_factory: Optional[Callable[[], object]] = None,
        target_language: str = "en",
    ) -> None:
        super().__init__(
            "translator",
            "translates text to different languages",
            capabilities=["translation"],
        )
        self._translator_factory = translator_factory
        self.target_language = target_language

    def _get_translator(self):
        if self._translator_factory is not None:
            return self._translator_factory()

        try:  # pragma: no cover - depends on optional dependency.
            from googletrans import Translator  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency missing.
            raise RuntimeError(
                "googletrans is not installed. Provide `translator_factory` to "
                "TranslatorTool for offline testing."
            ) from exc

        return Translator()

    def use(self, task: str) -> str:
        try:
            translator = self._get_translator()
            result = translator.translate(task, dest=self.target_language)
            return getattr(result, "text", str(result))
        except Exception as exc:
            raise RuntimeError(f"Failed to use TranslatorTool: {exc}")


class SummarizerTool(Tool):
    """Summarise text using a pluggable backend."""

    def __init__(
        self,
        *,
        summarizer_factory: Optional[Callable[[], Callable[[str], str]]] = None,
    ) -> None:
        super().__init__(
            "summarizer",
            "creates short summaries of longer documents",
            capabilities=["summarisation"],
        )
        self._summarizer_factory = summarizer_factory

    def _get_summarizer(self) -> Callable[[str], str]:
        if self._summarizer_factory is not None:
            return self._summarizer_factory()

        try:  # pragma: no cover - optional dependency.
            from summarizer import Summarizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "The `summarizer` package is not installed. Provide "
                "`summarizer_factory` when constructing SummarizerTool."
            ) from exc

        model = Summarizer()

        def _summarise(text: str, *, _model=model) -> str:
            return _model(text, min_length=60, max_length=100)

        return _summarise

    def use(self, task: str) -> str:
        try:
            summariser = self._get_summarizer()
            return summariser(task)
        except Exception as exc:
            raise RuntimeError(f"Failed to use SummarizerTool: {exc}")
