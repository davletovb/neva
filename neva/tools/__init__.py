"""Tool implementations used by Neva agents."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, Optional, Protocol, cast

import logging

from neva.agents.base import Tool
from neva.utils.exceptions import MissingDependencyError, ToolExecutionError

try:  # pragma: no cover - exercised when wikipedia is available.
    import wikipedia  # type: ignore
except Exception:  # pragma: no cover - fallback for optional dependency.
    wikipedia = None  # type: ignore


logger = logging.getLogger(__name__)


def _missing_dependency_message(package: str, *, fallback: str) -> str:
    return (
        f"The optional dependency `{package}` is unavailable. {fallback} "
        f"Install it with `pip install {package}` (or `pip install neva[tools]`) "
        "or provide a custom factory when constructing the tool to continue "
        "running experiments without network access."
    )


_ALLOWED_OPERATORS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


class TranslatorBackend(Protocol):
    """Protocol describing translator backends for :class:`TranslatorTool`."""

    def translate(self, text: str, *, target_language: str) -> str:
        """Translate ``text`` into ``target_language`` and return the result."""


class _CallableTranslatorWrapper:
    """Adapter turning callables into :class:`TranslatorBackend` instances."""

    def __init__(self, func: Callable[..., str]) -> None:
        self._func = func

    def translate(self, text: str, *, target_language: str) -> str:
        try:
            return self._func(text, target_language=target_language)
        except TypeError:
            try:
                return self._func(text, target_language)
            except TypeError:
                return self._func(text)


class _ObjectTranslatorWrapper:
    """Adapter for legacy translator objects following the googletrans API."""

    def __init__(self, backend: object, *, default_language: str) -> None:
        self._backend = backend
        self._default_language = default_language

    def translate(self, text: str, *, target_language: str) -> str:
        target = target_language or self._default_language
        translator = getattr(self._backend, "translate")
        try:
            result = translator(text, dest=target)
        except TypeError:
            try:
                result = translator(text, target_language=target)
            except TypeError:
                result = translator(text)
        return getattr(result, "text", str(result))


class _DeepTranslatorBackend:
    """Translation backend built on :mod:`deep_translator`."""

    def __init__(self, *, default_language: str) -> None:
        self._default_language = default_language

    def translate(self, text: str, *, target_language: str) -> str:
        from deep_translator import GoogleTranslator  # type: ignore

        target = target_language or self._default_language
        translator = GoogleTranslator(source="auto", target=target)
        return translator.translate(text)


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
                raise ToolExecutionError(
                    f"Unsupported operator: {op_type.__name__}"
                )
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _ALLOWED_OPERATORS[op_type](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ToolExecutionError("Unsupported expression")

    def use(self, task: str) -> str:
        try:
            expression = ast.parse(task, mode="eval")
            result = self._eval_node(expression.body)
            return str(result)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Failed to use MathTool: {exc}") from exc


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
            raise MissingDependencyError(
                _missing_dependency_message(
                    "wikipedia",
                    fallback="Wikipedia lookups require the `wikipedia` package.",
                )
            )

        try:
            return wikipedia.summary(task, sentences=self.summary_sentences)
        except Exception as exc:
            logger.warning("Wikipedia lookup failed: %s", exc)
            raise ToolExecutionError(
                "Failed to use WikipediaTool because the external service "
                f"returned an error: {exc}. Provide a stub via"
                " `summary_sentences` or monkeypatch the backend during tests."
            ) from exc


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

    def _wrap_translator(self, translator: object) -> TranslatorBackend:
        if callable(translator) and not hasattr(translator, "translate"):
            return _CallableTranslatorWrapper(cast(Callable[..., str], translator))
        if hasattr(translator, "translate"):
            return _ObjectTranslatorWrapper(translator, default_language=self.target_language)
        raise ToolExecutionError(
            "Translator factory must return a callable or object exposing a 'translate' method." 
        )

    def _get_translator(self) -> TranslatorBackend:
        if self._translator_factory is not None:
            translator = self._translator_factory()
            return self._wrap_translator(translator)

        try:  # pragma: no cover - depends on optional dependency.
            import deep_translator  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency missing.
            raise MissingDependencyError(
                _missing_dependency_message(
                    "deep-translator",
                    fallback="Translation requires the optional tools dependencies.",
                )
            ) from exc

        return _DeepTranslatorBackend(default_language=self.target_language)

    def use(self, task: str) -> str:
        try:
            translator = self._get_translator()
            return translator.translate(task, target_language=self.target_language)
        except Exception as exc:
            logger.warning("Translator backend failed: %s", exc)
            raise ToolExecutionError(
                "Failed to use TranslatorTool because the translation backend "
                f"raised an error: {exc}. Provide a stub translator factory for"
                " deterministic tests."
            ) from exc


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
            summarizer = self._summarizer_factory()
            return cast(Callable[[str], str], summarizer)

        try:  # pragma: no cover - optional dependency.
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyError(
                _missing_dependency_message(
                    "transformers",
                    fallback="Summarisation requires the optional tools dependencies.",
                )
            ) from exc

        summarizer = pipeline("summarization", model="t5-small")

        def _summarise(text: str, *, _summarizer=summarizer) -> str:
            results = _summarizer(text, truncation=True, max_length=150, min_length=40)
            if not results:
                return ""
            return results[0].get("summary_text", "").strip()

        return _summarise

    def use(self, task: str) -> str:
        try:
            summariser = self._get_summarizer()
            return summariser(task)
        except Exception as exc:
            logger.warning("Summariser backend failed: %s", exc)
            raise ToolExecutionError(
                "Failed to use SummarizerTool because the summarisation backend "
                f"raised an error: {exc}. Pass a lightweight factory during"
                " offline experimentation."
            ) from exc
