"""Translation tool implementation."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol, cast

from neva.agents.base import Tool
from neva.utils.exceptions import MissingDependencyError, ToolExecutionError

from .utils import missing_dependency_message

logger = logging.getLogger(__name__)


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
                missing_dependency_message(
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
