"""Text summarisation tool implementation."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from neva.agents.base import Tool
from neva.utils.exceptions import MissingDependencyError, ToolExecutionError

from .utils import missing_dependency_message

logger = logging.getLogger(__name__)


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
            return summarizer

        try:  # pragma: no cover - optional dependency.
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyError(
                missing_dependency_message(
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
