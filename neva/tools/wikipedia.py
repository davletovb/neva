"""Wikipedia lookup tool implementation."""

from __future__ import annotations

import logging

from neva.agents.base import Tool
from neva.utils.exceptions import MissingDependencyError, ToolExecutionError

from .utils import missing_dependency_message

try:  # pragma: no cover - exercised when wikipedia is available.
    import wikipedia  # type: ignore
except Exception:  # pragma: no cover - fallback for optional dependency.
    wikipedia = None  # type: ignore


logger = logging.getLogger(__name__)


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
                missing_dependency_message(
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
