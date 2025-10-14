"""Conversation summary memory implementation."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .base import MemoryModule, MemoryRecord


class SummaryMemory(MemoryModule):
    """Memory that maintains a running summary of the conversation."""

    def __init__(
        self,
        summarizer: Callable[[str, MemoryRecord], str],
        *,
        initial_summary: str = "",
        label: str = "summary",
    ) -> None:
        super().__init__(label=label)
        self._summarizer = summarizer
        self._summary = initial_summary.strip()
        self._history: List[MemoryRecord] = []

    @property
    def summary(self) -> str:
        return self._summary

    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        record = MemoryRecord(speaker=speaker, message=message, metadata=metadata)
        self._history.append(record)
        self._summary = self._summarizer(self._summary, record).strip()

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        if not self._summary:
            return ""
        if query:
            if query.lower() not in self._summary.lower():
                return ""
        return self._summary

    def clear(self) -> None:
        self._summary = ""
        self._history.clear()


__all__ = ["SummaryMemory"]
