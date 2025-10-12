"""Memory modules enabling agents to persist and recall dialogue context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from exceptions import MemoryConfigurationError


@dataclass
class MemoryRecord:
    """A single conversational memory entry."""

    speaker: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, object]] = None


class MemoryModule(ABC):
    """Abstract base class that defines the memory contract for agents."""

    def __init__(self, *, label: str = "memory") -> None:
        self.label = label

    @abstractmethod
    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        """Persist a conversational turn."""

    @abstractmethod
    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        """Return text describing the current memory state."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored memories."""


class ShortTermMemory(MemoryModule):
    """A bounded buffer that retains the most recent conversation turns."""

    def __init__(self, capacity: int = 6, *, label: str = "recent history") -> None:
        super().__init__(label=label)
        if capacity <= 0:
            raise MemoryConfigurationError("capacity must be a positive integer")
        self.capacity = capacity
        self._entries: Deque[MemoryRecord] = deque(maxlen=capacity)

    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        self._entries.append(
            MemoryRecord(speaker=speaker, message=message, metadata=metadata)
        )

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        if not self._entries:
            return ""

        records: Iterable[MemoryRecord]
        records = list(self._entries)
        if limit is not None:
            records = records[-limit:]

        if query:
            records = [r for r in records if query.lower() in r.message.lower()]
            if not records:
                return ""

        return "\n".join(f"{r.speaker}: {r.message}" for r in records)

    def clear(self) -> None:
        self._entries.clear()


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


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sqrt(sum(a * a for a in vec_a))
    mag_b = sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class VectorStoreMemory(MemoryModule):
    """Naïve in-memory vector store supporting similarity-based recall."""

    def __init__(
        self,
        embedder: Callable[[str], Sequence[float]],
        *,
        top_k: int = 3,
        label: str = "vector recall",
    ) -> None:
        super().__init__(label=label)
        if top_k <= 0:
            raise MemoryConfigurationError("top_k must be positive")
        self._embedder = embedder
        self._top_k = top_k
        self._vectors: List[Tuple[int, MemoryRecord, Tuple[float, ...]]] = []
        self._counter = 0

    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        text = f"{speaker}: {message}"
        vector = tuple(float(x) for x in self._embedder(text))
        record = MemoryRecord(speaker=speaker, message=message, metadata=metadata)
        self._vectors.append((self._counter, record, vector))
        self._counter += 1

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        if not self._vectors:
            return ""

        limit = limit or self._top_k
        limit = max(1, limit)

        if not query:
            records = [record for _, record, _ in self._vectors[-limit:]]
            return "\n".join(f"{r.speaker}: {r.message}" for r in records)

        query_vector = tuple(float(x) for x in self._embedder(query))
        scored = [
            (
                index,
                record,
                _cosine_similarity(vector, query_vector),
            )
            for index, record, vector in self._vectors
        ]
        scored.sort(key=lambda item: (item[2], item[0]), reverse=True)
        top_records = [record for _, record, score in scored[:limit] if score > 0]
        if not top_records:
            return ""
        return "\n".join(f"{r.speaker}: {r.message}" for r in top_records)

    def clear(self) -> None:
        self._vectors.clear()
        self._counter = 0


class CompositeMemory(MemoryModule):
    """Combine multiple memory modules and merge their recall outputs."""

    def __init__(self, modules: Sequence[MemoryModule], *, label: str = "composite") -> None:
        if not modules:
            raise MemoryConfigurationError("CompositeMemory requires at least one module")
        super().__init__(label=label)
        self._modules = list(modules)

    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        for module in self._modules:
            module.remember(speaker, message, metadata=metadata)

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        snippets = []
        for module in self._modules:
            snippet = module.recall(limit=limit, query=query)
            if snippet:
                snippets.append(f"{module.label}:\n{snippet}")
        return "\n\n".join(snippets)

    def clear(self) -> None:
        for module in self._modules:
            module.clear()

