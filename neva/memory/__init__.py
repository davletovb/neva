"""Memory modules enabling agents to persist and recall dialogue context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from neva.utils.exceptions import MemoryConfigurationError


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


class FaissVectorStoreMemory(MemoryModule):
    """Long-term memory backed by a FAISS vector index for semantic recall."""

    def __init__(
        self,
        embedder: Callable[[str], Sequence[float]],
        *,
        top_k: int = 5,
        index_factory: str = "Flat",
        normalize_embeddings: bool = True,
        label: str = "faiss vector store",
    ) -> None:
        super().__init__(label=label)
        if top_k <= 0:
            raise MemoryConfigurationError("top_k must be positive")

        try:  # Lazy import to keep dependency optional.
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via tests when available.
            raise MemoryConfigurationError(
                "FaissVectorStoreMemory requires the 'faiss' package. "
                "Install it with `pip install faiss-cpu`."
            ) from exc

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - numpy is required alongside faiss.
            raise MemoryConfigurationError(
                "FaissVectorStoreMemory requires NumPy alongside faiss."
            ) from exc

        self._faiss = faiss
        self._np = np
        self._embedder = embedder
        self._top_k = top_k
        self._index_factory = index_factory
        self._normalize_embeddings = normalize_embeddings
        self._base_index: Optional[faiss.Index] = None  # type: ignore[attr-defined]
        self._index: Optional[faiss.Index] = None  # type: ignore[attr-defined]
        self._records: Dict[int, MemoryRecord] = {}
        self._order: List[int] = []
        self._id_counter = 0

    def _build_index(self, dimension: int) -> None:
        self._base_index = self._faiss.index_factory(dimension, self._index_factory)
        self._index = self._faiss.IndexIDMap(self._base_index)

    def _encode(self, text: str) -> "np.ndarray":
        vector = self._np.asarray(tuple(float(x) for x in self._embedder(text)), dtype="float32")
        if vector.ndim != 1:
            raise MemoryConfigurationError("Embeddings must be one-dimensional sequences of floats")
        if self._normalize_embeddings:
            self._faiss.normalize_L2(vector.reshape(1, -1))
            vector = vector.reshape(-1)
        return vector

    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        text = f"{speaker}: {message}"
        vector = self._encode(text)
        if self._index is None:
            self._build_index(vector.shape[0])
        assert self._index is not None and self._base_index is not None  # for mypy type checking
        record_id = self._id_counter
        self._id_counter += 1
        self._records[record_id] = MemoryRecord(
            speaker=speaker,
            message=message,
            metadata=metadata,
        )
        self._order.append(record_id)
        vector = vector.reshape(1, -1)
        if not self._base_index.is_trained:
            self._base_index.train(vector)
        ids = self._np.asarray([record_id], dtype="int64")
        self._index.add_with_ids(vector, ids)

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        if not self._records:
            return ""

        limit = limit or self._top_k
        limit = max(1, limit)

        if not query:
            record_ids = self._order[-limit:]
            records = [self._records[rid] for rid in record_ids]
            return "\n".join(f"{r.speaker}: {r.message}" for r in records)

        if self._index is None:
            return ""

        query_vector = self._encode(query).reshape(1, -1)
        k = min(limit, len(self._records))
        scores, indices = self._index.search(query_vector, k)
        results: List[Tuple[float, int, MemoryRecord]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self._records.get(int(idx))
            if record is None:
                continue
            results.append((float(score), int(idx), record))

        if not results:
            return ""

        results.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        top_records = [record for _, _, record in results[:k]]
        return "\n".join(f"{r.speaker}: {r.message}" for r in top_records)

    def clear(self) -> None:
        if self._index is not None:
            self._index.reset()
        self._records.clear()
        self._order.clear()
        self._id_counter = 0


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

