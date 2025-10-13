"""Memory modules enabling agents to persist and recall dialogue context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

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


def _estimate_token_count(text: str) -> int:
    """Rudimentary token estimator used for budgeting heuristics."""

    if not text:
        return 0
    return max(1, len(text.split()))


class MemoryBudget:
    """Resource guard that bounds memory growth and embedding compute cost."""

    def __init__(
        self,
        *,
        max_records: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_embeddings: Optional[int] = None,
        token_estimator: Callable[[str], int] = _estimate_token_count,
    ) -> None:
        if max_records is not None and max_records <= 0:
            raise MemoryConfigurationError("max_records must be positive when provided")
        if max_tokens is not None and max_tokens <= 0:
            raise MemoryConfigurationError("max_tokens must be positive when provided")
        if max_embeddings is not None and max_embeddings <= 0:
            raise MemoryConfigurationError("max_embeddings must be positive when provided")

        self.max_records = max_records
        self.max_tokens = max_tokens
        self.max_embeddings = max_embeddings
        self._token_estimator = token_estimator
        self._embedding_calls = 0

    def token_count(self, message: str) -> int:
        """Return the estimated token count for ``message``."""

        return self._token_estimator(message)

    def trim_history(
        self,
        history: List[Tuple[int, MemoryRecord]],
        token_counts: Dict[int, int],
    ) -> List[int]:
        """Apply configured limits to ``history`` in-place.

        Parameters
        ----------
        history:
            Chronological list of ``(record_id, MemoryRecord)`` entries. Oldest
            entries are trimmed first.
        token_counts:
            Mapping of ``record_id`` to the cached token estimation used for the
            ``max_tokens`` constraint.

        Returns
        -------
        List[int]
            Identifiers for the records that were removed.
        """

        removed: List[int] = []

        if self.max_records is not None:
            while len(history) > self.max_records:
                record_id, _ = history.pop(0)
                removed.append(record_id)
                token_counts.pop(record_id, None)

        if self.max_tokens is not None:
            def total_tokens() -> int:
                return sum(token_counts[record_id] for record_id, _ in history)

            while history and total_tokens() > self.max_tokens:
                record_id, _ = history.pop(0)
                removed.append(record_id)
                token_counts.pop(record_id, None)

        return removed

    def can_embed(self) -> bool:
        """Return ``True`` when another embedding call is permitted."""

        return self.max_embeddings is None or self._embedding_calls < self.max_embeddings

    def note_embedding(self) -> None:
        """Record that an embedding has been computed."""

        if self.max_embeddings is None:
            return
        self._embedding_calls += 1

    def reset(self) -> None:
        """Reset tracked compute usage counters."""

        self._embedding_calls = 0


class AdaptiveConversationMemory(MemoryModule):
    """Hierarchical memory system that balances fidelity with resource usage.

    The adaptive memory keeps a complete chronological history for exact replay
    while exposing three specialised views:

    - A short-term buffer that retains the most recent turns verbatim.
    - A running summary that condenses the full dialogue.
    - An optional semantic index for similarity search over the backlog.

    A :class:`MemoryBudget` may be supplied to cap the amount of history stored
    and the number of embeddings generated, enabling simulation-scale
    deployments without unbounded growth.
    """

    def __init__(
        self,
        *,
        summarizer: Callable[[str, MemoryRecord], str],
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        short_term_capacity: int = 6,
        semantic_top_k: int = 3,
        initial_summary: str = "",
        budget: Optional[MemoryBudget] = None,
        label: str = "adaptive memory",
    ) -> None:
        super().__init__(label=label)
        if short_term_capacity <= 0:
            raise MemoryConfigurationError("short_term_capacity must be positive")
        if semantic_top_k <= 0:
            raise MemoryConfigurationError("semantic_top_k must be positive")

        self._summarizer = summarizer
        self._embedder = embedder
        self._short_term_capacity = short_term_capacity
        self._semantic_top_k = semantic_top_k
        self._initial_summary = initial_summary.strip()
        self._budget = budget

        self._history: List[Tuple[int, MemoryRecord]] = []
        self._token_counts: Dict[int, int] = {}
        self._semantic_entries: List[Tuple[int, MemoryRecord, Tuple[float, ...]]] = []
        self._vector_cache: Dict[int, Tuple[float, ...]] = {}
        self._id_counter = 0

        self._short_term_label = "recent history"
        self._summary_label = "summary"

        self._short_term = ShortTermMemory(
            capacity=short_term_capacity, label=self._short_term_label
        )
        self._summary = SummaryMemory(
            summarizer,
            initial_summary=initial_summary,
            label=self._summary_label,
        )

    def remember(
        self, speaker: str, message: str, *, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        record = MemoryRecord(speaker=speaker, message=message, metadata=metadata)
        record_id = self._id_counter
        self._id_counter += 1

        self._history.append((record_id, record))

        if self._budget is not None:
            self._token_counts[record_id] = self._budget.token_count(record.message)
        else:
            self._token_counts[record_id] = _estimate_token_count(record.message)

        self._short_term.remember(speaker, message, metadata=metadata)
        self._summary.remember(speaker, message, metadata=metadata)

        self._maybe_store_semantic(record_id, record)

        removed_ids = self._enforce_budget()
        if removed_ids:
            self._rebuild_views()

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        sections: List[Tuple[str, str]] = []

        if query:
            recent = self._short_term.recall(limit=limit, query=query)
            if recent:
                sections.append(("recent", recent))

            semantic = self._semantic_search(query, limit)
            if semantic:
                sections.append(("semantic", semantic))

            summary = self._summary.recall(query=query)
            if summary:
                sections.append(("summary", summary))
        else:
            recent = self._short_term.recall(limit=limit)
            if recent:
                sections.append(("recent", recent))

            summary = self._summary.recall()
            if summary:
                sections.append(("summary", summary))

        unique_sections = []
        seen_snippets: Set[str] = set()
        for label, text in sections:
            if text and text not in seen_snippets:
                unique_sections.append((label, text))
                seen_snippets.add(text)

        if not unique_sections:
            return ""

        return "\n\n".join(f"{label}:\n{text}" for label, text in unique_sections)

    def clear(self) -> None:
        self._history.clear()
        self._token_counts.clear()
        self._semantic_entries.clear()
        self._vector_cache.clear()
        self._id_counter = 0
        self._short_term = ShortTermMemory(
            capacity=self._short_term_capacity, label=self._short_term_label
        )
        self._summary = SummaryMemory(
            self._summarizer,
            initial_summary=self._initial_summary,
            label=self._summary_label,
        )
        if self._budget is not None:
            self._budget.reset()

    def iter_history(self, *, speaker: Optional[str] = None) -> Iterator[MemoryRecord]:
        """Yield historical records optionally filtered by ``speaker``."""

        for _, record in self._history:
            if speaker is None or record.speaker == speaker:
                yield record

    def recent_window(
        self, size: int, *, speaker: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Return the most recent ``size`` records, optionally filtered."""

        if size <= 0:
            return []
        filtered = list(self.iter_history(speaker=speaker))
        if not filtered:
            return []
        return filtered[-size:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _maybe_store_semantic(
        self, record_id: int, record: MemoryRecord
    ) -> None:
        if self._embedder is None:
            return

        if record_id in self._vector_cache:
            vector = self._vector_cache[record_id]
        else:
            if self._budget is not None and not self._budget.can_embed():
                return
            text = f"{record.speaker}: {record.message}"
            vector = tuple(float(x) for x in self._embedder(text))
            self._vector_cache[record_id] = vector
            if self._budget is not None:
                self._budget.note_embedding()

        self._semantic_entries.append((record_id, record, vector))

    def _enforce_budget(self) -> List[int]:
        if self._budget is None:
            return []

        removed_ids = self._budget.trim_history(self._history, self._token_counts)
        if not removed_ids:
            return []

        for record_id in removed_ids:
            self._vector_cache.pop(record_id, None)

        self._semantic_entries = [
            (record_id, record, vector)
            for record_id, record, vector in self._semantic_entries
            if record_id not in removed_ids
        ]

        return removed_ids

    def _rebuild_views(self) -> None:
        self._short_term = ShortTermMemory(
            capacity=self._short_term_capacity, label=self._short_term_label
        )
        self._summary = SummaryMemory(
            self._summarizer,
            initial_summary=self._initial_summary,
            label=self._summary_label,
        )

        for record_id, record in self._history:
            self._short_term.remember(record.speaker, record.message, metadata=record.metadata)
            self._summary.remember(record.speaker, record.message, metadata=record.metadata)

        self._semantic_entries = []
        for record_id, record in self._history:
            vector = self._vector_cache.get(record_id)
            if vector is None:
                continue
            self._semantic_entries.append((record_id, record, vector))

    def _semantic_search(self, query: str, limit: Optional[int]) -> str:
        if self._embedder is None or not self._semantic_entries:
            return ""

        query_vector = tuple(float(x) for x in self._embedder(query))
        size = limit or self._semantic_top_k
        size = max(1, size)

        scored = [
            (record_id, record, _cosine_similarity(vector, query_vector))
            for record_id, record, vector in self._semantic_entries
        ]

        scored = [item for item in scored if item[2] > 0]
        scored.sort(key=lambda item: (item[2], item[0]), reverse=True)

        top_records = [record for _, record, _score in scored[:size]]
        if not top_records:
            return ""

        return "\n".join(f"{record.speaker}: {record.message}" for record in top_records)


