"""Adaptive conversation memory implementation."""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from neva.utils.exceptions import MemoryConfigurationError

from .base import MemoryModule, MemoryRecord
from .budget import MemoryBudget
from .short_term import ShortTermMemory
from .summary import SummaryMemory
from .utils import cosine_similarity, estimate_token_count


class AdaptiveConversationMemory(MemoryModule):
    """Hierarchical memory system that balances fidelity with resource usage."""

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
            self._token_counts[record_id] = estimate_token_count(record.message)

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
            (record_id, record, cosine_similarity(vector, query_vector))
            for record_id, record, vector in self._semantic_entries
        ]

        scored = [item for item in scored if item[2] > 0]
        scored.sort(key=lambda item: (item[2], item[0]), reverse=True)

        top_records = [record for _, record, _score in scored[:size]]
        if not top_records:
            return ""

        return "\n".join(f"{record.speaker}: {record.message}" for record in top_records)


__all__ = ["AdaptiveConversationMemory"]
