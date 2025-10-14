"""Naïve in-memory vector store supporting similarity-based recall."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from neva.utils.exceptions import MemoryConfigurationError

from .base import MemoryModule, MemoryRecord
from .utils import cosine_similarity


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
                cosine_similarity(vector, query_vector),
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


__all__ = ["VectorStoreMemory"]
