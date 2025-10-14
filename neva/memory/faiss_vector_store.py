"""FAISS-backed vector memory implementation."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from neva.memory.base import MemoryModule, MemoryRecord
from neva.utils.exceptions import MemoryConfigurationError


if TYPE_CHECKING:  # pragma: no cover - imported for type checking only.
    import numpy as np


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


__all__ = ["FaissVectorStoreMemory"]
