"""FAISS-backed vector memory implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from neva.memory.base import MemoryModule, MemoryRecord
from neva.utils.exceptions import MemoryConfigurationError

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only.
    import faiss
    import numpy as np


class FaissVectorStoreMemory(MemoryModule):
    """Long-term memory backed by a FAISS vector index for semantic recall.

    This memory module uses FAISS (Facebook AI Similarity Search) to store
    and retrieve memories based on semantic similarity using vector embeddings.

    Args:
        embedder: A callable that converts text to embedding vectors.
        top_k: Number of top results to return in recall (default: 5).
        index_factory: FAISS index type specification (default: "Flat").
        normalize_embeddings: Whether to L2-normalize embeddings (default: True).
        label: Human-readable label for this memory module.

    Raises:
        MemoryConfigurationError: If dependencies are missing or config is invalid.
    """

    def __init__(
        self,
        embedder: Callable[[str], Sequence[float]],
        *,
        top_k: int = 5,
        index_factory: str = "Flat",
        normalize_embeddings: bool = True,
        label: str = "faiss vector store",
    ) -> None:
        """Initialize the FAISS vector store memory."""
        super().__init__(label=label)
        if top_k <= 0:
            raise MemoryConfigurationError("top_k must be positive")

        # Lazy import to keep dependencies optional
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as exc:
            raise MemoryConfigurationError(
                "FaissVectorStoreMemory requires the 'faiss' package. "
                "Install it with `pip install faiss-cpu`."
            ) from exc

        try:
            import numpy as np
        except ImportError as exc:
            raise MemoryConfigurationError(
                "FaissVectorStoreMemory requires NumPy alongside faiss."
            ) from exc

        self._faiss: faiss = faiss  # type: ignore[assignment]
        self._np: np = np  # type: ignore[assignment]
        self._embedder = embedder
        self._top_k = top_k
        self._index_factory = index_factory
        self._normalize_embeddings = normalize_embeddings
        self._base_index: Optional[faiss.Index] = None
        self._index: Optional[faiss.Index] = None
        self._records: Dict[int, MemoryRecord] = {}
        self._order: List[int] = []
        self._id_counter = 0

    def _build_index(self, dimension: int) -> None:
        """Build a new FAISS index with the specified dimension.

        Args:
            dimension: The dimensionality of the embedding vectors.
        """
        self._base_index = self._faiss.index_factory(
            dimension, self._index_factory
        )
        self._index = self._faiss.IndexIDMap(self._base_index)

    def _encode(self, text: str) -> np.ndarray:  # type: ignore[name-defined]
        """Encode text into a normalized embedding vector.

        Args:
            text: The text to encode.

        Returns:
            A numpy array containing the embedding vector.

        Raises:
            MemoryConfigurationError: If embeddings are not 1D sequences.
        """
        embedding = self._embedder(text)
        vector = self._np.asarray(
            tuple(float(x) for x in embedding), dtype="float32"
        )

        if vector.ndim != 1:
            msg = (
                "Embeddings must be one-dimensional sequences of floats"
            )
            raise MemoryConfigurationError(msg)

        if self._normalize_embeddings:
            # Normalize in-place for efficiency
            vector_2d = vector.reshape(1, -1)
            self._faiss.normalize_L2(vector_2d)
            vector = vector_2d.reshape(-1)

        return vector

    def remember(
        self,
        speaker: str,
        message: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Store a new memory in the vector store.

        Args:
            speaker: The speaker/author of the message.
            message: The message content to remember.
            metadata: Optional additional metadata to store with the memory.
        """
        text = f"{speaker}: {message}"
        vector = self._encode(text)

        # Initialize index on first memory
        if self._index is None:
            self._build_index(vector.shape[0])

        assert self._index is not None
        assert self._base_index is not None

        # Generate unique ID and store record
        record_id = self._id_counter
        self._id_counter += 1

        self._records[record_id] = MemoryRecord(
            speaker=speaker,
            message=message,
            metadata=metadata,
        )
        self._order.append(record_id)

        # Add to FAISS index
        vector_2d = vector.reshape(1, -1)
        if not self._base_index.is_trained:
            self._base_index.train(vector_2d)

        ids = self._np.asarray([record_id], dtype="int64")
        self._index.add_with_ids(vector_2d, ids)

    def recall(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> str:
        """Retrieve memories, optionally filtered by semantic similarity.

        Args:
            limit: Maximum number of memories to return (default: self._top_k).
            query: Optional query text for semantic search. If None, returns
                   the most recent memories.

        Returns:
            A formatted string containing the recalled memories, one per line.
        """
        if not self._records:
            return ""

        limit = limit or self._top_k
        limit = max(1, limit)

        # Return recent memories if no query provided
        if not query:
            record_ids = self._order[-limit:]
            records = [self._records[rid] for rid in record_ids]
            return "\n".join(
                f"{r.speaker}: {r.message}" for r in records
            )

        # Semantic search requires index
        if self._index is None:
            return ""

        # Perform vector similarity search
        query_vector = self._encode(query).reshape(1, -1)
        k = min(limit, len(self._records))
        scores, indices = self._index.search(query_vector, k)

        # Collect valid results
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

        # Sort by score (desc) then by recency (desc)
        results.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        top_records = [record for _, _, record in results[:k]]

        return "\n".join(
            f"{r.speaker}: {r.message}" for r in top_records
        )

    def clear(self) -> None:
        """Clear all stored memories and reset the index."""
        if self._index is not None:
            self._index.reset()
        self._records.clear()
        self._order.clear()
        self._id_counter = 0


__all__ = ["FaissVectorStoreMemory"]
