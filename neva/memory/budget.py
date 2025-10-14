"""Memory budgeting utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from neva.utils.exceptions import MemoryConfigurationError

from .base import MemoryRecord
from .utils import estimate_token_count


class MemoryBudget:
    """Resource guard that bounds memory growth and embedding compute cost."""

    def __init__(
        self,
        *,
        max_records: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_embeddings: Optional[int] = None,
        token_estimator: Callable[[str], int] = estimate_token_count,
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
        """Apply configured limits to ``history`` in-place."""

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


__all__ = ["MemoryBudget"]
