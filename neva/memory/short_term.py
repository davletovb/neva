"""Short-term memory implementation."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Optional

from neva.utils.exceptions import MemoryConfigurationError

from .base import MemoryModule, MemoryRecord


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
        self._entries.append(MemoryRecord(speaker=speaker, message=message, metadata=metadata))

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


__all__ = ["ShortTermMemory"]
