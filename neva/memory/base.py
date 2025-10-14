"""Base abstractions for memory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


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


__all__ = ["MemoryModule", "MemoryRecord"]
