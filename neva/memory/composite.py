"""Composite memory implementation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from neva.utils.exceptions import MemoryConfigurationError

from .base import MemoryModule


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


__all__ = ["CompositeMemory"]
