"""Caching helpers for LLM responses to reduce latency and cost."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Optional

from neva.utils.exceptions import CacheConfigurationError


class LLMCache:
    """A simple thread-safe LRU cache tailored to prompt/response pairs."""

    def __init__(self, max_size: int = 128) -> None:
        if max_size <= 0:
            raise CacheConfigurationError("max_size must be a positive integer")
        self._max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()
        self._lock = RLock()

    def get(self, prompt: str) -> Optional[str]:
        with self._lock:
            if prompt not in self._store:
                return None
            value = self._store.pop(prompt)
            self._store[prompt] = value
            return value

    def set(self, prompt: str, response: str) -> None:
        with self._lock:
            if prompt in self._store:
                self._store.pop(prompt)
            self._store[prompt] = response
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
