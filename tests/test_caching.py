import pytest

from caching import LLMCache
from exceptions import CacheConfigurationError


def test_llm_cache_behaves_as_lru() -> None:
    cache = LLMCache(max_size=2)
    cache.set("a", "1")
    cache.set("b", "2")
    assert cache.get("a") == "1"

    cache.set("c", "3")
    assert cache.get("b") is None  # LRU entry evicted
    assert cache.get("a") == "1"
    assert cache.get("c") == "3"


def test_llm_cache_rejects_invalid_size() -> None:
    with pytest.raises(CacheConfigurationError):
        LLMCache(max_size=0)
