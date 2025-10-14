import sys
import types

import pytest

from neva.agents import GPTAgent, TransformerAgent
from neva.utils.caching import LLMCache
from neva.utils.exceptions import CacheConfigurationError

# Ensure optional dependencies used by models are stubbed for the test run.
openai_stub = types.ModuleType("openai")
sys.modules.setdefault("openai", openai_stub)

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


def test_gpt_agent_reuses_cache_for_repeated_prompts() -> None:
    call_counter = {"count": 0}

    def backend(prompt: str) -> str:
        call_counter["count"] += 1
        return f"echo:{prompt}"

    agent = GPTAgent(llm_backend=backend, cache=LLMCache(max_size=8))

    first = agent.respond("Hello")
    second = agent.respond("Hello")

    assert first == second
    assert call_counter["count"] == 1


def test_transformer_agent_reuses_cache_for_repeated_prompts() -> None:
    call_counter = {"count": 0}

    def backend(prompt: str) -> str:
        call_counter["count"] += 1
        return f"processed:{prompt}"

    agent = TransformerAgent(llm_backend=backend, cache=LLMCache(max_size=4))

    first = agent.respond("Collect data")
    second = agent.respond("Collect data")

    assert first == second
    assert call_counter["count"] == 1
