import threading

import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.memory import (
    AdaptiveConversationMemory,
    MemoryBudget,
    ShortTermMemory,
    VectorStoreMemory,
)
from neva.schedulers import RoundRobinScheduler
from neva.utils.state_management import SimulationSnapshot


def _embed(text):
    return [float(len(text)), float(sum(ord(char) for char in text) % 97)]


def _summary(previous, record):
    piece = f"{record.speaker}:{record.message}"
    return f"{previous}|{piece}" if previous else piece


def _environment(memory):
    env = Environment(RoundRobinScheduler())
    agent = TransformerAgent(name="agent", llm_backend=lambda _: "reply")
    agent.set_memory(memory)
    env.register_agent(agent)
    return env


def test_vector_store_memory_roundtrips_without_reembedding():
    calls = []

    def embed(text):
        calls.append(text)
        return _embed(text)

    original_memory = VectorStoreMemory(embed, top_k=2, label="semantic")
    original_memory.remember("user", "alpha", metadata={"kind": "a"})
    original_memory.remember("agent", "beta", metadata={"kind": "b"})
    original = _environment(original_memory)

    expected = original_memory.recall(query="alpha")
    calls_before_snapshot = len(calls)
    snapshot = SimulationSnapshot.from_json(original.snapshot().to_json())

    restored_memory = VectorStoreMemory(embed, top_k=2, label="different before restore")
    restored = _environment(restored_memory)
    restored.restore(snapshot)

    assert len(calls) == calls_before_snapshot
    assert restored.agents[0].memory.label == "semantic"
    assert restored.agents[0].memory.recall() == original_memory.recall()
    assert restored.agents[0].memory.recall(query="alpha") == expected
    assert restored.agents[0].memory._counter == original_memory._counter
    assert restored.agents[0].memory._vectors == original_memory._vectors


def test_restored_memory_metadata_does_not_alias_snapshot_runtime():
    original_memory = ShortTermMemory(capacity=5)
    original_memory.remember("user", "alpha", metadata={"nested": {"value": 1}})
    snapshot = _environment(original_memory).snapshot()

    restored = _environment(ShortTermMemory(capacity=5))
    restored.restore(snapshot)

    first_record = list(restored.agents[0].memory._entries)[0]
    first_record.metadata["nested"]["value"] = 999

    saved_metadata = snapshot.runtime_state["agents"]["agent"]["memory"]["records"][0]["metadata"]
    assert saved_metadata == {"nested": {"value": 1}}

    restored.restore(snapshot)
    second_record = list(restored.agents[0].memory._entries)[0]
    assert second_record.metadata == {"nested": {"value": 1}}


def test_vector_store_checkpoint_rejects_configuration_mismatch():
    original = _environment(VectorStoreMemory(_embed, top_k=2))
    original.agents[0].memory.remember("user", "alpha")
    snapshot = original.snapshot()

    restored = _environment(VectorStoreMemory(_embed, top_k=3))
    with pytest.raises(ValueError, match="vector memory configuration"):
        restored.restore(snapshot)


def test_adaptive_memory_roundtrips_exact_state_without_recomputing_views():
    embed_calls = []
    summary_calls = []

    def embed(text):
        embed_calls.append(text)
        return _embed(text)

    def summarize(previous, record):
        summary_calls.append(record.message)
        return _summary(previous, record)

    budget = MemoryBudget(max_records=4, max_tokens=100, max_embeddings=2)
    original_memory = AdaptiveConversationMemory(
        summarizer=summarize,
        embedder=embed,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=budget,
        label="adaptive",
    )
    original_memory.remember("user", "alpha")
    original_memory.remember("agent", "beta")
    original_memory.remember("user", "gamma")
    original = _environment(original_memory)

    expected_plain = original_memory.recall()
    expected_query = original_memory.recall(query="alpha")
    embed_calls_before = len(embed_calls)
    summary_calls_before = len(summary_calls)
    snapshot = SimulationSnapshot.from_json(original.snapshot().to_json())

    restored_budget = MemoryBudget(max_records=4, max_tokens=100, max_embeddings=2)
    restored_memory = AdaptiveConversationMemory(
        summarizer=summarize,
        embedder=embed,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=restored_budget,
        label="placeholder",
    )
    restored = _environment(restored_memory)
    restored.restore(snapshot)

    memory = restored.agents[0].memory
    assert len(embed_calls) == embed_calls_before
    assert len(summary_calls) == summary_calls_before
    assert memory.label == "adaptive"
    assert memory.recall() == expected_plain
    assert memory.recall(query="alpha") == expected_query
    assert list(memory.iter_history()) == list(original_memory.iter_history())
    assert memory._token_counts == original_memory._token_counts
    assert memory._vector_cache == original_memory._vector_cache
    assert memory._id_counter == original_memory._id_counter
    assert memory._budget._embedding_calls == original_memory._budget._embedding_calls

    before_vectors = dict(memory._vector_cache)
    memory.remember("agent", "delta")
    assert memory._budget._embedding_calls == 2
    assert memory._vector_cache == before_vectors


@pytest.mark.parametrize(
    "replacement",
    [
        AdaptiveConversationMemory(
            summarizer=_summary,
            embedder=_embed,
            short_term_capacity=3,
            semantic_top_k=2,
            initial_summary="seed",
            budget=MemoryBudget(max_records=4, max_tokens=100, max_embeddings=2),
        ),
        AdaptiveConversationMemory(
            summarizer=_summary,
            embedder=None,
            short_term_capacity=2,
            semantic_top_k=2,
            initial_summary="seed",
            budget=MemoryBudget(max_records=4, max_tokens=100, max_embeddings=2),
        ),
        AdaptiveConversationMemory(
            summarizer=_summary,
            embedder=_embed,
            short_term_capacity=2,
            semantic_top_k=2,
            initial_summary="seed",
            budget=MemoryBudget(max_records=5, max_tokens=100, max_embeddings=2),
        ),
    ],
)
def test_adaptive_checkpoint_rejects_configuration_mismatch(replacement):
    original_memory = AdaptiveConversationMemory(
        summarizer=_summary,
        embedder=_embed,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=MemoryBudget(max_records=4, max_tokens=100, max_embeddings=2),
    )
    original_memory.remember("user", "alpha")
    snapshot = _environment(original_memory).snapshot()

    restored = _environment(replacement)
    with pytest.raises(ValueError, match="adaptive memory|budget configuration"):
        restored.restore(snapshot)


class _LockedEmbedder:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []

    def __call__(self, text):
        with self.lock:
            self.calls.append(text)
        return _embed(text)


class _LockedSummarizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []

    def __call__(self, previous, record):
        with self.lock:
            self.calls.append(record.message)
        return _summary(previous, record)


def test_vector_restore_preserves_non_deepcopyable_embedder_identity():
    original = _environment(VectorStoreMemory(_embed, top_k=2))
    original.agents[0].memory.remember("user", "alpha")
    snapshot = original.snapshot()

    embedder = _LockedEmbedder()
    restored_memory = VectorStoreMemory(embedder, top_k=2)
    restored = _environment(restored_memory)

    restored.restore(snapshot)

    memory = restored.agents[0].memory
    assert memory._embedder is embedder
    memory.recall(query="alpha")
    assert embedder.calls == ["alpha"]


def test_adaptive_restore_preserves_callables_and_budget_identity():
    original_memory = AdaptiveConversationMemory(
        summarizer=_summary,
        embedder=_embed,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=MemoryBudget(max_records=4, max_tokens=100, max_embeddings=3),
    )
    original_memory.remember("user", "alpha")
    original_memory.remember("agent", "beta")
    snapshot = _environment(original_memory).snapshot()

    summarizer = _LockedSummarizer()
    embedder = _LockedEmbedder()
    budget = MemoryBudget(max_records=4, max_tokens=100, max_embeddings=3)
    restored_memory = AdaptiveConversationMemory(
        summarizer=summarizer,
        embedder=embedder,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=budget,
    )
    restored = _environment(restored_memory)

    restored.restore(snapshot)

    memory = restored.agents[0].memory
    assert memory._summarizer is summarizer
    assert memory._summary._summarizer is summarizer
    assert memory._embedder is embedder
    assert memory._budget is budget
    assert budget._embedding_calls == original_memory._budget._embedding_calls

    memory.remember("user", "gamma")
    assert summarizer.calls == ["gamma"]
    assert embedder.calls == ["user: gamma"]
    assert budget._embedding_calls == 3


def test_shared_adaptive_budget_identity_survives_restore():
    original_budget = MemoryBudget(max_records=4, max_tokens=100, max_embeddings=4)
    original = Environment(RoundRobinScheduler())
    for name in ("A", "B"):
        memory = AdaptiveConversationMemory(
            summarizer=_summary,
            embedder=_embed,
            budget=original_budget,
        )
        memory.remember("user", name)
        agent = TransformerAgent(name=name, llm_backend=lambda _: "reply")
        agent.set_memory(memory)
        original.register_agent(agent)
    snapshot = original.snapshot()

    restored_budget = MemoryBudget(max_records=4, max_tokens=100, max_embeddings=4)
    restored = Environment(RoundRobinScheduler())
    for name in ("A", "B"):
        memory = AdaptiveConversationMemory(
            summarizer=_summary,
            embedder=_embed,
            budget=restored_budget,
        )
        agent = TransformerAgent(name=name, llm_backend=lambda _: "reply")
        agent.set_memory(memory)
        restored.register_agent(agent)

    restored.restore(snapshot)

    first_budget = restored.agents[0].memory._budget
    second_budget = restored.agents[1].memory._budget
    assert first_budget is restored_budget
    assert second_budget is restored_budget
    assert restored_budget._embedding_calls == original_budget._embedding_calls


def test_adaptive_without_embedder_roundtrips():
    original_memory = AdaptiveConversationMemory(
        summarizer=_summary,
        embedder=None,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=None,
    )
    original_memory.remember("user", "alpha")
    snapshot = _environment(original_memory).snapshot()

    restored_memory = AdaptiveConversationMemory(
        summarizer=_summary,
        embedder=None,
        short_term_capacity=2,
        semantic_top_k=2,
        initial_summary="seed",
        budget=None,
    )
    restored = _environment(restored_memory)
    restored.restore(snapshot)

    memory = restored.agents[0].memory
    assert memory._embedder is None
    assert memory._budget is None
    assert list(memory.iter_history()) == list(original_memory.iter_history())
    assert memory.recall() == original_memory.recall()


@pytest.mark.parametrize(
    "original_has_budget, restored_has_budget",
    [(True, False), (False, True)],
)
def test_adaptive_checkpoint_rejects_budget_presence_mismatch(
    original_has_budget, restored_has_budget
):
    original_memory = AdaptiveConversationMemory(
        summarizer=_summary,
        embedder=None,
        budget=MemoryBudget(max_records=4) if original_has_budget else None,
    )
    original_memory.remember("user", "alpha")
    snapshot = _environment(original_memory).snapshot()

    restored_memory = AdaptiveConversationMemory(
        summarizer=_summary,
        embedder=None,
        budget=MemoryBudget(max_records=4) if restored_has_budget else None,
    )
    restored = _environment(restored_memory)

    with pytest.raises(ValueError, match="budget configuration"):
        restored.restore(snapshot)
