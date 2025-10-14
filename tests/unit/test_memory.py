from typing import List, Sequence

import pytest

from neva.agents import AIAgent
from neva.memory import (
    AdaptiveConversationMemory,
    CompositeMemory,
    MemoryBudget,
    MemoryRecord,
    ShortTermMemory,
    SummaryMemory,
    VectorStoreMemory,
)
from neva.utils.exceptions import MemoryConfigurationError


class EchoAgent(AIAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts: List[str] = []

    def respond(self, message: str) -> str:
        prompt = self.prepare_prompt(message)
        self.prompts.append(prompt)
        return f"echo:{message}"


def test_short_term_memory_retains_recent_turns():
    memory = ShortTermMemory(capacity=3)
    for idx in range(5):
        memory.remember("user", f"message-{idx}")

    recalled = memory.recall()
    assert "message-1" not in recalled
    assert recalled.splitlines() == [
        "user: message-2",
        "user: message-3",
        "user: message-4",
    ]


def test_summary_memory_updates_with_records():
    def summarizer(summary: str, record: MemoryRecord) -> str:
        history = f"{record.speaker} -> {record.message}"
        if not summary:
            return history
        return f"{summary}; {history}"

    memory = SummaryMemory(summarizer, initial_summary="start")
    memory.remember("user", "hello")
    memory.remember("agent", "hi")
    assert memory.recall() == "start; user -> hello; agent -> hi"


def test_vector_memory_scores_similar_messages():
    def embed(text: str):
        vowels = sum(1 for char in text.lower() if char in "aeiou")
        spaces = text.count(" ")
        return [float(len(text)), float(spaces), float(vowels)]

    memory = VectorStoreMemory(embedder=embed, top_k=2)
    memory.remember("user", "hi")
    memory.remember("agent", "greetings")
    memory.remember("user", "another message")

    recalled = memory.recall(query="longer content")
    lines = recalled.splitlines()
    assert "user: hi" not in lines
    assert {"user: another message", "agent: greetings"}.issubset(set(lines))


def test_agent_prompt_includes_memory_context():
    memory = ShortTermMemory(capacity=4)
    agent = EchoAgent(name="mem", memory=memory)

    agent.receive("System setup", sender="system")
    agent.receive("Hello", sender="user")

    agent.process_input("Follow up")
    assert any("Relevant memory" in prompt for prompt in agent.prompts)


def test_composite_memory_merges_modules():
    short_term = ShortTermMemory(capacity=2, label="buffer")

    def summarizer(summary: str, record: MemoryRecord) -> str:
        return f"last:{record.message}"

    summary = SummaryMemory(summarizer, label="summary")
    composite = CompositeMemory([short_term, summary])

    composite.remember("user", "alpha")
    composite.remember("agent", "beta")

    recalled = composite.recall()
    assert "buffer" in recalled
    assert "summary" in recalled


def test_memory_invalid_configurations_raise():
    with pytest.raises(MemoryConfigurationError):
        ShortTermMemory(capacity=0)

    with pytest.raises(MemoryConfigurationError):
        VectorStoreMemory(embedder=lambda text: [1.0], top_k=0)

    with pytest.raises(MemoryConfigurationError):
        CompositeMemory([])


def _adaptive_summariser(summary: str, record: MemoryRecord) -> str:
    snippet = f"{record.speaker} -> {record.message}"
    return f"{summary}; {snippet}" if summary else snippet


def _vowel_embedder(text: str) -> Sequence[float]:
    lowered = text.lower()
    return [float(lowered.count(ch)) for ch in "aeiou"]


def test_adaptive_memory_combines_views():
    memory = AdaptiveConversationMemory(
        summarizer=_adaptive_summariser,
        embedder=_vowel_embedder,
        short_term_capacity=2,
        semantic_top_k=2,
    )

    memory.remember("user", "Discuss launch plan")
    memory.remember("agent", "Draft proposal outline")
    memory.remember("user", "Review launch checklist")
    memory.remember("agent", "Finalize launch plan")

    combined = memory.recall()
    assert "recent:\nuser: Review launch checklist" in combined
    assert "summary:\nuser -> Discuss launch plan" in combined

    semantic = memory.recall(query="launch plan")
    assert "semantic:\n" in semantic
    assert "agent: Finalize launch plan" in semantic


def test_adaptive_memory_budget_trims_history():
    budget = MemoryBudget(max_records=3, max_tokens=20)
    memory = AdaptiveConversationMemory(
        summarizer=_adaptive_summariser,
        short_term_capacity=5,
        budget=budget,
    )

    for idx in range(6):
        memory.remember("user", f"turn-{idx}")

    remaining = [record.message for record in memory.iter_history()]
    assert remaining == ["turn-3", "turn-4", "turn-5"]

    window = memory.recent_window(2)
    assert [record.message for record in window] == ["turn-4", "turn-5"]


def test_memory_budget_limits_embedding_calls():
    embed_calls = 0

    def embed(text: str) -> Sequence[float]:
        nonlocal embed_calls
        embed_calls += 1
        return [float(len(text))]

    budget = MemoryBudget(max_embeddings=2)
    memory = AdaptiveConversationMemory(
        summarizer=_adaptive_summariser,
        embedder=embed,
        budget=budget,
    )

    for idx in range(5):
        memory.remember("user", f"item-{idx}")

    assert embed_calls == 2

