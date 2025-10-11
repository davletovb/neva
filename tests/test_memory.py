import os
import sys
from typing import List

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory import (
    CompositeMemory,
    MemoryRecord,
    ShortTermMemory,
    SummaryMemory,
    VectorStoreMemory,
)
from models import AIAgent


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
    with pytest.raises(ValueError):
        ShortTermMemory(capacity=0)

    with pytest.raises(ValueError):
        VectorStoreMemory(embedder=lambda text: [1.0], top_k=0)

    with pytest.raises(ValueError):
        CompositeMemory([])
