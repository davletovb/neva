"""Minimal conversational simulation between two Neva agents.

This script demonstrates the core building blocks of the Neva library:

* ``AgentManager`` for creating agents.
* ``BasicEnvironment`` to provide shared context.
* ``RoundRobinScheduler`` to coordinate turn-taking.

Both agents use lightweight stubbed language-model backends so the example
runs without external services.  Each call to :func:`environment.step`
progresses the conversation by one message, showcasing the emergent behaviour
focus of the project.
"""

from __future__ import annotations

from typing import List

from pathlib import Path

from neva.agents import AgentManager
from neva.environments import BasicEnvironment
from neva.memory import CompositeMemory, MemoryRecord, ShortTermMemory, SummaryMemory
from neva.schedulers import RoundRobinScheduler


class ConversationEnvironment(BasicEnvironment):
    """Environment that keeps a transcript of agent responses."""

    def __init__(self, name: str, description: str, scheduler: RoundRobinScheduler):
        super().__init__(name, description, scheduler)
        self.transcript: List[str] = []

    def context(self) -> str:
        if not self.transcript:
            return (
                "Introduce yourself and propose a collaborative goal for this "
                "simulation."
            )
        return "Conversation so far: " + " | ".join(self.transcript[-3:])

    def step(self) -> str | None:
        response = super().step()
        if response:
            self.transcript.append(response)
        return response


def make_reflective_backend(name: str):
    """Return a stub backend that reacts to the environment context."""

    def backend(prompt: str) -> str:
        return f"{name} reflects on: {prompt.split(':')[-1].strip()}"

    return backend


def make_agent_memory() -> CompositeMemory:
    """Create a composite memory that mixes recency and summary signals."""

    def summarizer(summary: str, record: MemoryRecord) -> str:
        statement = f"{record.speaker} noted '{record.message}'"
        if not summary:
            return statement
        return f"{summary} | {statement}"

    return CompositeMemory(
        [ShortTermMemory(capacity=4, label="recent"), SummaryMemory(summarizer)],
        label="agent cognition",
    )


def main() -> None:
    manager = AgentManager()
    scheduler = RoundRobinScheduler()
    environment = ConversationEnvironment(
        name="Idea Lab",
        description="A creative space for brainstorming new agent behaviours.",
        scheduler=scheduler,
    )

    explorer = manager.create_agent(
        "transformer",
        name="Explorer",
        llm_backend=make_reflective_backend("Explorer"),
        memory=make_agent_memory(),
    )
    curator = manager.create_agent(
        "transformer",
        name="Curator",
        llm_backend=make_reflective_backend("Curator"),
        memory=make_agent_memory(),
    )

    environment.register_agent(explorer)
    environment.register_agent(curator)

    print("=== Conversation Start ===")
    for step, message in enumerate(environment.run(6), start=1):
        print(f"Step {step}: {message}")
    print("=== Conversation End ===")

    observer = scheduler.simulation_observer
    snapshot = observer.latest_snapshot()
    print("\n=== Observer Metrics (latest snapshot) ===")
    for name, value in snapshot.items():
        print(f"{name}: {value}")

    export_path = Path("quickstart_metrics.json")
    observer.export_to_json(export_path)
    print(f"Metrics exported to {export_path.resolve()}")


if __name__ == "__main__":
    main()
