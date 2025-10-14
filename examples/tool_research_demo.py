"""Tool-augmented research simulation showcasing the Wikipedia tool."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.showcase_common import (
    TranscriptEnvironment,
    install_wikipedia_stub,
    make_persona_backend,
    run_simulation,
)
from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler
from neva.tools import WikipediaTool


class ResearchLabEnvironment(TranscriptEnvironment):
    """Environment encouraging tool usage for factual grounding."""

    def __init__(
        self,
        topic: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Research Lab", "Collaborative fact-finding sprint.", scheduler)
        self.topic = topic
        self.state["research_notes"] = []

    def context(self) -> str:  # type: ignore[override]
        latest_note = self.state["research_notes"][-1] if self.state["research_notes"] else "None"
        return (
            f"Topic: {self.topic}. Use the wikipedia tool to ground claims. "
            f"Latest verified note: {latest_note}. "
            f"Recent discussion: {self.recent_dialogue()}"
        )


def make_researcher_backend(
    name: str,
    persona: str,
    *,
    tool: WikipediaTool,
    environment: ResearchLabEnvironment,
) -> Callable[[str], str]:
    """Create a backend that consults Wikipedia exactly once."""

    used_tool = False

    def formatter(prompt: str) -> str:
        nonlocal used_tool
        del prompt
        topic = environment.topic
        if not used_tool:
            used_tool = True
            summary = tool.use(topic)
            environment.state["research_notes"].append(summary)
            return f"consults wikipedia about {topic}: {summary}"
        context = environment.recent_dialogue()
        return f"synthesises findings about {topic} using prior notes ({context})."

    return make_persona_backend(name, persona, formatter)


def make_analyst_backend(
    name: str,
    persona: str,
    *,
    environment: ResearchLabEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        latest = (
            environment.state["research_notes"][-1]
            if environment.state["research_notes"]
            else "hypothesis pending"
        )
        return f"reviews the shared notes and proposes experiments building on {latest}."

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    install_wikipedia_stub()
    scheduler = RoundRobinScheduler()
    environment = ResearchLabEnvironment(topic="Mars habitat design", scheduler=scheduler)
    manager = AgentManager()

    researcher_tool = WikipediaTool(summary_sentences=1)
    researcher = manager.create_agent(
        "transformer",
        name="Atlas",
        llm_backend=make_researcher_backend(
            "Atlas",
            "curious field researcher",
            tool=researcher_tool,
            environment=environment,
        ),
    )
    researcher.set_attribute("role", "Research Lead")
    researcher.register_tool(researcher_tool)

    analyst = manager.create_agent(
        "transformer",
        name="Nova",
        llm_backend=make_analyst_backend(
            "Nova",
            "systems analyst",
            environment=environment,
        ),
    )
    analyst.set_attribute("role", "Data Analyst")

    environment.register_agent(researcher)
    environment.register_agent(analyst)

    run_simulation("Tool-Augmented Research", environment, steps=4)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
