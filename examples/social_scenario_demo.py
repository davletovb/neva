"""Community coordination and social dynamics example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.showcase_common import TranscriptEnvironment, make_persona_backend, run_simulation
from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler


class CommunityScenarioEnvironment(TranscriptEnvironment):
    """Environment modelling social coordination dynamics."""

    def __init__(
        self,
        scenario: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Community Hub", "Neighbours co-create solutions.", scheduler)
        self.scenario = scenario
        self.state["shared_resources"]: List[str] = []
        self.state["mood"]: Dict[str, str] = {}

    def record_contribution(self, agent: str, contribution: str, mood: str) -> None:
        self.state["shared_resources"].append(f"{agent}: {contribution}")
        self.state["shared_resources"] = self.state["shared_resources"][-6:]
        self.state["mood"][agent] = mood

    def _mood_summary(self) -> str:
        if not self.state["mood"]:
            return "Balanced"
        counts: Dict[str, int] = {}
        for mood in self.state["mood"].values():
            counts[mood] = counts.get(mood, 0) + 1
        dominant = max(counts, key=counts.get)
        return f"Dominant mood: {dominant}"

    def context(self) -> str:  # type: ignore[override]
        resources = "; ".join(self.state["shared_resources"][-3:]) or "None"
        return (
            f"Scenario: {self.scenario}. "
            f"Mood snapshot: {self._mood_summary()}. "
            f"Shared resources: {resources}. "
            f"Recent coordination: {self.recent_dialogue()}"
        )


def make_social_actor_backend(
    name: str,
    persona: str,
    *,
    focus: str,
    environment: CommunityScenarioEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        if focus == "logistics":
            contribution = "secures venue layout and timeline checkpoints"
            mood = "organised"
        elif focus == "inclusion":
            contribution = "maps accessibility needs and rotating volunteer pairs"
            mood = "caring"
        else:
            contribution = "launches story-sharing circle with musical interludes"
            mood = "energised"

        environment.record_contribution(name, contribution, mood)
        return (
            f"leans on {focus} expertise for '{environment.scenario}', contributes {contribution}, "
            f"and keeps the group feeling {mood}."
        )

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = CommunityScenarioEnvironment(
        scenario="Coordinate an inclusive neighbourhood resilience festival",
        scheduler=scheduler,
    )
    manager = AgentManager()

    logistics = manager.create_agent(
        "transformer",
        name="Logistics",
        llm_backend=make_social_actor_backend(
            "Logistics",
            "operations volunteer",
            focus="logistics",
            environment=environment,
        ),
    )
    logistics.set_attribute("role", "Planner")

    inclusion = manager.create_agent(
        "transformer",
        name="Inclusion",
        llm_backend=make_social_actor_backend(
            "Inclusion",
            "community advocate",
            focus="inclusion",
            environment=environment,
        ),
    )
    inclusion.set_attribute("role", "Wellbeing Lead")

    culture = manager.create_agent(
        "transformer",
        name="Culture",
        llm_backend=make_social_actor_backend(
            "Culture",
            "arts facilitator",
            focus="culture",
            environment=environment,
        ),
    )
    culture.set_attribute("role", "Experience Designer")

    environment.register_agent(logistics)
    environment.register_agent(inclusion)
    environment.register_agent(culture)

    run_simulation("Social Dynamics Simulation", environment, steps=6)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
