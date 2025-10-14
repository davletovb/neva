"""Structured debate and negotiation example."""

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


class DebateEnvironment(TranscriptEnvironment):
    """Environment guiding agents through a structured debate."""

    def __init__(
        self,
        motion: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Debate Chamber", "Negotiation toward a shared statement.", scheduler)
        self.motion = motion
        self.state["stances"]: Dict[str, str] = {}
        self.state["offers"]: List[str] = []
        self.state["consensus"] = "Undecided"

    def _summarise_consensus(self) -> str:
        stances = self.state["stances"]
        if not stances:
            return "No positions recorded yet"

        counts: Dict[str, int] = {}
        for stance in stances.values():
            counts[stance] = counts.get(stance, 0) + 1

        majority = max(counts.values())
        leaders = [stance for stance, value in counts.items() if value == majority]
        if len(leaders) == 1:
            leader = leaders[0]
            if majority == len(stances):
                return f"Full alignment on {leader}"
            if majority >= len(stances) - 1:
                return f"Emerging consensus leaning {leader}"
        breakdown = ", ".join(f"{stance}: {count}" for stance, count in sorted(counts.items()))
        return f"Split viewpoints ({breakdown})"

    def record_position(self, agent: str, stance: str, proposal: str) -> None:
        self.state["stances"][agent] = stance
        self.state["offers"].append(f"{agent}: {proposal}")
        self.state["offers"] = self.state["offers"][-6:]
        self.state["consensus"] = self._summarise_consensus()

    def context(self) -> str:  # type: ignore[override]
        offers = "; ".join(self.state["offers"][-3:]) or "None"
        return (
            f"Debate motion: {self.motion}. "
            f"Consensus trend: {self.state['consensus']}. "
            f"Recent offers: {offers}. "
            f"Latest remarks: {self.recent_dialogue()}"
        )


def make_debater_backend(
    name: str,
    persona: str,
    *,
    stance: str,
    environment: DebateEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        if stance == "support":
            proposal = (
                "amending the charter to rapidly pilot the motion with transparent "
                "evaluation checkpoints"
            )
        elif stance == "oppose":
            proposal = (
                "slowing adoption until risk studies address funding, training, and "
                "long-term governance"
            )
        else:
            proposal = (
                "a compromise timeline combining a small trial with joint oversight "
                "from both camps"
            )

        environment.record_position(name, stance, proposal)
        return (
            f"argues from a {stance} stance on '{environment.motion}', builds on recent "
            f"offers, and proposes {proposal}."
        )

    return make_persona_backend(name, persona, formatter)


def make_moderator_backend(
    name: str,
    persona: str,
    *,
    environment: DebateEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        consensus = environment._summarise_consensus()
        environment.state["consensus"] = consensus
        return (
            f"synthesises the debate motion '{environment.motion}', highlights {consensus}, "
            "and invites actionable next steps."
        )

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = DebateEnvironment(
        motion="Adopt autonomous rovers for frontier habitat logistics",
        scheduler=scheduler,
    )
    manager = AgentManager()

    moderator = manager.create_agent(
        "transformer",
        name="Moderator",
        llm_backend=make_moderator_backend(
            "Moderator",
            "neutral facilitator",
            environment=environment,
        ),
    )
    moderator.set_attribute("role", "Moderator")

    advocate = manager.create_agent(
        "transformer",
        name="Advocate",
        llm_backend=make_debater_backend(
            "Advocate",
            "optimistic strategist",
            stance="support",
            environment=environment,
        ),
    )
    advocate.set_attribute("role", "Debater")

    skeptic = manager.create_agent(
        "transformer",
        name="Skeptic",
        llm_backend=make_debater_backend(
            "Skeptic",
            "risk analyst",
            stance="oppose",
            environment=environment,
        ),
    )
    skeptic.set_attribute("role", "Debater")

    mediator = manager.create_agent(
        "transformer",
        name="Mediator",
        llm_backend=make_debater_backend(
            "Mediator",
            "consensus architect",
            stance="compromise",
            environment=environment,
        ),
    )
    mediator.set_attribute("role", "Debater")

    environment.register_agent(moderator)
    environment.register_agent(advocate)
    environment.register_agent(skeptic)
    environment.register_agent(mediator)

    run_simulation("Multi-Agent Debate", environment, steps=6)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
