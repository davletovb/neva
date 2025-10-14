"""Cooperative tactical party coordination example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.showcase_common import TranscriptEnvironment, make_persona_backend, run_simulation
from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler


class QuestEncounterEnvironment(TranscriptEnvironment):
    """Environment capturing a tactical party responding to encounters."""

    def __init__(
        self,
        encounters: Iterable[Tuple[str, str]],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Quest Encounter", "Cooperative party strategy.", scheduler)
        self.encounters: List[Tuple[str, str]] = list(encounters)
        self.state["stage"] = 0
        self.state["team_status"]: Dict[str, str] = {}

    def _current_encounter(self) -> Tuple[str, str]:
        index = int(self.state["stage"]) % max(len(self.encounters), 1)
        return self.encounters[index]

    def context(self) -> str:  # type: ignore[override]
        encounter, threat = self._current_encounter()
        status = (
            ", ".join(
                f"{agent}: {state}" for agent, state in sorted(self.state["team_status"].items())
            )
            or "Awaiting orders"
        )
        return (
            f"Active encounter: {encounter} (threat {threat}). "
            f"Team status: {status}. "
            f"Recent tactical chatter: {self.recent_dialogue()}"
        )

    def step(self) -> Optional[str]:  # type: ignore[override]
        message = super().step()
        if message:
            self.state["stage"] = (int(self.state["stage"]) + 1) % max(len(self.encounters), 1)
        return message


def make_party_member_backend(
    name: str,
    persona: str,
    *,
    role: str,
    environment: QuestEncounterEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        encounter, threat = environment._current_encounter()
        if role == "tank":
            action = "raises a barrier and calls focus fire"
        elif role == "healer":
            action = "deploys protective wards and triage rotations"
        else:
            action = "channels elemental combos targeting enemy weaknesses"

        environment.state["team_status"][name] = f"{role} covering {encounter}"
        return (
            f"evaluates the {encounter} (threat {threat}), {action}, and synchronises "
            "cooldown timings with the squad."
        )

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = QuestEncounterEnvironment(
        encounters=(
            ("ambush in the crystalline canyon", "high"),
            ("arcane puzzle locking the vault", "medium"),
            ("hydra boss awakening", "critical"),
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    tank = manager.create_agent(
        "transformer",
        name="Vanguard",
        llm_backend=make_party_member_backend(
            "Vanguard",
            "shield-bearing tactician",
            role="tank",
            environment=environment,
        ),
    )
    tank.set_attribute("role", "Frontline")

    healer = manager.create_agent(
        "transformer",
        name="Lumina",
        llm_backend=make_party_member_backend(
            "Lumina",
            "radiant healer",
            role="healer",
            environment=environment,
        ),
    )
    healer.set_attribute("role", "Support")

    mage = manager.create_agent(
        "transformer",
        name="Aeris",
        llm_backend=make_party_member_backend(
            "Aeris",
            "elemental tactician",
            role="caster",
            environment=environment,
        ),
    )
    mage.set_attribute("role", "Damage")

    environment.register_agent(tank)
    environment.register_agent(healer)
    environment.register_agent(mage)

    run_simulation("Tactical Party Coordination", environment, steps=6)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
