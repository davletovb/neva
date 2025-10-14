"""Emergent NPC banter inside a tavern environment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.showcase_common import TranscriptEnvironment, make_persona_backend, run_simulation
from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler


class TavernEnvironment(TranscriptEnvironment):
    """Lightweight environment simulating NPC banter."""

    def __init__(
        self,
        events: Iterable[str],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Tavern", "A cosy tavern filled with dynamic NPCs.", scheduler)
        self.events: List[str] = list(events)
        self.state["event_index"] = 0

    def _current_event(self) -> str:
        index = int(self.state["event_index"]) % len(self.events)
        return self.events[index]

    def context(self) -> str:  # type: ignore[override]
        return (
            f"Current tavern event: {self._current_event()}. "
            f"Play to your archetype and riff on the rumours. "
            f"Recent rumours: {self.recent_dialogue()}"
        )

    def step(self) -> Optional[str]:  # type: ignore[override]
        message = super().step()
        if message:
            self.state["event_index"] = (int(self.state["event_index"]) + 1) % len(self.events)
        return message


def make_npc_backend(
    name: str,
    persona: str,
    *,
    environment: TavernEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        event = environment._current_event()
        rumours = environment.recent_dialogue()
        return f"reacts to '{event}' while weaving it into tavern gossip ({rumours})."

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = TavernEnvironment(
        events=(
            "A mysterious traveller shares news of a dragon",
            "A bard starts an upbeat melody",
            "A sudden blackout reveals a hidden map",
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    barkeep = manager.create_agent(
        "transformer",
        name="Barkeep",
        llm_backend=make_npc_backend(
            "Barkeep",
            "gruff host",
            environment=environment,
        ),
    )
    bard = manager.create_agent(
        "transformer",
        name="Bard",
        llm_backend=make_npc_backend(
            "Bard",
            "optimistic musician",
            environment=environment,
        ),
    )
    rogue = manager.create_agent(
        "transformer",
        name="Rogue",
        llm_backend=make_npc_backend(
            "Rogue",
            "whispering informant",
            environment=environment,
        ),
    )

    environment.register_agent(barkeep)
    environment.register_agent(bard)
    environment.register_agent(rogue)

    run_simulation("Game NPC Roleplay", environment, steps=6)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
