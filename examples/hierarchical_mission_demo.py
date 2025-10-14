"""Hierarchical leader-follower planning example."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler

from examples.showcase_common import TranscriptEnvironment, make_persona_backend, run_simulation


class MissionControlEnvironment(TranscriptEnvironment):
    """Environment coordinating hierarchical directives."""

    def __init__(
        self,
        directives: Iterable[str],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Mission Control", "Leader assigns directives to specialists.", scheduler)
        self._directives: Iterator[str] = iter(directives)
        self.state["directive"] = "Awaiting assignment"
        self.state["progress"]: Dict[str, str] = {}

    def next_directive(self) -> str:
        try:
            directive = next(self._directives)
        except StopIteration:
            directive = "Wrap up and summarise outcomes"
        self.state["directive"] = directive
        return directive

    def context(self) -> str:  # type: ignore[override]
        progress = (
            ", ".join(
                f"{agent}: {status}" for agent, status in sorted(self.state["progress"].items())
            )
            or "No updates yet"
        )
        return (
            f"Active directive: {self.state['directive']}. "
            f"Progress so far: {progress}. "
            f"Recent chatter: {self.recent_dialogue()}"
        )


def make_leader_backend(
    name: str,
    persona: str,
    *,
    environment: MissionControlEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        directive = environment.next_directive()
        return f"issues directive -> {directive}"

    return make_persona_backend(name, persona, formatter)


def make_specialist_backend(
    name: str,
    persona: str,
    *,
    speciality: str,
    environment: MissionControlEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        directive = environment.state["directive"]
        update = f"{speciality} executing '{directive}'"
        environment.state["progress"][name] = update
        return f"acknowledges directive '{directive}' and reports {update}."

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = MissionControlEnvironment(
        directives=(
            "Scout resource locations",
            "Synthesize habitat risk assessment",
            "Draft daily routine for colonists",
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    leader = manager.create_agent(
        "transformer",
        name="Command",
        llm_backend=make_leader_backend(
            "Command",
            "mission lead",
            environment=environment,
        ),
    )
    leader.set_attribute("role", "Leader")

    scout = manager.create_agent(
        "transformer",
        name="Scout",
        llm_backend=make_specialist_backend(
            "Scout",
            "terrain specialist",
            speciality="terrain analysis",
            environment=environment,
        ),
    )
    scout.set_attribute("role", "Follower")

    medic = manager.create_agent(
        "transformer",
        name="Medic",
        llm_backend=make_specialist_backend(
            "Medic",
            "biomedical expert",
            speciality="health planning",
            environment=environment,
        ),
    )
    medic.set_attribute("role", "Follower")

    environment.register_agent(leader)
    environment.register_agent(scout)
    environment.register_agent(medic)

    run_simulation("Hierarchical Mission Planning", environment, steps=5)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
