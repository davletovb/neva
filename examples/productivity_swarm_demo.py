"""Self-organising productivity swarm example."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Callable, Deque, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neva.agents import AgentManager
from neva.schedulers import RoundRobinScheduler
from neva.tools import MathTool

from examples.showcase_common import TranscriptEnvironment, make_persona_backend, run_simulation


class ProductivityEnvironment(TranscriptEnvironment):
    """Environment coordinating a self-organising productivity swarm."""

    def __init__(
        self,
        backlog: Iterable[str],
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__("Productivity Swarm", "Async collective planning.", scheduler)
        self.backlog: Deque[str] = deque(backlog)
        self.state["active_task"]: Optional[str] = None
        self.state["swarm_progress"]: Dict[str, str] = {}
        self.state["completed_tasks"]: List[str] = []

    def _ensure_active_task(self) -> None:
        if self.state["active_task"] is None and self.backlog:
            self.state["active_task"] = self.backlog.popleft()

    def context(self) -> str:  # type: ignore[override]
        active = self.state["active_task"] or "None"
        backlog_preview = ", ".join(list(self.backlog)[:2]) or "Empty"
        progress = (
            ", ".join(
                f"{agent}: {status}"
                for agent, status in sorted(self.state["swarm_progress"].items())
            )
            or "No updates yet"
        )
        completed = ", ".join(self.state["completed_tasks"][-2:]) or "None"
        return (
            f"Active task: {active}. Backlog preview: {backlog_preview}. "
            f"Progress: {progress}. Completed recently: {completed}. "
            f"Recent sync: {self.recent_dialogue()}"
        )

    def step(self) -> Optional[str]:  # type: ignore[override]
        self._ensure_active_task()
        return super().step()


def make_orchestrator_backend(
    name: str,
    persona: str,
    *,
    environment: ProductivityEnvironment,
) -> Callable[[str], str]:
    def formatter(prompt: str) -> str:
        del prompt
        environment._ensure_active_task()
        task = environment.state["active_task"]
        if task is None:
            return "confirms all backlog items are complete and thanks the swarm."
        environment.state["swarm_progress"].clear()
        return f"announces focus on '{task}' and requests progress updates."

    return make_persona_backend(name, persona, formatter)


def make_worker_backend(
    name: str,
    persona: str,
    *,
    specialty: str,
    environment: ProductivityEnvironment,
    tool: Optional[MathTool] = None,
) -> Callable[[str], str]:
    iteration = 0

    def formatter(prompt: str) -> str:
        del prompt
        nonlocal iteration
        iteration += 1
        task = environment.state["active_task"]
        if task is None:
            return "reviews documentation while awaiting the next assignment."

        progress = environment.state["swarm_progress"]
        if iteration == 1 and tool is not None:
            estimation = tool.use("2 * 3 + 1")
            progress[name] = f"planning estimates ({estimation} hours)"
            return (
                f"takes ownership of '{task}', runs quick maths with the calculator tool, "
                f"and shares the estimate of {estimation} hours for the team."
            )

        if iteration == 2:
            progress[name] = f"executing {specialty} tasks on '{task}'"
            return f"dives into {specialty} work for '{task}' and narrates intermediate findings."

        progress[name] = f"completed {specialty} contribution"
        environment.state["completed_tasks"].append(task)
        environment.state["active_task"] = None
        return f"wraps up {specialty} duties for '{task}' and signals readiness for handoff."

    return make_persona_backend(name, persona, formatter)


def run_demo() -> None:
    scheduler = RoundRobinScheduler()
    environment = ProductivityEnvironment(
        backlog=(
            "Design solar panel layout",
            "Draft emergency response checklist",
        ),
        scheduler=scheduler,
    )
    manager = AgentManager()

    orchestrator = manager.create_agent(
        "transformer",
        name="Pilot",
        llm_backend=make_orchestrator_backend(
            "Pilot",
            "swarm orchestrator",
            environment=environment,
        ),
    )
    orchestrator.set_attribute("role", "Coordinator")

    math_tool = MathTool()
    engineer = manager.create_agent(
        "transformer",
        name="Engineer",
        llm_backend=make_worker_backend(
            "Engineer",
            "systems engineer",
            specialty="systems integration",
            environment=environment,
            tool=math_tool,
        ),
    )
    engineer.set_attribute("role", "Maker")
    engineer.register_tool(math_tool)

    designer = manager.create_agent(
        "transformer",
        name="Designer",
        llm_backend=make_worker_backend(
            "Designer",
            "ux specialist",
            specialty="human-centred design",
            environment=environment,
        ),
    )
    designer.set_attribute("role", "Maker")

    environment.register_agent(orchestrator)
    environment.register_agent(engineer)
    environment.register_agent(designer)

    run_simulation("Productivity Swarm", environment, steps=6)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
