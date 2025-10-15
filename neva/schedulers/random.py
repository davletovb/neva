"""Random scheduler implementation."""

from __future__ import annotations

from random import SystemRandom

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.observer import SimulationObserver


class RandomScheduler(Scheduler):
    """Activate agents in a random order on each scheduling decision."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._rng = SystemRandom()

    def add(self, agent: AIAgent, **_: object) -> None:
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        active_agents = self._active_agents()
        if not active_agents:
            raise SchedulingError("RandomScheduler has no active agents to schedule.")

        agent = self._rng.choice(active_agents)
        self.record_metrics(agent)
        return agent


__all__ = ["RandomScheduler"]
