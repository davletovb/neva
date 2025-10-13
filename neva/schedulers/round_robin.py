"""Round-robin scheduler implementation."""

from __future__ import annotations

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.observer import SimulationObserver


class RoundRobinScheduler(Scheduler):
    """Activate agents sequentially in a round-robin cycle."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self.current_index = 0

    def add(self, agent: AIAgent, **_: object) -> None:
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self.agents:
            raise SchedulingError("RoundRobinScheduler has no agents to schedule.")

        total_agents = len(self.agents)
        for _ in range(total_agents):
            agent = self.agents[self.current_index]
            self.current_index = (self.current_index + 1) % total_agents
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise SchedulingError("RoundRobinScheduler has no active agents to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        if not self.agents:
            self.current_index = 0
            return
        self.current_index %= len(self.agents)


__all__ = ["RoundRobinScheduler"]
