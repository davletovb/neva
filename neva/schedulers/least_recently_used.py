"""Least Recently Used (LRU) scheduler implementation."""

from __future__ import annotations

from typing import List

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.observer import SimulationObserver


class LeastRecentlyUsedScheduler(Scheduler):
    """Activate the agent that has waited the longest since last execution."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._queue: List[AIAgent] = []

    def add(self, agent: AIAgent, **_: object) -> None:
        if agent not in self.agents:
            self.agents.append(agent)
            self._queue.append(agent)
        elif agent not in self._queue:
            self._queue.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._queue:
            raise SchedulingError(
                "LeastRecentlyUsedScheduler has no agents to schedule."
            )

        total_considered = len(self._queue)
        for _ in range(total_considered):
            agent = self._queue.pop(0)
            self._queue.append(agent)
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise SchedulingError(
            "LeastRecentlyUsedScheduler has no active agents to schedule."
        )

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._queue = [queued for queued in self._queue if queued is not agent]


__all__ = ["LeastRecentlyUsedScheduler"]
