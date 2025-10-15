"""Priority-based scheduler implementation."""

from __future__ import annotations

from typing import Any, List, Tuple, cast

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.observer import SimulationObserver


class PriorityScheduler(Scheduler):
    """Activate agents based on priority (higher values run first)."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._queue: List[Tuple[int, AIAgent]] = []

    def add(self, agent: AIAgent, **kwargs: object) -> None:
        raw_priority = kwargs.get("priority", 1)
        try:
            priority = int(cast(Any, raw_priority))
        except (TypeError, ValueError):
            raise SchedulingError("priority must be an integer value") from None
        self._queue.append((priority, agent))
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._queue:
            raise SchedulingError("PriorityScheduler has no agents to schedule.")

        self._queue.sort(key=lambda item: item[0], reverse=True)
        for _ in range(len(self._queue)):
            priority, agent = self._queue.pop(0)
            self._queue.append((priority, agent))
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise SchedulingError("PriorityScheduler has no active agents to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._queue = [
            (priority, queued) for priority, queued in self._queue if queued is not agent
        ]


__all__ = ["PriorityScheduler"]
