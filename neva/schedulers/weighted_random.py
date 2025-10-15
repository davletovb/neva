"""Weighted random scheduler implementation."""

from __future__ import annotations

import random
from typing import List, Tuple

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.observer import SimulationObserver


class WeightedRandomScheduler(Scheduler):
    """Activate agents randomly while respecting configured weights."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._entries: List[Tuple[float, AIAgent]] = []

    def add(self, agent: AIAgent, **kwargs: object) -> None:
        raw_weight = kwargs.get("weight", 1.0)
        if not isinstance(raw_weight, (int, float, str)):
            raise SchedulingError("weight must be convertible to float")
        try:
            weight = float(raw_weight)
        except ValueError:
            raise SchedulingError("weight must be convertible to float") from None
        self._entries.append((weight, agent))
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._entries:
            raise SchedulingError("WeightedRandomScheduler has no agents to schedule.")

        active_entries = [
            (weight, agent) for weight, agent in self._entries if not self.is_paused(agent)
        ]
        if not active_entries:
            raise SchedulingError("WeightedRandomScheduler has no active agents to schedule.")

        total_weight = sum(weight for weight, _ in active_entries)
        random_weight = random.uniform(0, total_weight)
        for weight, agent in active_entries:
            if random_weight <= weight:
                self.record_metrics(agent)
                return agent
            random_weight -= weight

        agent = active_entries[-1][1]
        self.record_metrics(agent)
        return agent

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._entries = [
            (weight, queued) for weight, queued in self._entries if queued is not agent
        ]


__all__ = ["WeightedRandomScheduler"]
