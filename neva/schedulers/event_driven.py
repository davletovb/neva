"""Event-driven scheduler implementation."""

from __future__ import annotations

from collections import deque
from typing import Deque

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.observer import SimulationObserver


class EventDrivenScheduler(Scheduler):
    """Only activate agents that have published an event signal."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._event_queue: Deque[AIAgent] = deque()

    def add(self, agent: AIAgent, **_: object) -> None:
        if agent not in self.agents:
            self.agents.append(agent)

    def notify_event(self, agent: AIAgent) -> None:
        """Register an event for ``agent`` so it becomes eligible for execution."""

        if agent not in self.agents or agent in self._event_queue:
            return
        self._event_queue.append(agent)

    def get_next_agent(self) -> AIAgent:
        while self._event_queue:
            agent = self._event_queue.popleft()
            if agent not in self.agents:
                continue
            if self.is_paused(agent):
                self._event_queue.append(agent)
                continue
            self.record_metrics(agent)
            return agent

        raise SchedulingError("EventDrivenScheduler has no pending events to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._event_queue = deque(existing for existing in self._event_queue if existing is not agent)


__all__ = ["EventDrivenScheduler"]
