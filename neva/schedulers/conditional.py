"""Conditional scheduler implementation."""

from __future__ import annotations

from typing import Callable, Dict, cast

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import ConfigurationError, SchedulingError
from neva.utils.observer import SimulationObserver

Condition = Callable[[AIAgent], bool]


class ConditionalScheduler(Scheduler):
    """Activate agents only when associated conditions evaluate to ``True``."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._conditions: Dict[AIAgent, Condition] = {}
        self._current_index = 0

    def add(self, agent: AIAgent, **kwargs: object) -> None:
        condition = kwargs.get("condition") or kwargs.get("predicate")
        if condition is None:
            condition = lambda _: True
        if not callable(condition):
            raise ConfigurationError("ConditionalScheduler requires a callable condition.")

        self._conditions[agent] = cast(Condition, condition)
        if agent not in self.agents:
            self.agents.append(agent)

    def set_condition(self, agent: AIAgent, condition: Condition) -> None:
        """Update the condition controlling when ``agent`` can execute."""

        if not callable(condition):
            raise ConfigurationError("ConditionalScheduler requires a callable condition.")
        if agent not in self.agents:
            raise ConfigurationError("Agent must be registered before assigning a condition.")
        self._conditions[agent] = cast(Condition, condition)

    def get_next_agent(self) -> AIAgent:
        if not self.agents:
            raise SchedulingError("ConditionalScheduler has no agents to schedule.")

        total_agents = len(self.agents)
        for _ in range(total_agents):
            agent = self.agents[self._current_index]
            self._current_index = (self._current_index + 1) % total_agents
            if self.is_paused(agent):
                continue

            condition = self._conditions.get(agent)
            try:
                allowed = condition(agent) if condition is not None else True
            except Exception as exc:  # pragma: no cover - defensive guard
                raise SchedulingError("Condition callable raised an exception") from exc

            if allowed:
                self.record_metrics(agent)
                return agent

        raise SchedulingError("ConditionalScheduler has no agents meeting their condition.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._conditions.pop(agent, None)
        if self.agents:
            self._current_index %= len(self.agents)
        else:
            self._current_index = 0


__all__ = ["ConditionalScheduler"]
