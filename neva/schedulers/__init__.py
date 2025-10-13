"""Schedulers controlling agent execution order."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from neva.agents.base import AIAgent
from neva.utils.exceptions import ConfigurationError, SchedulingError
from neva.utils.observer import SimulationObserver

from .base import Scheduler
from .round_robin import RoundRobinScheduler

if TYPE_CHECKING:  # pragma: no cover - import used only for typing.
    from neva.environments.base import Environment


class RandomScheduler(Scheduler):
    """Activate agents in a random order each step."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()

    def add(self, agent: AIAgent, **_: object) -> None:
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        active_agents = self._active_agents()
        if not active_agents:
            raise SchedulingError("RandomScheduler has no active agents to schedule.")
        agent = random.choice(active_agents)
        self.record_metrics(agent)
        return agent


class PriorityScheduler(Scheduler):
    """Activate agents based on their priority (higher first)."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._queue: List[Tuple[int, AIAgent]] = []

    def add(self, agent: AIAgent, **kwargs: object) -> None:
        priority = kwargs.get("priority", 1)
        self._queue.append((priority, agent))
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._queue:
            raise SchedulingError("PriorityScheduler has no agents to schedule.")

        self._queue.sort(reverse=True)
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


class LeastRecentlyUsedScheduler(Scheduler):
    """Activate the agent that has waited the longest."""

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
            raise SchedulingError("LeastRecentlyUsedScheduler has no agents to schedule.")

        total_considered = len(self._queue)
        for _ in range(total_considered):
            agent = self._queue.pop(0)
            self._queue.append(agent)
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise SchedulingError("LeastRecentlyUsedScheduler has no active agents to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._queue = [queued for queued in self._queue if queued is not agent]


class WeightedRandomScheduler(Scheduler):
    """Activate agents randomly according to provided weights."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._entries: List[Tuple[float, AIAgent]] = []

    def add(self, agent: AIAgent, **kwargs: object) -> None:
        weight = float(kwargs.get("weight", 1.0))
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
            raise SchedulingError(
                "WeightedRandomScheduler has no active agents to schedule."
            )

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


class CompositeScheduler(Scheduler):
    """Coordinate multiple sub-schedulers managing agent sub-groups."""

    def __init__(self, group_scheduler_factory: Optional[Callable[[], Scheduler]] = None) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._group_scheduler_factory: Callable[[], Scheduler] = (
            group_scheduler_factory or RoundRobinScheduler
        )
        self._group_schedulers: Dict[str, Scheduler] = {}
        self._group_membership: Dict[AIAgent, str] = {}
        self._group_order: List[str] = []
        self._group_index = 0

    def set_environment(self, environment: "Environment") -> None:
        super().set_environment(environment)
        for scheduler in self._group_schedulers.values():
            scheduler.set_environment(environment)

    def add(self, agent: AIAgent, **kwargs: object) -> None:
        group = kwargs.pop("group", "default")
        scheduler_override = kwargs.pop("scheduler", None)

        existing_group = self._group_membership.get(agent)
        if existing_group is not None and existing_group != group:
            self._detach_agent_from_group(agent, existing_group)

        group_scheduler = self._ensure_group_scheduler(group, scheduler_override)

        if agent not in self.agents:
            self.agents.append(agent)
        if agent not in group_scheduler.agents:
            group_scheduler.add(agent, **kwargs)
        self._group_membership[agent] = group

        if group not in self._group_order:
            self._group_order.append(group)

    def get_next_agent(self) -> AIAgent:
        if not self._group_order:
            raise SchedulingError("CompositeScheduler has no groups to schedule.")

        total_groups = len(self._group_order)
        for _ in range(total_groups):
            group = self._group_order[self._group_index]
            self._group_index = (self._group_index + 1) % total_groups
            scheduler = self._group_schedulers[group]
            try:
                agent = scheduler.get_next_agent()
            except SchedulingError:
                continue
            if self.is_paused(agent):
                scheduler.pause(agent)
                continue
            self.record_metrics(agent)
            return agent

        raise SchedulingError("CompositeScheduler has no active agents to schedule.")

    def _ensure_group_scheduler(
        self, group: str, scheduler_override: Optional[Scheduler]
    ) -> Scheduler:
        if group not in self._group_schedulers:
            scheduler = scheduler_override or self._group_scheduler_factory()
            self._validate_scheduler(scheduler)
            self._group_schedulers[group] = scheduler
            if self.environment is not None:
                scheduler.set_environment(self.environment)
        return self._group_schedulers[group]

    def _detach_agent_from_group(self, agent: AIAgent, group: str) -> None:
        scheduler = self._group_schedulers.get(group)
        if scheduler is None:
            return
        scheduler.terminate(agent)
        self._group_membership.pop(agent, None)
        if not scheduler.agents:
            self._group_schedulers.pop(group, None)
            self._group_order = [existing for existing in self._group_order if existing != group]
            if self._group_index >= len(self._group_order) and self._group_order:
                self._group_index %= len(self._group_order)

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        group = self._group_membership.pop(agent, None)
        if group is None:
            return
        scheduler = self._group_schedulers.get(group)
        if scheduler is None:
            return
        scheduler.terminate(agent)
        if not scheduler.agents:
            self._group_schedulers.pop(group, None)
            self._group_order = [existing for existing in self._group_order if existing != group]
            if self._group_index >= len(self._group_order) and self._group_order:
                self._group_index %= len(self._group_order)

    @staticmethod
    def _validate_scheduler(scheduler: Scheduler) -> None:
        if not isinstance(scheduler, Scheduler):
            raise ConfigurationError("Provided scheduler must be an instance of Scheduler")


__all__ = [
    "CompositeScheduler",
    "LeastRecentlyUsedScheduler",
    "PriorityScheduler",
    "RandomScheduler",
    "RoundRobinScheduler",
    "Scheduler",
    "WeightedRandomScheduler",
]
