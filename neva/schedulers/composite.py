"""Composite scheduler implementation."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.schedulers.round_robin import RoundRobinScheduler
from neva.utils.exceptions import ConfigurationError, SchedulingError
from neva.utils.observer import SimulationObserver

if False:  # pragma: no cover - for type checking only
    from neva.environments.base import Environment


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
        group_value = kwargs.pop("group", "default")
        if not isinstance(group_value, str):
            raise ConfigurationError("group must be provided as a string identifier")
        group = group_value

        scheduler_override_value = kwargs.pop("scheduler", None)
        scheduler_override: Optional[Scheduler]
        if scheduler_override_value is None:
            scheduler_override = None
        elif isinstance(scheduler_override_value, Scheduler):
            scheduler_override = scheduler_override_value
        else:
            raise ConfigurationError("scheduler override must be an instance of Scheduler")

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
            if agent is None:
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


__all__ = ["CompositeScheduler"]
