"""Scheduler implementations for controlling agent execution order."""

import random
from typing import Callable, Dict, List, Optional, Tuple

from models import AIAgent, Scheduler
from observer import SimulationObserver

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
            raise RuntimeError("RandomScheduler has no active agents to schedule.")
        agent = random.choice(active_agents)
        self.record_metrics(agent)
        return agent


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
            raise RuntimeError("RoundRobinScheduler has no agents to schedule.")

        total_agents = len(self.agents)
        for _ in range(total_agents):
            agent = self.agents[self.current_index]
            self.current_index = (self.current_index + 1) % total_agents
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise RuntimeError("RoundRobinScheduler has no active agents to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        if not self.agents:
            self.current_index = 0
            return
        self.current_index %= len(self.agents)


class PriorityScheduler(Scheduler):
    """Activate agents based on their priority (higher first)."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._queue: List[Tuple[int, AIAgent]] = []

    def add(self, agent: AIAgent, **kwargs) -> None:
        priority = kwargs.get("priority", 1)
        self._queue.append((priority, agent))
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._queue:
            raise RuntimeError("PriorityScheduler has no agents to schedule.")

        self._queue.sort(reverse=True)
        for _ in range(len(self._queue)):
            priority, agent = self._queue.pop(0)
            self._queue.append((priority, agent))
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise RuntimeError("PriorityScheduler has no active agents to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._queue = [(priority, queued) for priority, queued in self._queue if queued is not agent]

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
            raise RuntimeError("LeastRecentlyUsedScheduler has no agents to schedule.")

        total_considered = len(self._queue)
        for _ in range(total_considered):
            agent = self._queue.pop(0)
            self._queue.append(agent)
            if self.is_paused(agent):
                continue
            self.record_metrics(agent)
            return agent

        raise RuntimeError("LeastRecentlyUsedScheduler has no active agents to schedule.")

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._queue = [queued for queued in self._queue if queued is not agent]

class WeightedRandomScheduler(Scheduler):
    """Activate agents randomly according to provided weights."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._entries: List[Tuple[float, AIAgent]] = []

    def add(self, agent: AIAgent, **kwargs) -> None:
        weight = float(kwargs.get("weight", 1.0))
        self._entries.append((weight, agent))
        if agent not in self.agents:
            self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._entries:
            raise RuntimeError("WeightedRandomScheduler has no agents to schedule.")

        active_entries = [(weight, agent) for weight, agent in self._entries if not self.is_paused(agent)]
        if not active_entries:
            raise RuntimeError("WeightedRandomScheduler has no active agents to schedule.")

        total_weight = sum(weight for weight, _ in active_entries)
        random_weight = random.uniform(0, total_weight)
        for weight, agent in active_entries:
            if random_weight <= weight:
                self.record_metrics(agent)
                return agent
            random_weight -= weight

        # Fallback due to floating point rounding.
        agent = active_entries[-1][1]
        self.record_metrics(agent)
        return agent

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        self._entries = [(weight, queued) for weight, queued in self._entries if queued is not agent]


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

    def set_environment(self, environment) -> None:
        super().set_environment(environment)
        for scheduler in self._group_schedulers.values():
            scheduler.set_environment(environment)

    def add(self, agent: AIAgent, **kwargs) -> None:
        group = kwargs.pop("group", "default")
        scheduler_override = kwargs.pop("scheduler", None)

        existing_group = self._group_membership.get(agent)
        if existing_group is not None and existing_group != group:
            self._detach_agent_from_group(agent, existing_group)

        group_scheduler = self._ensure_group_scheduler(group, scheduler_override)

        if agent not in self.agents:
            self.agents.append(agent)

        self._group_membership[agent] = group
        group_scheduler.add(agent, **kwargs)
        # Ensure paused state propagates if the agent was previously paused.
        if self.is_paused(agent):
            group_scheduler.pause(agent)

    def pause(self, agent: AIAgent) -> None:
        super().pause(agent)
        group = self._group_membership.get(agent)
        if group is not None:
            self._group_schedulers[group].pause(agent)

    def resume(self, agent: AIAgent) -> None:
        super().resume(agent)
        group = self._group_membership.get(agent)
        if group is not None:
            self._group_schedulers[group].resume(agent)

    def terminate(self, agent: AIAgent) -> None:
        group = self._group_membership.pop(agent, None)
        if group is not None and group in self._group_schedulers:
            self._group_schedulers[group].terminate(agent)
            if not self._group_schedulers[group].agents:
                self._remove_group(group)
        super().terminate(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._group_order:
            raise RuntimeError("CompositeScheduler has no agents to schedule.")

        visited_groups = 0
        total_groups = len(self._group_order)

        while visited_groups < total_groups:
            group_name = self._group_order[self._group_index]
            scheduler = self._group_schedulers[group_name]
            self._group_index = (self._group_index + 1) % total_groups
            try:
                agent = scheduler.get_next_agent()
            except RuntimeError:
                visited_groups += 1
                continue

            if agent is None or self.is_paused(agent):
                visited_groups += 1
                continue

            self.record_metrics(agent)
            return agent

        raise RuntimeError("CompositeScheduler has no active agents to schedule.")

    def _ensure_group_scheduler(
        self, group: str, scheduler_override: Optional[Scheduler]
    ) -> Scheduler:
        if group in self._group_schedulers:
            if scheduler_override is not None and scheduler_override is not self._group_schedulers[group]:
                raise ValueError(f"Group '{group}' already has an assigned scheduler.")
            return self._group_schedulers[group]

        scheduler = scheduler_override or self._group_scheduler_factory()
        # Avoid nested schedulers double-counting metrics. The composite scheduler
        # records metrics once agents are selected.
        if hasattr(scheduler, "simulation_observer"):
            scheduler.simulation_observer = None  # type: ignore[assignment]
        scheduler.set_environment(self.environment)
        self._group_schedulers[group] = scheduler
        self._group_order.append(group)
        self._group_index %= len(self._group_order)
        return scheduler

    def _remove_group(self, group: str) -> None:
        if group not in self._group_schedulers:
            return
        self._group_schedulers.pop(group)
        if group in self._group_order:
            index = self._group_order.index(group)
            self._group_order.pop(index)
            if self._group_order:
                self._group_index %= len(self._group_order)
            else:
                self._group_index = 0

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        group = self._group_membership.get(agent)
        if group is None:
            return
        self._detach_agent_from_group(agent, group)

    def _detach_agent_from_group(self, agent: AIAgent, group: str) -> None:
        self._group_membership.pop(agent, None)
        scheduler = self._group_schedulers.get(group)
        if scheduler is None:
            return
        if agent in scheduler.agents:
            scheduler.terminate(agent)
        if not scheduler.agents:
            self._remove_group(group)
