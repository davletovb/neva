"""Scheduler implementations for controlling agent execution order."""

import random
from typing import List, Tuple

from models import AIAgent, Scheduler
from observer import SimulationObserver

class RandomScheduler(Scheduler):
    """Activate agents in a random order each step."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()

    def add(self, agent: AIAgent, **_: object) -> None:
        self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self.agents:
            raise RuntimeError("RandomScheduler has no agents to schedule.")
        agent = random.choice(self.agents)
        self.record_metrics(agent)
        return agent


class RoundRobinScheduler(Scheduler):
    """Activate agents sequentially in a round-robin cycle."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self.current_index = 0

    def add(self, agent: AIAgent, **_: object) -> None:
        self.agents.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self.agents:
            raise RuntimeError("RoundRobinScheduler has no agents to schedule.")
        agent = self.agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.agents)
        self.record_metrics(agent)
        return agent


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
        priority, agent = self._queue.pop(0)
        # Reinsert agent with same priority for future scheduling
        self._queue.append((priority, agent))
        self.record_metrics(agent)
        return agent

class LeastRecentlyUsedScheduler(Scheduler):
    """Activate the agent that has waited the longest."""

    def __init__(self) -> None:
        super().__init__()
        self.simulation_observer = SimulationObserver()
        self._queue: List[AIAgent] = []

    def add(self, agent: AIAgent, **_: object) -> None:
        self.agents.append(agent)
        self._queue.append(agent)

    def get_next_agent(self) -> AIAgent:
        if not self._queue:
            raise RuntimeError("LeastRecentlyUsedScheduler has no agents to schedule.")
        agent = self._queue.pop(0)
        self._queue.append(agent)
        self.record_metrics(agent)
        return agent

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
        total_weight = sum(weight for weight, _ in self._entries)
        random_weight = random.uniform(0, total_weight)
        for weight, agent in self._entries:
            if random_weight <= weight:
                self.record_metrics(agent)
                return agent
            random_weight -= weight

        # Fallback due to floating point rounding.
        agent = self._entries[-1][1]
        self.record_metrics(agent)
        return agent
