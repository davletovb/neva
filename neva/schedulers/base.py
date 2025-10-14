"""Base scheduler abstraction for coordinating agent execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Set

from neva.agents.base import AIAgent

if TYPE_CHECKING:  # pragma: no cover - import used only for typing.
    from neva.environments.base import Environment
    from neva.utils.observer import SimulationObserver


class Scheduler(ABC):
    """Base class for iterating over a collection of agents."""

    def __init__(self) -> None:
        self.agents: List[AIAgent] = []
        self.environment: Optional["Environment"] = None
        self._paused_agents: Set[AIAgent] = set()
        self._termination_hooks: List[Callable[[AIAgent], None]] = []

    def set_environment(self, environment: "Environment") -> None:
        self.environment = environment

    def record_metrics(self, active_agent: Optional[AIAgent] = None) -> None:
        observer = getattr(self, "simulation_observer", None)
        if observer is not None:
            try:
                observer.collect_data(
                    list(self.agents), self.environment, active_agent=active_agent
                )
            except TypeError:
                observer.collect_data(list(self.agents), self.environment)

    # ------------------------------------------------------------------
    # Agent lifecycle controls
    # ------------------------------------------------------------------
    def register_termination_hook(self, hook: Callable[[AIAgent], None]) -> None:
        """Invoke ``hook`` whenever an agent is terminated."""

        self._termination_hooks.append(hook)

    def pause(self, agent: AIAgent) -> None:
        """Temporarily remove ``agent`` from the active scheduling pool."""

        if agent in self.agents:
            self._paused_agents.add(agent)

    def resume(self, agent: AIAgent) -> None:
        """Return ``agent`` to the active scheduling pool."""

        self._paused_agents.discard(agent)

    def is_paused(self, agent: AIAgent) -> bool:
        """Return ``True`` when ``agent`` is currently paused."""

        return agent in self._paused_agents

    def terminate(self, agent: AIAgent) -> None:
        """Remove ``agent`` from the scheduler and trigger termination hooks."""

        if agent not in self.agents:
            return

        self.agents = [existing for existing in self.agents if existing is not agent]
        self._paused_agents.discard(agent)
        self._handle_agent_removal(agent)
        for hook in list(self._termination_hooks):
            hook(agent)

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        """Allow subclasses to update internal state after ``agent`` removal."""

        return

    def _active_agents(self) -> List[AIAgent]:
        """Return the list of agents that are not currently paused."""

        return [agent for agent in self.agents if agent not in self._paused_agents]

    @abstractmethod
    def add(self, agent: AIAgent, **kwargs) -> None:
        """Register an agent with the scheduler."""

    @abstractmethod
    def get_next_agent(self) -> Optional[AIAgent]:
        """Return the next agent to act."""


__all__ = ["Scheduler"]
