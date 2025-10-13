"""Environment abstractions coordinating agent simulations."""

from __future__ import annotations

from typing import Dict, List, Optional

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.state_management import ConversationState, SimulationSnapshot, create_snapshot


class Environment:
    """Coordinate agents and schedulers while maintaining shared state."""

    def __init__(self, scheduler: Optional[Scheduler] = None) -> None:
        self.state: Dict[str, object] = {}
        self.scheduler = scheduler
        self.agents: List[AIAgent] = []
        if self.scheduler is not None:
            self.scheduler.set_environment(self)

    def register_agent(self, agent: AIAgent) -> None:
        agent.set_environment(self)
        self.agents.append(agent)
        if self.scheduler is not None:
            self.scheduler.add(agent)

    def context(self) -> str:
        """Return a textual description of the environment state."""

        return ""

    def step(self) -> Optional[str]:
        if self.scheduler is None or not self.agents:
            return None

        agent = self.scheduler.get_next_agent()
        if agent is None:
            return None
        return agent.step(self.context())

    def run(self, steps: int) -> List[Optional[str]]:
        return [self.step() for _ in range(steps)]

    def snapshot(self) -> SimulationSnapshot:
        return create_snapshot(
            environment_state=dict(self.state),
            agent_states=(agent.conversation_state for agent in self.agents),
        )

    def restore(self, snapshot: SimulationSnapshot) -> None:
        self.state = dict(snapshot.environment_state)
        name_to_state = {
            name: ConversationState.from_dict(state)
            for name, state in snapshot.agent_states.items()
        }
        for agent in self.agents:
            if agent.name in name_to_state:
                agent.set_conversation_state(name_to_state[agent.name])


__all__ = ["Environment"]
