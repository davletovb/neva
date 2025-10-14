"""Environment abstractions coordinating agent simulations."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from uuid import uuid4

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.state_management import ConversationState, SimulationSnapshot, create_snapshot
from neva.utils.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class Environment:
    """Coordinate agents and schedulers while maintaining shared state."""

    def __init__(self, scheduler: Optional[Scheduler] = None) -> None:
        self.state: Dict[str, object] = {}
        self.scheduler = scheduler
        self.agents: List[AIAgent] = []
        self.conversation_id = f"conversation-{uuid4()}"
        if self.scheduler is not None:
            self.scheduler.set_environment(self)

    def register_agent(self, agent: AIAgent) -> None:
        agent.set_environment(self)
        self.agents.append(agent)
        if self.scheduler is not None:
            self.scheduler.add(agent)
        telemetry = get_telemetry()
        if telemetry is not None:
            try:
                telemetry.record_agent_registration(
                    conversation_id=self.conversation_id,
                    agent_name=agent.name,
                    attributes={"environment.class": self.__class__.__name__},
                )
            except Exception:  # pragma: no cover - telemetry failures should not break execution.
                logger.debug("Failed to emit agent registration telemetry", exc_info=True)

    def context(self) -> str:
        """Return a textual description of the environment state."""

        return ""

    def step(self) -> Optional[str]:
        if self.scheduler is None or not self.agents:
            return None

        agent = self.scheduler.get_next_agent()
        if agent is None:
            return None
        telemetry = get_telemetry()
        if telemetry is not None:
            try:
                telemetry.record_scheduler_decision(
                    conversation_id=self.conversation_id,
                    scheduler_name=self.scheduler.__class__.__name__,
                    agent_name=agent.name,
                )
            except Exception:  # pragma: no cover - telemetry failures should not break execution.
                logger.debug("Failed to emit scheduler telemetry", exc_info=True)
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
