"""Environment abstractions coordinating agent simulations."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, List, Optional
from uuid import uuid4

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.state_management import ConversationState, SimulationSnapshot, create_snapshot
from neva.utils.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class Environment:
    """Coordinate agents and schedulers while maintaining shared state."""

    def __init__(
        self,
        scheduler: Optional[Scheduler] = None,
        *,
        error_policy: str = "raise",
        error_value: Optional[str] = None,
    ) -> None:
        if error_policy not in {"raise", "return"}:
            raise ValueError("error_policy must be 'raise' or 'return'")
        self.state: Dict[str, object] = {}
        self.scheduler = scheduler
        self.error_policy = error_policy
        self.error_value = error_value
        self._agent_error_policies: Dict[str, Dict[str, Optional[str]]] = {}
        self.agents: List[AIAgent] = []

        self.conversation_id = f"conversation-{uuid4()}"
        if self.scheduler is not None:
            self.scheduler.set_environment(self)

    def register_agent(
        self,
        agent: AIAgent,
        *,
        error_policy: Optional[str] = None,
        error_value: Optional[str] = None,
    ) -> None:
        """Register an agent with an optional turn-failure policy override.

        None inherits the environment policy and fallback. An explicit 'return'
        uses this registration's error_value (default None); 'raise' propagates.
        Overrides cover the selected turn, including context/completion hooks;
        scheduler-selection failures still use the environment policy. Version-2
        checkpoints preserve overrides with the agents' restored UUIDs.
        """
        if error_policy not in (None, "raise", "return"):
            raise ValueError("error_policy must be 'raise', 'return', or None")
        if error_policy is None and error_value is not None:
            raise ValueError("error_value requires an explicit error_policy")
        if error_policy == "raise" and error_value is not None:
            raise ValueError("error_value is unused with error_policy='raise'")
        if error_policy is not None:
            if not hasattr(self, "_agent_error_policies"):
                self._agent_error_policies = {}
            self._agent_error_policies[str(agent.id)] = {
                "policy": error_policy,
                "value": error_value,
            }
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

    def on_turn_complete(self, response: str) -> None:
        """Update transcript/state before metrics; override instead of wrapping step."""

    def step(self) -> Optional[str]:
        if self.scheduler is None or not self.agents:
            return None

        try:
            agent = self.scheduler.get_next_agent()
        except SchedulingError as exc:
            if self.error_policy == "return":
                logger.debug("Scheduler failed to select an agent: %s", exc)
                return self.error_value
            raise
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
        started = perf_counter()
        try:
            response = agent.step(self.context())
            self.on_turn_complete(response)
        except Exception as exc:
            self.scheduler.record_metrics(agent, status="failed", error=repr(exc))
            overrides = getattr(self, "_agent_error_policies", {})
            policy = overrides.get(str(agent.id))
            if policy is not None:
                if policy.get("policy") == "return":
                    return policy.get("value")
            elif self.error_policy == "return":
                return self.error_value
            raise
        self.scheduler.record_metrics(
            agent, status="completed", latency=perf_counter() - started, response=response
        )
        return response

    def run(self, steps: int) -> List[Optional[str]]:
        return [self.step() for _ in range(steps)]

    def snapshot(self) -> SimulationSnapshot:
        from neva.utils.checkpoint import capture_runtime

        snapshot = create_snapshot(
            environment_state=self.state,
            agent_states=(agent.conversation_state for agent in self.agents),
        )
        snapshot.version = 2
        snapshot.runtime_state = capture_runtime(self)
        return snapshot

    def restore(self, snapshot: SimulationSnapshot) -> None:
        from copy import deepcopy

        from neva.utils.checkpoint import restore_runtime

        if snapshot.version == 2:
            restore_runtime(self, snapshot.runtime_state)
        elif snapshot.version != 1:
            raise ValueError(f"Unsupported snapshot version: {snapshot.version}")
        self.state = deepcopy(snapshot.environment_state)
        name_to_state = {
            name: ConversationState.from_dict(state)
            for name, state in snapshot.agent_states.items()
        }
        for agent in self.agents:
            if agent.name in name_to_state:
                agent.set_conversation_state(name_to_state[agent.name])


__all__ = ["Environment"]
