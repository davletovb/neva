"""Environment abstractions coordinating agent simulations."""

from __future__ import annotations

import logging
from time import perf_counter, time
from typing import Dict, List, Optional
from uuid import uuid4

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.failures import FailureLog, FailureRecord
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
        failure_log: Optional[FailureLog] = None,
    ) -> None:
        if error_policy not in {"raise", "return"}:
            raise ValueError("error_policy must be 'raise' or 'return'")
        if failure_log is not None and not isinstance(failure_log, FailureLog):
            raise ValueError("failure_log must be a FailureLog instance")
        self.state: Dict[str, object] = {}
        self.scheduler = scheduler
        self.error_policy = error_policy
        self.error_value = error_value
        self.failure_log = failure_log
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
            self._record_failure(agent_name=None, exc=exc, context=None, policy=self.error_policy)
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
        return self._execute_turn(agent)

    def _execute_turn(self, agent: AIAgent, *, context: Optional[str] = None) -> Optional[str]:
        """Run one turn for ``agent`` with the standard failure dispatch."""

        scheduler = self.scheduler
        if scheduler is None:  # pragma: no cover - step()/replay_failure() guard this.
            return None
        started = perf_counter()
        try:
            if context is None:
                context = self.context()
            response = agent.step(context)
            self.on_turn_complete(response)
        except Exception as exc:
            # Record before observer metrics: a raising observer must not
            # prevent the durable record from being written.
            policy = self._effective_error_policy(agent)
            self._record_failure(agent_name=agent.name, exc=exc, context=context, policy=policy)
            scheduler.record_metrics(agent, status="failed", error=repr(exc))
            overrides = getattr(self, "_agent_error_policies", {})
            override = overrides.get(str(agent.id))
            if override is not None:
                if override.get("policy") == "return":
                    return override.get("value")
            elif self.error_policy == "return":
                return self.error_value
            raise
        scheduler.record_metrics(
            agent, status="completed", latency=perf_counter() - started, response=response
        )
        return response

    def _effective_error_policy(self, agent: AIAgent) -> str:
        overrides = getattr(self, "_agent_error_policies", {})
        override = overrides.get(str(agent.id))
        if override is not None and override.get("policy") is not None:
            return str(override["policy"])
        return self.error_policy

    def _record_failure(
        self,
        *,
        agent_name: Optional[str],
        exc: BaseException,
        context: Optional[str],
        policy: str,
    ) -> None:
        """Append a durable failure record; logging problems never break a run."""

        failure_log = getattr(self, "failure_log", None)
        if failure_log is None:
            return
        try:
            failure_log.append(
                FailureRecord(
                    timestamp=time(),
                    conversation_id=getattr(self, "conversation_id", None),
                    environment=self.__class__.__name__,
                    agent_name=agent_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    policy=policy,
                    context=context,
                )
            )
        except Exception:
            logger.warning("Failed to record turn failure", exc_info=True)

    def replay_failure(
        self, record: FailureRecord, *, agent: Optional[AIAgent] = None
    ) -> Optional[str]:
        """Re-dispatch a recorded failed turn through the normal turn path.

        Resolves the agent from ``record.agent_name`` (the first registered
        match) unless one is passed explicitly, in which case it must be
        registered with this environment. The recorded context is used when
        captured, otherwise the current environment context. The agent's
        error policy still applies, so a repeated failure is recorded again.

        Raises ``ValueError`` when the scheduler is unavailable, no agents
        are registered, or the record's agent cannot be resolved: unlike
        ``step()``, a replay that cannot run must not silently look like a
        return-policy result.
        """

        if not isinstance(record, FailureRecord):
            raise TypeError("record must be a FailureRecord")
        if agent is None and record.agent_name is None:
            raise ValueError("record has no agent_name; pass agent explicitly")
        if self.scheduler is None:
            raise ValueError("cannot replay a failure without a scheduler")
        if not self.agents:
            raise ValueError("no registered agents to replay against")
        if agent is None:
            matches = [
                candidate for candidate in self.agents if candidate.name == record.agent_name
            ]
            if not matches:
                raise ValueError(f"no registered agent named {record.agent_name!r}")
            agent = matches[0]
        elif agent not in self.agents:
            raise ValueError("agent is not registered with this environment")
        return self._execute_turn(agent, context=record.context)

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
