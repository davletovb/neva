"""Environment abstractions coordinating agent simulations."""

from __future__ import annotations

import logging
import threading
from time import perf_counter, sleep, time
from typing import Dict, List, Optional
from uuid import uuid4

from neva.agents.base import AIAgent
from neva.schedulers.base import Scheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.failures import FailureLog, FailureRecord
from neva.utils.recovery import RecoveryPolicy, RecoveryState
from neva.utils.state_management import (
    CheckpointLimits,
    ConversationState,
    SimulationSnapshot,
    create_snapshot,
)
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
        recovery_policy: Optional[RecoveryPolicy] = None,
    ) -> None:
        if error_policy not in {"raise", "return"}:
            raise ValueError("error_policy must be 'raise' or 'return'")
        if failure_log is not None and not isinstance(failure_log, FailureLog):
            raise ValueError("failure_log must be a FailureLog instance")
        if recovery_policy is not None and not isinstance(recovery_policy, RecoveryPolicy):
            raise ValueError("recovery_policy must be a RecoveryPolicy instance")
        self.state: Dict[str, object] = {}
        self.scheduler = scheduler
        self.error_policy = error_policy
        self.error_value = error_value
        self.failure_log = failure_log
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self._recovery = RecoveryState()
        self._recovery_lock = threading.Lock()
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

        recovery = self.recovery_policy
        agent: Optional[AIAgent] = None
        for attempt in range(1, recovery.max_attempts + 1):
            try:
                agent = self.scheduler.get_next_agent()
                break
            except SchedulingError as exc:
                retry = recovery.should_retry(exc, attempt)
                policy = (
                    self.error_policy if recovery.escalation == "inherit" else recovery.escalation
                )
                action = "retry" if retry else policy
                wrote = self._record_failure(
                    agent_name=None,
                    exc=exc,
                    context=None,
                    policy=policy,
                    attempt=attempt,
                    max_attempts=recovery.max_attempts,
                    action=action,
                )
                self._note_recovery(
                    exc=exc,
                    agent_name=None,
                    action=action,
                    record_written=wrote,
                    retry=retry,
                    exhausted=(
                        recovery.max_retries > 0
                        and not retry
                        and attempt >= recovery.max_attempts
                        and recovery.is_retryable(exc)
                    ),
                    escalated=not retry,
                )
                if retry:
                    delay = recovery.delay_for(attempt)
                    if delay:
                        sleep(delay)
                    continue
                if policy == "return":
                    logger.debug("Scheduler failed to select an agent: %s", exc)
                    return self.error_value
                raise

        if agent is None:
            return None
        if attempt > 1:
            with self._recovery_lock:
                self._recovery.recoveries_succeeded += 1
                self._recovery.last_action = "recovered"
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
        """Run one turn with optional automatic retry and final escalation."""

        scheduler = self.scheduler
        if scheduler is None:  # pragma: no cover - step()/replay_failure() guard this.
            return None

        started = perf_counter()
        recovery = self.recovery_policy
        resolved_context = context
        response = ""

        for attempt in range(1, recovery.max_attempts + 1):
            try:
                if resolved_context is None:
                    resolved_context = self.context()
                response = agent.step(resolved_context)
            except Exception as exc:
                retry = recovery.should_retry(exc, attempt)
                inherited = self._effective_error_policy(agent)
                policy = inherited if recovery.escalation == "inherit" else recovery.escalation
                action = "retry" if retry else policy
                wrote = self._record_failure(
                    agent_name=agent.name,
                    exc=exc,
                    context=resolved_context,
                    policy=policy,
                    attempt=attempt,
                    max_attempts=recovery.max_attempts,
                    action=action,
                )
                self._note_recovery(
                    exc=exc,
                    agent_name=agent.name,
                    action=action,
                    record_written=wrote,
                    retry=retry,
                    exhausted=(
                        recovery.max_retries > 0
                        and not retry
                        and attempt >= recovery.max_attempts
                        and recovery.is_retryable(exc)
                    ),
                    escalated=not retry,
                )
                if retry:
                    delay = recovery.delay_for(attempt)
                    if delay:
                        sleep(delay)
                    continue

                scheduler.record_metrics(agent, status="failed", error=repr(exc))
                if policy == "return":
                    return self._effective_error_value(agent)
                raise
            break

        try:
            self.on_turn_complete(response)
        except Exception as exc:
            inherited = self._effective_error_policy(agent)
            policy = inherited if recovery.escalation == "inherit" else recovery.escalation
            wrote = self._record_failure(
                agent_name=agent.name,
                exc=exc,
                context=resolved_context,
                policy=policy,
                attempt=attempt,
                max_attempts=recovery.max_attempts,
                action=policy,
            )
            self._note_recovery(
                exc=exc,
                agent_name=agent.name,
                action=policy,
                record_written=wrote,
                retry=False,
                exhausted=False,
                escalated=True,
            )
            scheduler.record_metrics(agent, status="failed", error=repr(exc))
            if policy == "return":
                return self._effective_error_value(agent)
            raise

        if attempt > 1:
            with self._recovery_lock:
                self._recovery.recoveries_succeeded += 1
                self._recovery.last_action = "recovered"
        scheduler.record_metrics(
            agent,
            status="completed",
            latency=perf_counter() - started,
            response=response,
        )
        return response

    def _effective_error_policy(self, agent: AIAgent) -> str:
        overrides = getattr(self, "_agent_error_policies", {})
        override = overrides.get(str(agent.id))
        if override is not None and override.get("policy") is not None:
            return str(override["policy"])
        return self.error_policy

    def _effective_error_value(self, agent: AIAgent) -> Optional[str]:
        overrides = getattr(self, "_agent_error_policies", {})
        override = overrides.get(str(agent.id))
        if override is not None and override.get("policy") == "return":
            return override.get("value")
        return self.error_value

    def _note_recovery(
        self,
        *,
        exc: BaseException,
        agent_name: Optional[str],
        action: str,
        record_written: bool,
        retry: bool,
        exhausted: bool,
        escalated: bool,
    ) -> None:
        with self._recovery_lock:
            self._recovery.failures_seen += 1
            self._recovery.retries_attempted += int(retry)
            self._recovery.retries_exhausted += int(exhausted)
            self._recovery.escalations += int(escalated)
            self._recovery.failure_records_written += int(record_written)
            self._recovery.last_action = action
            self._recovery.last_error_type = type(exc).__name__
            self._recovery.last_error_message = str(exc)
            self._recovery.last_agent_name = agent_name

    def recovery_state(self) -> Dict[str, object]:
        """Return a consistent snapshot of automatic recovery counters."""

        with self._recovery_lock:
            return dict(self._recovery.to_dict())

    def _record_failure(
        self,
        *,
        agent_name: Optional[str],
        exc: BaseException,
        context: Optional[str],
        policy: str,
        attempt: int = 1,
        max_attempts: int = 1,
        action: Optional[str] = None,
    ) -> bool:
        """Append a durable failure record; logging problems never break a run."""

        failure_log = getattr(self, "failure_log", None)
        if failure_log is None:
            return False
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
                    attempt=attempt,
                    max_attempts=max_attempts,
                    action=action or policy,
                )
            )
            return True
        except Exception:
            logger.warning("Failed to record turn failure", exc_info=True)
            return False

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

    def snapshot(
        self,
        *,
        limits: Optional[CheckpointLimits] = None,
    ) -> SimulationSnapshot:
        from neva.utils.checkpoint import capture_runtime

        snapshot = create_snapshot(
            environment_state=self.state,
            agent_states=(agent.conversation_state for agent in self.agents),
            limits=limits,
        )
        snapshot.version = 2
        snapshot.runtime_state = capture_runtime(self, limits=limits)
        snapshot.validate_limits(limits)
        return snapshot

    def restore(
        self,
        snapshot: SimulationSnapshot,
        *,
        limits: Optional[CheckpointLimits] = None,
    ) -> None:
        from copy import deepcopy

        from neva.utils.checkpoint import restore_runtime

        snapshot.validate_limits(limits)
        if snapshot.version == 2:
            # The complete snapshot was just validated, so avoid walking the
            # runtime subtree a second time during environment restore.
            restore_runtime(
                self,
                snapshot.runtime_state,
                limits=limits,
                validate_limits=False,
            )
        elif snapshot.version != 1:
            raise ValueError(f"Unsupported snapshot version: {snapshot.version}")

        # The graph has already been checked against the optional envelope, so
        # this isolated environment-state copy cannot grow beyond that bound.
        self.state = deepcopy(snapshot.environment_state)
        name_to_state = {
            name: ConversationState.from_dict(state)
            for name, state in snapshot.agent_states.items()
        }
        for agent in self.agents:
            if agent.name in name_to_state:
                agent.set_conversation_state(name_to_state[agent.name])


__all__ = ["Environment"]
