import json
import multiprocessing
import threading

import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.schedulers import RoundRobinScheduler
from neva.utils.exceptions import (
    CircuitOpenError,
    RateLimiterCancelledError,
    SchedulingError,
)
from neva.utils.failures import FailureLog, FailureRecord
from neva.utils.recovery import RecoveryPolicy
from neva.utils.safety import CircuitBreaker


class FlakyScheduler(RoundRobinScheduler):
    def __init__(self, failures=1):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def get_next_agent(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise SchedulingError("scheduler unavailable")
        return super().get_next_agent()


def _record(name, *, message="boom", context=None):
    return FailureRecord(
        timestamp=1.0,
        environment="Environment",
        error_type="RuntimeError",
        error_message=message,
        policy="return",
        conversation_id="conversation-test",
        agent_name=name,
        context=context,
        action="return",
    )


def _rotating_failure_writer(path, prefix, count, rotate_bytes, backup_count):
    log = FailureLog(
        path,
        rotate_bytes=rotate_bytes,
        backup_count=backup_count,
        fsync=False,
    )
    for index in range(count):
        log.append(_record(f"{prefix}-{index}", message="x" * 80))


def test_agent_failure_recovers_with_retry_and_observable_state(tmp_path):
    calls = []

    def backend(prompt):
        calls.append(prompt)
        if len(calls) < 3:
            raise RuntimeError("temporary")
        return "recovered"

    log = FailureLog(tmp_path / "failures.jsonl", include_context=True)
    env = Environment(
        RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
        recovery_policy=RecoveryPolicy(max_retries=2),
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=backend))

    assert env.step() == "recovered"
    assert len(calls) == 3

    records = log.load()
    assert [record.attempt for record in records] == [1, 2]
    assert [record.action for record in records] == ["retry", "retry"]
    assert all(record.max_attempts == 3 for record in records)

    state = env.recovery_state()
    assert state["failures_seen"] == 2
    assert state["retries_attempted"] == 2
    assert state["recoveries_succeeded"] == 1
    assert state["retries_exhausted"] == 0
    assert state["escalations"] == 0
    assert state["failure_records_written"] == 2
    assert state["last_action"] == "recovered"


def test_exhausted_retry_escalates_to_existing_return_policy(tmp_path):
    calls = []

    def backend(prompt):
        calls.append(prompt)
        raise RuntimeError("still down")

    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(
        RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
        recovery_policy=RecoveryPolicy(max_retries=2),
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=backend))

    assert env.step() == "offline"
    assert len(calls) == 3
    records = log.load()
    assert [record.action for record in records] == ["retry", "retry", "return"]
    assert [record.attempt for record in records] == [1, 2, 3]

    state = env.recovery_state()
    assert state["retries_exhausted"] == 1
    assert state["escalations"] == 1
    assert state["last_action"] == "return"


def test_recovery_policy_can_force_raise_after_retry():
    calls = []

    def backend(prompt):
        calls.append(prompt)
        raise RuntimeError("fatal")

    env = Environment(
        RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        recovery_policy=RecoveryPolicy(max_retries=1, escalation="raise"),
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=backend))

    with pytest.raises(RuntimeError, match="fatal"):
        env.step()
    assert len(calls) == 2
    state = env.recovery_state()
    assert state["retries_attempted"] == 1
    assert state["retries_exhausted"] == 1
    assert state["escalations"] == 1
    assert state["last_action"] == "raise"


def test_scheduler_selection_uses_same_recovery_policy(tmp_path):
    scheduler = FlakyScheduler(failures=1)
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(
        scheduler,
        failure_log=log,
        recovery_policy=RecoveryPolicy(max_retries=1),
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=lambda prompt: "ok"))

    assert env.step() == "ok"
    assert scheduler.calls == 2
    records = log.load()
    assert len(records) == 1
    assert records[0].agent_name is None
    assert records[0].action == "retry"
    state = env.recovery_state()
    assert state["retries_attempted"] == 1
    assert state["recoveries_succeeded"] == 1


def test_recovery_backoff_is_exponential(monkeypatch):
    calls = []
    delays = []

    def backend(prompt):
        calls.append(prompt)
        if len(calls) < 3:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr("neva.environments.base.sleep", delays.append)
    env = Environment(
        RoundRobinScheduler(),
        recovery_policy=RecoveryPolicy(
            max_retries=2,
            backoff=0.25,
            backoff_multiplier=3.0,
        ),
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=backend))

    assert env.step() == "ok"
    assert delays == [0.25, 0.75]


def test_deliberate_cancellation_is_never_retried():
    calls = []

    def backend(prompt):
        calls.append(prompt)
        raise RateLimiterCancelledError("cancelled")

    env = Environment(
        RoundRobinScheduler(),
        error_policy="return",
        error_value="cancelled",
        recovery_policy=RecoveryPolicy(max_retries=5),
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=backend))

    assert env.step() == "cancelled"
    assert len(calls) == 1
    state = env.recovery_state()
    assert state["retries_attempted"] == 0
    assert state["escalations"] == 1


def test_failure_log_writes_physical_jsonl_newlines(tmp_path):
    path = tmp_path / "failures.jsonl"
    log = FailureLog(path, fsync=False)
    log.append(_record("alice"))
    log.append(_record("bob"))

    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert len(raw.splitlines()) == 2
    assert [json.loads(line)["agent_name"] for line in raw.splitlines()] == [
        "alice",
        "bob",
    ]


def test_failure_log_enforces_utf8_record_ceiling(tmp_path):
    path = tmp_path / "failures.jsonl"
    baseline = _record("alice", message="", context=None)
    baseline_bytes = len((json.dumps(baseline.to_dict(), sort_keys=True) + "\n").encode("utf-8"))
    limit = baseline_bytes + 45

    log = FailureLog(
        path,
        include_context=True,
        max_record_bytes=limit,
        fsync=False,
    )
    log.append(_record("alice", message="💥" * 300, context="secret " * 200))

    assert path.stat().st_size <= limit
    loaded = log.load()
    assert len(loaded) == 1
    assert loaded[0].truncated is True
    assert loaded[0].context is None
    assert loaded[0].error_message.endswith("...[truncated]")


def test_failure_log_rejects_structural_metadata_that_cannot_fit(tmp_path):
    path = tmp_path / "failures.jsonl"
    log = FailureLog(path, max_record_bytes=32, fsync=False)

    with pytest.raises(ValueError, match="metadata exceeds"):
        log.append(_record("alice", message="x" * 100))
    assert not path.exists()


def test_failure_log_rotation_is_coordinated_across_processes(tmp_path):
    path = tmp_path / "failures.jsonl"
    rotate_bytes = 1400
    backup_count = 40
    count = 30
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_rotating_failure_writer,
            args=(str(path), prefix, count, rotate_bytes, backup_count),
        )
        for prefix in ("a", "b", "c")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        assert process.exitcode == 0

    records = FailureLog(
        path,
        rotate_bytes=rotate_bytes,
        backup_count=backup_count,
        fsync=False,
    ).load()
    names = [record.agent_name for record in records]
    assert len(names) == count * 3
    assert len(set(names)) == count * 3


def test_failure_record_backward_compatibility_defaults_recovery_metadata():
    payload = _record("alice").to_dict()
    payload.pop("attempt")
    payload.pop("max_attempts")
    payload.pop("action")
    payload.pop("truncated")

    restored = FailureRecord.from_dict(payload)
    assert restored.attempt == 1
    assert restored.max_attempts == 1
    assert restored.action == restored.policy
    assert restored.truncated is False


def test_circuit_breaker_allows_only_one_concurrent_half_open_probe():
    breaker = CircuitBreaker(failure_threshold=1, cooldown=0.0)
    breaker.record_failure()
    barrier = threading.Barrier(12)
    allowed = []
    refused = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        try:
            breaker.allow()
        except CircuitOpenError:
            with lock:
                refused.append(True)
        else:
            with lock:
                allowed.append(True)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert len(allowed) == 1
    assert len(refused) == 11

    # Treat the admitted probe as interrupted/non-provider rejection. The
    # breaker must release it and permit a later probe rather than deadlock.
    breaker.record_rejected()
    breaker.allow()
    breaker.record_success()
    breaker.allow()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"backoff": -1},
        {"backoff_multiplier": 0.5},
        {"escalation": "ignore"},
        {"retry_on": ()},
    ],
)
def test_recovery_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RecoveryPolicy(**kwargs)


def test_context_failure_stays_inside_recovery_dispatch(tmp_path):
    class FlakyContextEnvironment(Environment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context_calls = 0

        def context(self):
            self.context_calls += 1
            if self.context_calls == 1:
                raise RuntimeError("context unavailable")
            return "stable context"

    prompts = []
    log = FailureLog(tmp_path / "failures.jsonl", include_context=True)
    env = FlakyContextEnvironment(
        RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
        recovery_policy=RecoveryPolicy(max_retries=1),
    )
    env.register_agent(
        TransformerAgent(name="alice", llm_backend=lambda prompt: prompts.append(prompt) or "ok")
    )

    assert env.step() == "ok"
    assert env.context_calls == 2
    assert len(prompts) == 1
    record = log.load()[0]
    assert record.action == "retry"
    assert record.context is None


def test_completion_hook_failure_does_not_rerun_agent(tmp_path):
    class ExplodingHookEnvironment(Environment):
        def on_turn_complete(self, response):
            raise RuntimeError("hook failed")

    calls = []
    log = FailureLog(tmp_path / "failures.jsonl")
    env = ExplodingHookEnvironment(
        RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
        recovery_policy=RecoveryPolicy(max_retries=4),
    )
    env.register_agent(
        TransformerAgent(name="alice", llm_backend=lambda prompt: calls.append(prompt) or "done")
    )

    assert env.step() == "offline"
    assert len(calls) == 1
    records = log.load()
    assert len(records) == 1
    assert records[0].error_message == "hook failed"
    assert records[0].action == "return"
    assert env.recovery_state()["retries_attempted"] == 0


def test_checkpoint_excludes_recovery_runtime_objects(tmp_path):
    policy = RecoveryPolicy(max_retries=2, backoff=0.1)
    env = Environment(RoundRobinScheduler(), recovery_policy=policy)
    env.register_agent(TransformerAgent(name="alice", llm_backend=lambda prompt: "ok"))

    snapshot = env.snapshot()
    raw = snapshot.to_json()
    assert "RecoveryPolicy" not in raw
    assert "_recovery_lock" not in raw

    restored = Environment(
        RoundRobinScheduler(),
        recovery_policy=policy,
        failure_log=FailureLog(tmp_path / "restored.jsonl"),
    )
    restored.register_agent(TransformerAgent(name="alice", llm_backend=lambda prompt: "ok"))
    restored.restore(snapshot)

    assert restored.recovery_policy is policy
    assert restored.recovery_state()["failures_seen"] == 0
