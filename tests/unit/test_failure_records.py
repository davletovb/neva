import json
import threading

import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.environments.basic import BasicEnvironment
from neva.schedulers import RoundRobinScheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.failures import FailureLog, FailureRecord
from neva.utils.state_management import SimulationSnapshot


def fail(prompt):
    raise RuntimeError("provider down")


class FailingScheduler(RoundRobinScheduler):
    def get_next_agent(self):
        raise SchedulingError("no agents available")


class DriftingEnvironment(BasicEnvironment):
    """Context changes with state, so recorded vs current context differ."""

    def context(self):
        return f"context-{self.state.get('version', 0)}"


def make_record(**overrides):
    defaults = dict(
        timestamp=1234.5,
        environment="Environment",
        error_type="RuntimeError",
        error_message="boom",
        policy="return",
        conversation_id="conversation-x",
        agent_name="alice",
    )
    defaults.update(overrides)
    return FailureRecord(**defaults)


def encoded_record_size(record):
    line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
    return len(line.encode("utf-8"))


def test_record_round_trip():
    record = make_record(context="prompt text")
    assert FailureRecord.from_dict(record.to_dict()) == record


def test_log_persists_across_instances(tmp_path):
    path = tmp_path / "nested" / "failures.jsonl"
    first = FailureLog(path)
    first.append(make_record())
    first.append(make_record(error_type="ValueError", error_message="second"))
    loaded = FailureLog(path).load()
    assert len(loaded) == 2
    assert loaded[0] == make_record()
    assert loaded[1].error_type == "ValueError"


def test_load_missing_file_returns_empty(tmp_path):
    assert FailureLog(tmp_path / "absent.jsonl").load() == []


def test_load_skips_malformed_lines(tmp_path):
    path = tmp_path / "failures.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(make_record().to_dict()),
                "not json",
                '{"timestamp": 1.0, "environment": "E"',  # truncated tail
                '{"timestamp": 1.0}',  # valid JSON, missing required fields
                "",
                json.dumps(make_record(agent_name="bob").to_dict()),
            ]
        ),
        encoding="utf-8",
    )
    loaded = FailureLog(path).load()
    assert [record.agent_name for record in loaded] == ["alice", "bob"]


def test_load_survives_undecodable_bytes(tmp_path):
    path = tmp_path / "failures.jsonl"
    path.write_bytes(
        json.dumps(make_record().to_dict()).encode()
        + b"\n"
        + b"\xff\xfe partial \n"
        + json.dumps(make_record(agent_name="bob").to_dict()).encode()
        + b"\n"
    )
    loaded = FailureLog(path).load()
    assert [record.agent_name for record in loaded] == ["alice", "bob"]


def test_append_after_torn_tail_keeps_record(tmp_path):
    path = tmp_path / "failures.jsonl"
    path.write_bytes(b'{"timestamp": 1.0')  # crash mid-write, no newline
    log = FailureLog(path)
    log.append(make_record())
    loaded = log.load()
    assert [record.agent_name for record in loaded] == ["alice"]
    assert b'{"timestamp": 1.0\n' in path.read_bytes()  # tail stays separate


def test_context_omitted_unless_opted_in(tmp_path):
    private = FailureLog(tmp_path / "private.jsonl")
    private.append(make_record(context="secret prompt"))
    assert private.load()[0].context is None

    opt_in = FailureLog(tmp_path / "opt-in.jsonl", include_context=True)
    opt_in.append(make_record(context="secret prompt"))
    assert opt_in.load()[0].context == "secret prompt"


def test_concurrent_appends_are_not_interleaved(tmp_path):
    path = tmp_path / "failures.jsonl"
    log = FailureLog(path)

    def worker(worker_id):
        for index in range(25):
            log.append(make_record(agent_name=f"w{worker_id}-{index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    loaded = log.load()
    assert len(loaded) == 200
    assert len({record.agent_name for record in loaded}) == 200


def test_invalid_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="path"):
        FailureLog(42)
    with pytest.raises(ValueError, match="directory"):
        FailureLog(tmp_path)
    with pytest.raises(ValueError, match="directory"):
        FailureLog("")


@pytest.mark.parametrize(
    "timestamp",
    [float("nan"), float("inf"), -float("inf"), -1.0, True, "soon", None],
)
def test_from_dict_rejects_invalid_timestamps(timestamp):
    payload = make_record().to_dict()
    payload["timestamp"] = timestamp
    with pytest.raises(ValueError, match="timestamp"):
        FailureRecord.from_dict(payload)


def test_load_skips_invalid_policy_and_timestamp(tmp_path):
    path = tmp_path / "failures.jsonl"
    bad_policy = make_record().to_dict()
    bad_policy["policy"] = "ignore"
    bad_timestamp = make_record().to_dict()
    bad_timestamp["timestamp"] = "soon"
    bad_nan = make_record().to_dict()
    bad_nan["timestamp"] = float("nan")
    bad_inf = make_record().to_dict()
    bad_inf["timestamp"] = float("inf")
    path.write_text(
        "\n".join(
            [
                json.dumps(bad_policy),
                json.dumps(bad_timestamp),
                json.dumps(bad_nan),
                json.dumps(bad_inf),
                json.dumps(make_record().to_dict()),
            ]
        ),
        encoding="utf-8",
    )
    loaded = FailureLog(path).load()
    assert [record.agent_name for record in loaded] == ["alice"]


def test_environment_rejects_non_log():
    with pytest.raises(ValueError, match="FailureLog"):
        Environment(RoundRobinScheduler(), failure_log=object())


def test_environment_records_returned_failure(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = BasicEnvironment(
        name="lab",
        description="test",
        scheduler=RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    assert env.step() == "offline"
    records = log.load()
    assert len(records) == 1
    record = records[0]
    assert record.agent_name == "alice"
    assert record.error_type == "RuntimeError"
    assert record.error_message == "provider down"
    assert record.policy == "return"
    assert record.environment == "BasicEnvironment"
    assert record.conversation_id == env.conversation_id
    assert record.context is None  # privacy default
    assert record.timestamp > 0


def test_environment_records_raised_failure(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(RoundRobinScheduler(), failure_log=log)
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    with pytest.raises(RuntimeError, match="provider down"):
        env.step()
    records = log.load()
    assert len(records) == 1
    assert records[0].policy == "raise"
    assert records[0].agent_name == "alice"


def test_per_agent_override_recorded(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(RoundRobinScheduler(), failure_log=log)
    env.register_agent(
        TransformerAgent(name="alice", llm_backend=fail),
        error_policy="return",
        error_value="handled",
    )
    assert env.step() == "handled"
    assert log.load()[0].policy == "return"


def test_scheduler_selection_failure_recorded(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(
        FailingScheduler(), error_policy="return", error_value="offline", failure_log=log
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    assert env.step() == "offline"
    records = log.load()
    assert len(records) == 1
    assert records[0].agent_name is None
    assert records[0].error_type == "SchedulingError"
    assert records[0].policy == "return"


@pytest.mark.parametrize("policy", ["return", "raise"])
def test_logging_failure_does_not_mask_turn_outcome(tmp_path, monkeypatch, policy):
    log = FailureLog(tmp_path / "failures.jsonl")

    def exploding_append(record):
        raise OSError("disk full")

    monkeypatch.setattr(log, "append", exploding_append)
    env = Environment(
        RoundRobinScheduler(), error_policy=policy, error_value="offline", failure_log=log
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    if policy == "return":
        assert env.step() == "offline"
    else:
        with pytest.raises(RuntimeError, match="provider down"):
            env.step()


def test_context_recorded_when_opted_in(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl", include_context=True)
    env = BasicEnvironment(
        name="lab",
        description="test",
        scheduler=RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    assert env.step() == "offline"
    assert log.load()[0].context == "This is a lab. test"


def test_replay_failure_returns_response_after_recovery(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = BasicEnvironment(
        name="lab",
        description="test",
        scheduler=RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
    )
    prompts = []
    attempts = {"n": 0}

    def flaky(prompt):
        prompts.append(prompt)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("provider down")
        return "recovered"

    env.register_agent(TransformerAgent(name="alice", llm_backend=flaky))
    assert env.step() == "offline"
    record = log.load()[0]
    statuses = []

    class Recorder:
        def collect_data(self, *args, **kwargs):
            statuses.append(kwargs.get("status"))

    env.scheduler.simulation_observer = Recorder()
    assert env.replay_failure(record) == "recovered"
    assert len(log.load()) == 1  # no new failure recorded
    assert "completed" in statuses
    assert "This is a lab. test" in prompts[-1]


def test_replay_failure_uses_recorded_context(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl", include_context=True)
    env = DriftingEnvironment(
        name="lab",
        description="test",
        scheduler=RoundRobinScheduler(),
        error_policy="return",
        error_value="offline",
        failure_log=log,
    )
    prompts = []

    def recording_backend(prompt):
        prompts.append(prompt)
        raise RuntimeError("provider down")

    env.register_agent(TransformerAgent(name="alice", llm_backend=recording_backend))
    assert env.step() == "offline"
    record = log.load()[0]
    assert record.context == "context-0"
    env.state["version"] = 1
    assert env.replay_failure(record) == "offline"  # agent still fails
    assert "context-0" in prompts[-1]  # recorded context, not current "context-1"
    assert "context-1" not in prompts[-1]


def test_replay_failure_records_repeated_failure(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(
        RoundRobinScheduler(), error_policy="return", error_value="offline", failure_log=log
    )
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    assert env.step() == "offline"
    record = log.load()[0]
    assert env.replay_failure(record) == "offline"
    records = log.load()
    assert len(records) == 2
    assert records[1].agent_name == "alice"


def test_failure_recorded_even_if_observer_raises(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(RoundRobinScheduler(), failure_log=log)
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))

    class ExplodingObserver:
        def collect_data(self, *args, **kwargs):
            if kwargs.get("status") == "failed":
                raise RuntimeError("observer bug")

    env.scheduler.simulation_observer = ExplodingObserver()
    with pytest.raises(RuntimeError, match="observer bug"):
        env.step()
    # The durable record must exist even though failed-turn metrics blew up.
    assert [record.agent_name for record in log.load()] == ["alice"]


def test_replay_raises_when_nothing_can_be_resolved():
    empty = Environment(RoundRobinScheduler())  # no agents registered
    with pytest.raises(ValueError, match="no registered agents"):
        empty.replay_failure(make_record())
    bare = Environment()  # no scheduler at all
    with pytest.raises(ValueError, match="scheduler"):
        bare.replay_failure(make_record())


def test_snapshot_round_trip_preserves_failure_log(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    env = Environment(RoundRobinScheduler(), failure_log=log)
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    snapshot = SimulationSnapshot.from_json(env.snapshot().to_json())

    restored_log = FailureLog(tmp_path / "restored.jsonl")
    restored = Environment(RoundRobinScheduler(), failure_log=restored_log)
    restored.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    restored.restore(snapshot)
    # The log is external durable storage: restore must keep the environment's
    # own instance rather than deleting or replacing it.
    assert restored.failure_log is restored_log


def test_replay_failure_requires_matching_agent():
    env = Environment(RoundRobinScheduler(), error_policy="return", error_value="offline")
    env.register_agent(TransformerAgent(name="alice", llm_backend=fail))
    with pytest.raises(ValueError, match="ghost"):
        env.replay_failure(make_record(agent_name="ghost"))
    with pytest.raises(ValueError, match="agent_name"):
        env.replay_failure(make_record(agent_name=None))
    agent = TransformerAgent(name="bob", llm_backend=fail)
    env.register_agent(agent)
    # An explicit agent bypasses the name lookup but must be registered.
    assert env.replay_failure(make_record(agent_name=None), agent=agent) == "offline"
    stranger = TransformerAgent(name="carol", llm_backend=fail)
    with pytest.raises(ValueError, match="not registered"):
        env.replay_failure(make_record(agent_name="alice"), agent=stranger)


def test_replay_nameless_record_validated_before_empty_environment():
    env = Environment(RoundRobinScheduler())
    with pytest.raises(ValueError, match="agent_name"):
        env.replay_failure(make_record(agent_name=None))



@pytest.mark.parametrize("rotate_bytes", [0, -1, True, 1.5, "100"])
def test_invalid_rotation_threshold_rejected(tmp_path, rotate_bytes):
    with pytest.raises(ValueError, match="rotate_bytes"):
        FailureLog(tmp_path / "failures.jsonl", rotate_bytes=rotate_bytes)


@pytest.mark.parametrize("backup_count", [-1, True, 1.5, "2"])
def test_invalid_backup_count_rejected(tmp_path, backup_count):
    with pytest.raises(ValueError, match="backup_count"):
        FailureLog(tmp_path / "failures.jsonl", rotate_bytes=100, backup_count=backup_count)


def test_rotation_retains_records_and_loads_oldest_first(tmp_path):
    path = tmp_path / "failures.jsonl"
    records = [make_record(agent_name=name) for name in ("a", "b", "c")]
    threshold = encoded_record_size(records[0]) + encoded_record_size(records[1])
    log = FailureLog(path, rotate_bytes=threshold, backup_count=2)

    for record in records:
        log.append(record)

    assert (tmp_path / "failures.jsonl.1").exists()
    assert path.exists()
    assert [record.agent_name for record in log.load()] == ["a", "b", "c"]


def test_rotation_discards_oldest_backup_after_retention_limit(tmp_path):
    path = tmp_path / "failures.jsonl"
    records = [make_record(agent_name=name) for name in ("a", "b", "c", "d")]
    threshold = encoded_record_size(records[0])
    log = FailureLog(path, rotate_bytes=threshold, backup_count=2)

    for record in records:
        log.append(record)

    assert [record.agent_name for record in log.load()] == ["b", "c", "d"]
    assert (tmp_path / "failures.jsonl.1").exists()
    assert (tmp_path / "failures.jsonl.2").exists()
    assert not (tmp_path / "failures.jsonl.3").exists()


def test_zero_backup_count_keeps_only_active_file(tmp_path):
    path = tmp_path / "failures.jsonl"
    first = make_record(agent_name="a")
    second = make_record(agent_name="b")
    log = FailureLog(
        path,
        rotate_bytes=encoded_record_size(first),
        backup_count=0,
    )

    log.append(first)
    log.append(second)

    assert [record.agent_name for record in log.load()] == ["b"]
    assert not (tmp_path / "failures.jsonl.1").exists()


def test_single_oversized_record_is_preserved_intact(tmp_path):
    path = tmp_path / "failures.jsonl"
    first = make_record(agent_name="a", error_message="x" * 500)
    second = make_record(agent_name="b", error_message="y" * 500)
    log = FailureLog(path, rotate_bytes=64, backup_count=1)

    log.append(first)
    assert path.stat().st_size > 64

    log.append(second)

    assert [record.agent_name for record in log.load()] == ["a", "b"]
    assert path.stat().st_size > 64
    assert (tmp_path / "failures.jsonl.1").exists()


def test_rotated_log_can_be_reopened_with_same_policy(tmp_path):
    path = tmp_path / "failures.jsonl"
    first = make_record(agent_name="a")
    second = make_record(agent_name="b")
    threshold = encoded_record_size(first)
    FailureLog(path, rotate_bytes=threshold, backup_count=1).append(first)
    FailureLog(path, rotate_bytes=threshold, backup_count=1).append(second)

    reopened = FailureLog(path, rotate_bytes=threshold, backup_count=1)

    assert [record.agent_name for record in reopened.load()] == ["a", "b"]
