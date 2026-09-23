from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import pytest

import neva.utils.checkpoint as checkpoint
import neva.utils.state_management as state_management
from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.schedulers import RoundRobinScheduler
from neva.utils.state_management import (
    CheckpointLimits,
    ConversationState,
    SimulationSnapshot,
    create_snapshot,
    load_snapshot,
    save_snapshot,
)


def _limits(**overrides):
    values = {
        "max_depth": 20,
        "max_nodes": 10_000,
        "max_string_bytes": 100_000,
        "max_total_string_bytes": 1_000_000,
    }
    values.update(overrides)
    return CheckpointLimits(**values)


@pytest.mark.parametrize(
    "field",
    ["max_depth", "max_nodes", "max_string_bytes", "max_total_string_bytes"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_invalid_checkpoint_graph_limits_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        CheckpointLimits(**{field: value})


def test_create_snapshot_rejects_depth_before_environment_deepcopy(monkeypatch):
    nested = {"level": {"level": {"value": "x"}}}

    def fail_deepcopy(value):
        raise AssertionError("oversized graph must be rejected before deepcopy")

    monkeypatch.setattr(state_management, "deepcopy", fail_deepcopy)

    with pytest.raises(ValueError, match="max_depth"):
        create_snapshot(
            environment_state=nested,
            limits=_limits(max_depth=2),
        )


def test_create_snapshot_rejects_node_and_string_budgets():
    with pytest.raises(ValueError, match="max_nodes"):
        create_snapshot(
            environment_state={"items": list(range(20))},
            limits=_limits(max_nodes=5),
        )

    with pytest.raises(ValueError, match="max_string_bytes"):
        create_snapshot(
            environment_state={"message": "x" * 50},
            limits=_limits(max_string_bytes=10),
        )

    with pytest.raises(ValueError, match="max_total_string_bytes"):
        create_snapshot(
            environment_state={"a": "12345", "b": "67890"},
            limits=_limits(max_total_string_bytes=8),
        )


def test_load_preflights_large_string_before_json_parse(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    path.write_text(
        '{"created_at":"2026-09-23T00:00:00","environment_state":{"message":"'
        + ("x" * 1000)
        + '"},"agent_states":{},"version":1}',
        encoding="utf-8",
    )

    def fail_loads(raw):
        raise AssertionError("JSON parser should not run after preflight rejection")

    monkeypatch.setattr(state_management.json, "loads", fail_loads)

    with pytest.raises(ValueError, match="max_string_bytes"):
        load_snapshot(path, limits=_limits(max_string_bytes=100))


def test_load_preflight_uses_decoded_utf8_size_for_escaped_unicode(tmp_path):
    state = ConversationState("agent")
    snapshot = create_snapshot(
        environment_state={"message": "你" * 10},
        agent_states=[state],
        limits=_limits(max_string_bytes=30),
    )
    path = tmp_path / "snapshot.json"

    save_snapshot(snapshot, path, limits=_limits(max_string_bytes=30))
    assert b"\\u" in path.read_bytes()

    loaded = load_snapshot(path, limits=_limits(max_string_bytes=30))
    assert loaded.environment_state["message"] == "你" * 10


def test_json_preflight_total_string_bytes_match_decoded_unicode():
    raw = b'{"\\u503c":"\\u4f60\\u597d"}'
    state_management._preflight_json_bytes(
        raw,
        CheckpointLimits(max_total_string_bytes=9),
    )
    with pytest.raises(ValueError, match="max_total_string_bytes"):
        state_management._preflight_json_bytes(
            raw,
            CheckpointLimits(max_total_string_bytes=8),
        )


def test_load_preflights_depth_before_json_parse(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    path.write_text(
        '{"created_at":"2026-09-23T00:00:00","environment_state":{"a":{"b":{"c":1}}},'
        '"agent_states":{},"version":1}',
        encoding="utf-8",
    )

    def fail_loads(raw):
        raise AssertionError("JSON parser should not run after preflight rejection")

    monkeypatch.setattr(state_management.json, "loads", fail_loads)

    with pytest.raises(ValueError, match="max_depth"):
        load_snapshot(path, limits=_limits(max_depth=2))


def test_checkpoint_limits_roundtrip_with_file_byte_limit(tmp_path):
    state = ConversationState("agent", max_turns=2, max_history_bytes=32)
    state.record_turn("user", "hello")
    snapshot = create_snapshot(
        environment_state={"mode": "bounded"},
        agent_states=[state],
        limits=_limits(),
    )
    path = tmp_path / "snapshot.json"

    save_snapshot(snapshot, path, max_bytes=100_000, limits=_limits())
    loaded = load_snapshot(path, max_bytes=100_000, limits=_limits())

    assert loaded.environment_state == snapshot.environment_state
    assert loaded.agent_states == snapshot.agent_states


def test_save_rejects_graph_limit_before_creating_staging_file(tmp_path):
    snapshot = create_snapshot(environment_state={"message": "x" * 100})
    path = tmp_path / "snapshot.json"

    with pytest.raises(ValueError, match="max_string_bytes"):
        save_snapshot(snapshot, path, limits=_limits(max_string_bytes=10))

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def _environment():
    env = Environment(RoundRobinScheduler())
    agent = TransformerAgent(name="agent", llm_backend=lambda _: "ok")
    env.register_agent(agent)
    return env, agent


def test_environment_snapshot_applies_graph_limits_to_runtime_state():
    env, _ = _environment()
    env.large_extra = "z" * 200

    with pytest.raises(ValueError, match="max_string_bytes"):
        env.snapshot(limits=_limits(max_string_bytes=100))


def test_environment_restore_validates_limits_before_mutation():
    source, _ = _environment()
    source.state["payload"] = "x" * 200
    snapshot = source.snapshot()

    restored, _ = _environment()
    restored.state["sentinel"] = "unchanged"

    with pytest.raises(ValueError, match="max_string_bytes"):
        restored.restore(snapshot, limits=_limits(max_string_bytes=100))

    assert restored.state == {"sentinel": "unchanged"}


def test_runtime_capture_preserves_json_roundtrip_normalization():
    env, agent = _environment()
    env.tuple_extra = ("a", "b")
    agent.attributes["coords"] = (1, 2)

    snapshot = env.snapshot()

    assert snapshot.runtime_state["environment_extra"]["tuple_extra"] == ["a", "b"]
    assert snapshot.runtime_state["agents"]["agent"]["attributes"]["coords"] == [1, 2]


def test_runtime_capture_normalizes_numeric_subclasses_and_keys():
    class Level(IntEnum):
        ONE = 1

    class FancyStr(str):
        def __str__(self):
            return "overridden"

    env, _ = _environment()
    env.extra = {
        "enum_value": Level.ONE,
        "str_value": FancyStr("content"),
        "numeric_key": {Level.ONE: "value"},
    }

    snapshot = env.snapshot()
    extra = snapshot.runtime_state["environment_extra"]["extra"]

    assert extra["enum_value"] == 1
    assert type(extra["enum_value"]) is int
    assert extra["str_value"] == "content"
    assert type(extra["str_value"]) is str
    assert extra["numeric_key"] == {"1": "value"}


def test_restore_does_not_deepcopy_entire_runtime_graph():
    source, _ = _environment()
    source.extra = {"items": [1, 2, 3]}
    snapshot = source.snapshot()

    class NoRootDeepcopy(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("restore must not deepcopy the complete runtime graph")

    runtime = NoRootDeepcopy(snapshot.runtime_state)
    restored, _ = _environment()

    checkpoint.restore_runtime(restored, runtime)

    assert restored.extra == {"items": [1, 2, 3]}


def test_restore_targeted_copies_do_not_alias_snapshot_runtime():
    source, source_agent = _environment()
    source.extra = {"nested": ["original"]}
    source_agent.attributes["labels"] = ["original"]
    snapshot = source.snapshot()
    frozen_runtime = deepcopy(snapshot.runtime_state)

    restored, restored_agent = _environment()
    restored.restore(snapshot)
    restored.extra["nested"].append("changed")
    restored_agent.attributes["labels"].append("changed")

    assert snapshot.runtime_state == frozen_runtime


def test_streamed_save_does_not_deepcopy_dataclass_payload(tmp_path):
    class NoDeepcopyList(list):
        def __deepcopy__(self, memo):
            raise AssertionError("streamed dataclass encoding must not recursively deepcopy fields")

    @dataclass
    class Payload:
        items: object

    snapshot = SimulationSnapshot(
        created_at=datetime(2026, 9, 23),
        environment_state={"payload": Payload(NoDeepcopyList(["x", "y"]))},
        agent_states={},
    )
    path = tmp_path / "dataclass.json"

    save_snapshot(snapshot, path)
    loaded = load_snapshot(path)

    assert loaded.environment_state == {"payload": {"items": ["x", "y"]}}
