import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.memory import CompositeMemory, ShortTermMemory, SummaryMemory
from neva.schedulers import RoundRobinScheduler
from neva.utils.state_management import SimulationSnapshot


def test_completed_turn_metrics_recorded_after_step():
    env = make_environment()
    env.step()
    env.step()
    snapshot = env.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["turn_count"] == 2
    assert snapshot["scheduled_turn_count"] == 2
    assert snapshot["failed_turn_count"] == 0
    assert snapshot["per_agent_participation"] == {"A": 1, "B": 1}
    assert snapshot["latest_response_latency_seconds"] is not None


def test_checkpoint_preserves_observer_metrics():
    env = make_environment()
    env.agents[0].set_llm_backend(lambda prompt: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        env.step()
    env.agents[0].set_llm_backend(lambda prompt: "A")
    env.step()
    snapshot = SimulationSnapshot.from_json(env.snapshot().to_json())
    restored = make_environment()
    restored.restore(snapshot)
    original = env.scheduler.simulation_observer.latest_snapshot()
    restored_metrics = restored.scheduler.simulation_observer.latest_snapshot()
    assert restored_metrics["turn_count"] == original["turn_count"] == 1
    assert restored_metrics["scheduled_turn_count"] == original["scheduled_turn_count"] == 2
    assert restored_metrics["failed_turn_count"] == 1
    assert restored_metrics["per_agent_participation"] == {"A": 0, "B": 1}
    assert restored_metrics["latest_response_latency_seconds"] is not None


def test_failed_turn_metrics_recorded():
    def boom(prompt):
        raise RuntimeError("backend down")

    env = make_environment()
    env.agents[0].set_llm_backend(boom)
    with pytest.raises(RuntimeError):
        env.step()
    snapshot = env.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["failed_turn_count"] == 1
    assert snapshot["turn_count"] == 0


def make_environment():
    env = Environment(RoundRobinScheduler())
    for name in ("A", "B"):
        env.register_agent(TransformerAgent(name=name, llm_backend=lambda prompt, n=name: n))
    return env


def test_checkpoint_resumes_next_turn_and_isolates_nested_state():
    original = make_environment()
    original.state["nested"] = {"value": 1}
    assert original.step() == "A"
    snapshot = original.snapshot()
    original.state["nested"]["value"] = 2
    assert snapshot.environment_state["nested"]["value"] == 1
    restored = make_environment()
    restored.restore(SimulationSnapshot.from_json(snapshot.to_json()))
    assert restored.step() == original.step() == "B"


def test_checkpoint_restores_memory_and_custom_environment_fields():
    def summary(previous, record):
        return previous + record.message

    def configured():
        env = make_environment()
        env.transcript = []
        env.agents[0].set_memory(
            CompositeMemory(
                [ShortTermMemory(capacity=2), SummaryMemory(summary, initial_summary="seed")]
            )
        )
        return env

    original = configured()
    original.agents[0].receive("remember this")
    original.transcript.append("custom dialogue")
    expected = original.agents[0].recall_memory()
    snapshot = SimulationSnapshot.from_json(original.snapshot().to_json())
    restored = configured()
    restored.restore(snapshot)
    assert restored.agents[0].recall_memory() == expected
    assert restored.transcript == ["custom dialogue"]
    assert restored.agents[0].id == original.agents[0].id
    restored.transcript.append("later")
    assert original.transcript == ["custom dialogue"]


def test_checkpoint_rejects_duplicate_names_and_future_versions():
    env = make_environment()
    env.agents[1].name = "A"
    with pytest.raises(ValueError, match="unique"):
        env.snapshot()
    with pytest.raises(ValueError, match="version"):
        SimulationSnapshot.from_json('{"version": 99}')


def test_checkpoint_memory_mismatch_does_not_mutate_state():
    original = make_environment()
    original.agents[0].set_memory(ShortTermMemory(capacity=2))
    snapshot = original.snapshot()
    restored = make_environment()
    restored.state["unchanged"] = True
    restored.agents[0].set_memory(ShortTermMemory(capacity=3))
    old_id = restored.agents[0].id
    with pytest.raises(ValueError, match="capacity"):
        restored.restore(snapshot)
    assert restored.agents[0].id == old_id
    assert restored.state == {"unchanged": True}


def test_checkpoint_rejects_unsupported_custom_state():
    env = make_environment()
    env.unserializable = object()
    with pytest.raises(ValueError, match="checkpoint"):
        env.snapshot()
