import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.schedulers import RoundRobinScheduler
from neva.utils.exceptions import SchedulingError
from neva.utils.state_management import SimulationSnapshot


def fail(prompt):
    raise RuntimeError("provider down")


@pytest.mark.parametrize("policy", ["ignore", "", True, 1])
def test_invalid_override_does_not_register_agent(policy):
    env = Environment(RoundRobinScheduler())
    agent = TransformerAgent(name="bad", llm_backend=fail)
    with pytest.raises(ValueError, match="error_policy"):
        env.register_agent(agent, error_policy=policy)
    assert env.agents == []


def test_fallback_requires_explicit_policy():
    env = Environment(RoundRobinScheduler())
    with pytest.raises(ValueError, match="error_value"):
        env.register_agent(TransformerAgent(llm_backend=fail), error_value="offline")
    assert env.agents == []


def test_agent_override_survives_checkpoint_restore():
    env = Environment(RoundRobinScheduler())
    agent = TransformerAgent(name="flaky", llm_backend=fail)
    env.register_agent(agent, error_policy="return", error_value="offline")
    snapshot = SimulationSnapshot.from_json(env.snapshot().to_json())

    restored = Environment(RoundRobinScheduler())
    replacement = TransformerAgent(name="flaky", llm_backend=fail)
    restored.register_agent(replacement)
    assert replacement.id != agent.id
    restored.restore(snapshot)
    assert replacement.id == agent.id
    assert restored.step() == "offline"


def test_legacy_checkpoint_allows_registering_new_override():
    env = Environment(RoundRobinScheduler(), error_policy="return", error_value="default")
    env.register_agent(TransformerAgent(name="old", llm_backend=fail))
    snapshot = env.snapshot()
    del snapshot.runtime_state["environment_extra"]["_agent_error_policies"]
    env.restore(SimulationSnapshot.from_json(snapshot.to_json()))
    assert env.step() == "default"
    env.register_agent(
        TransformerAgent(name="new", llm_backend=fail),
        error_policy="return",
        error_value="new fallback",
    )
    assert "new fallback" in env.run(2)


def test_raise_override_and_default_inheritance():
    env = Environment(RoundRobinScheduler(), error_policy="return", error_value="default")
    env.register_agent(TransformerAgent(name="strict", llm_backend=fail), error_policy="raise")
    env.register_agent(TransformerAgent(name="inherited", llm_backend=fail))
    with pytest.raises(RuntimeError, match="provider down"):
        env.step()
    assert env.step() == "default"


def test_raise_override_rejects_error_value():
    env = Environment(RoundRobinScheduler())
    with pytest.raises(ValueError, match="unused"):
        env.register_agent(
            TransformerAgent(llm_backend=fail), error_policy="raise", error_value="x"
        )
    assert env.agents == []


def test_malformed_override_without_value_returns_none():
    env = Environment(RoundRobinScheduler())
    env.register_agent(TransformerAgent(llm_backend=fail), error_policy="return")
    snapshot = env.snapshot()
    del snapshot.runtime_state["environment_extra"]["_agent_error_policies"][str(env.agents[0].id)][
        "value"
    ]
    env.restore(SimulationSnapshot.from_json(snapshot.to_json()))
    assert env.step() is None


def test_return_override_defaults_to_none_not_environment_value():
    env = Environment(RoundRobinScheduler(), error_policy="return", error_value="default")
    env.register_agent(TransformerAgent(llm_backend=fail), error_policy="return")
    assert env.step() is None


def test_checkpoint_restore_replaces_preexisting_overrides():
    env = Environment(RoundRobinScheduler())
    env.register_agent(
        TransformerAgent(name="A", llm_backend=fail),
        error_policy="return",
        error_value="source policy",
    )
    env.register_agent(TransformerAgent(name="B", llm_backend=fail))
    snapshot = SimulationSnapshot.from_json(env.snapshot().to_json())

    restored = Environment(RoundRobinScheduler())
    restored.register_agent(
        TransformerAgent(name="A", llm_backend=fail),
        error_policy="return",
        error_value="preexisting",
    )
    restored.register_agent(
        TransformerAgent(name="B", llm_backend=fail),
        error_policy="return",
        error_value="orphaned",
    )
    restored.restore(snapshot)
    assert restored.step() == "source policy"  # A override restored, B entry dropped
    assert str(restored.agents[1].id) not in restored._agent_error_policies


def test_legacy_checkpoint_clears_existing_overrides():
    env = Environment(RoundRobinScheduler(), error_policy="return", error_value="default")
    env.register_agent(TransformerAgent(name="A", llm_backend=fail), error_policy="raise")
    env.register_agent(TransformerAgent(name="B", llm_backend=fail), error_policy="return")
    snapshot = env.snapshot()
    del snapshot.runtime_state["environment_extra"]["_agent_error_policies"]
    env.restore(SimulationSnapshot.from_json(snapshot.to_json()))
    assert getattr(env, "_agent_error_policies", {}) == {}
    assert env.step() == "default"
    assert env.step() == "default"


def test_context_failure_uses_agent_override_not_environment_policy():
    class BrokenContextEnvironment(Environment):
        def context(self) -> str:
            if self.context_broken:
                raise RuntimeError("context assembly down")
            return ""

    env = BrokenContextEnvironment(RoundRobinScheduler(), error_policy="return", error_value="env")
    env.context_broken = True
    env.register_agent(TransformerAgent(llm_backend=lambda _: "ok"), error_policy="raise")
    with pytest.raises(RuntimeError, match="context assembly down"):
        env.step()

    env.context_broken = False
    env2 = BrokenContextEnvironment(RoundRobinScheduler())
    env2.context_broken = True
    env2.register_agent(
        TransformerAgent(llm_backend=lambda _: "ok"),
        error_policy="return",
        error_value="offline",
    )
    assert env2.step() == "offline"


def test_on_turn_complete_failure_respects_agent_override():
    def broken_hook(response):
        raise RuntimeError("transcript sink down")

    env = Environment(RoundRobinScheduler(), error_policy="return", error_value="env")
    env.on_turn_complete = broken_hook
    env.register_agent(TransformerAgent(llm_backend=lambda _: "ok"), error_policy="raise")
    with pytest.raises(RuntimeError, match="transcript sink down"):
        env.step()

    env2 = Environment(RoundRobinScheduler(), error_policy="raise")
    env2.on_turn_complete = broken_hook
    env2.register_agent(
        TransformerAgent(llm_backend=lambda _: "ok"),
        error_policy="return",
        error_value="sink down",
    )
    assert env2.step() == "sink down"
    snapshot = env2.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["failed_turn_count"] == 1


def test_scheduler_selection_failure_uses_environment_policy_not_override():
    from neva.schedulers import EventDrivenScheduler

    env = Environment(EventDrivenScheduler(), error_policy="raise")
    env.register_agent(
        TransformerAgent(llm_backend=fail),
        error_policy="return",
        error_value="unused",
    )
    with pytest.raises(SchedulingError, match="no pending events"):
        env.step()

    env2 = Environment(EventDrivenScheduler(), error_policy="return", error_value="NO_EVENT")
    env2.register_agent(TransformerAgent(llm_backend=lambda _: "ok"))
    assert env2.step() == "NO_EVENT"


def test_agent_can_return_while_environment_raises():
    env = Environment(RoundRobinScheduler())
    env.register_agent(
        TransformerAgent(name="flaky", llm_backend=fail),
        error_policy="return",
        error_value="offline",
    )
    env.register_agent(TransformerAgent(name="healthy", llm_backend=lambda _: "ok"))
    assert env.run(2) == ["offline", "ok"]
    metrics = env.scheduler.simulation_observer.latest_snapshot()
    assert metrics["failed_turn_count"] == 1
    assert metrics["turn_count"] == 1
