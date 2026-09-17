import pytest

from neva.agents import TransformerAgent
from neva.environments import BasicEnvironment, Environment
from neva.schedulers import EventDrivenScheduler, RoundRobinScheduler
from neva.utils.exceptions import SchedulingError


def test_basic_environment_exposes_error_policy():
    env = BasicEnvironment(
        "Lab", "Test", RoundRobinScheduler(), error_policy="return", error_value="unavailable"
    )

    def fail(prompt):
        raise RuntimeError("down")

    env.register_agent(TransformerAgent(llm_backend=fail))
    assert env.step() == "unavailable"
    assert env.scheduler.simulation_observer.latest_snapshot()["failed_turn_count"] == 1
    with pytest.raises(ValueError, match="error_policy"):
        BasicEnvironment("Lab", "Test", error_policy="invalid")


def make_environment(error_policy="raise", error_value=None):
    env = Environment(RoundRobinScheduler(), error_policy=error_policy, error_value=error_value)
    for name in ("A", "B"):
        env.register_agent(TransformerAgent(name=name, llm_backend=lambda prompt, n=name: n))
    return env


def test_dialogue_length_matches_completed_turns():
    # Mirrors examples/quickstart_conversation.py ConversationEnvironment:
    # a real environment appends each completed response to a transcript.
    env = make_environment()
    env.transcript = []
    env.on_turn_complete = env.transcript.append
    env.step()
    env.step()
    snapshot = env.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["dialogue_length"] == snapshot["turn_count"]


def test_step_survives_single_agent_failure_with_error_policy():
    env = make_environment(error_policy="return", error_value=None)

    def boom(prompt):
        raise RuntimeError("provider down")

    env.agents[0].set_llm_backend(boom)
    responses = env.run(4)
    assert responses.count("B") >= 1
    snapshot = env.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["failed_turn_count"] >= 1
    assert snapshot["turn_count"] == len([r for r in responses if r is not None])


def test_step_reraises_when_error_policy_is_raise():
    def boom(prompt):
        raise RuntimeError("provider down")

    env = make_environment(error_policy="raise")
    env.agents[0].set_llm_backend(boom)
    with pytest.raises(RuntimeError):
        env.step()


def test_step_returns_error_sentinel_when_configured():
    def boom(prompt):
        raise RuntimeError("provider down")

    env = make_environment(error_policy="return", error_value="AGENT_DOWN")
    env.agents[0].set_llm_backend(boom)
    assert env.step() == "AGENT_DOWN"
    snapshot = env.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["failed_turn_count"] == 1
    assert snapshot["turn_count"] == 0


def test_error_policy_covers_scheduling_error():
    env = Environment(EventDrivenScheduler(), error_policy="return", error_value="NO_EVENT")
    env.register_agent(TransformerAgent(name="A", llm_backend=lambda prompt: "A"))
    assert env.step() == "NO_EVENT"

    env_raise = Environment(EventDrivenScheduler(), error_policy="raise")
    env_raise.register_agent(TransformerAgent(name="A", llm_backend=lambda prompt: "A"))
    with pytest.raises(SchedulingError, match="no pending events"):
        env_raise.step()
