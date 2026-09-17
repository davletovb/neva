import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.schedulers import RoundRobinScheduler


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
    original_step = env.step

    def step_with_transcript():
        message = original_step()
        if message:
            env.transcript.append(message)
        return message

    env.step = step_with_transcript
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
