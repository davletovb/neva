"""Integration tests covering environment and scheduler collaboration."""

from __future__ import annotations

from neva.agents.base import AIAgent
from neva.environments import BasicEnvironment
from neva.schedulers import RoundRobinScheduler


class EchoAgent(AIAgent):
    """Simple agent that mirrors observations with a configurable prefix."""

    def __init__(self, *, name: str, response_prefix: str) -> None:
        super().__init__(name=name)
        self._response_prefix = response_prefix

    def respond(self, message: str) -> str:
        return f"{self._response_prefix}{message}"


def test_round_robin_environment_cycle() -> None:
    """Agents registered with the environment rotate according to the scheduler."""

    scheduler = RoundRobinScheduler()
    environment = BasicEnvironment(
        name="integration-test-environment",
        description="Ensures schedulers and agents cooperate correctly.",
        scheduler=scheduler,
    )

    agents = [
        EchoAgent(name="alpha", response_prefix="alpha heard: "),
        EchoAgent(name="beta", response_prefix="beta heard: "),
        EchoAgent(name="gamma", response_prefix="gamma heard: "),
    ]

    for agent in agents:
        environment.register_agent(agent)

    context = environment.context()
    outputs = environment.run(steps=4)

    assert outputs == [
        "alpha heard: " + context,
        "beta heard: " + context,
        "gamma heard: " + context,
        "alpha heard: " + context,
    ]

    alpha_history = agents[0].conversation_state.turns
    assert alpha_history[-2].message == context
    assert alpha_history[-1].message == outputs[-1]

    # The next scheduled agent should continue the rotation with beta.
    next_agent = scheduler.get_next_agent()
    assert next_agent is agents[1]
