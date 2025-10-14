from __future__ import annotations

from neva.agents.base import AIAgent
from neva.environments.base import Environment


class EchoAgent(AIAgent):
    def respond(self, message: str) -> str:
        return f"echo:{message}"


class DummyTelemetryCollector:
    def __init__(self):
        self.turns = []

    def record_agent_turn(self, **kwargs):
        self.turns.append(kwargs)

    def record_agent_registration(self, **kwargs):  # pragma: no cover - optional hook.
        return

    def record_scheduler_decision(self, **kwargs):  # pragma: no cover - optional hook.
        return


def test_receive_emits_agent_turn(monkeypatch):
    telemetry = DummyTelemetryCollector()
    monkeypatch.setattr("neva.agents.base.get_telemetry", lambda: telemetry)

    agent = EchoAgent(name="echo")
    env = Environment()
    env.register_agent(agent)

    response = agent.receive("hello", sender="tester")
    assert response.startswith("echo:")
    assert telemetry.turns
    turn = telemetry.turns[-1]
    assert turn["agent_name"] == "echo"
    assert turn["prompt"] == "hello"
    assert turn["response"].startswith("echo:")
