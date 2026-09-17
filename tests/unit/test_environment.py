from neva.agents import AIAgent
from neva.environments import Environment
from neva.schedulers import RoundRobinScheduler


class StubAgent(AIAgent):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.messages = []

    def respond(self, message: str) -> str:
        self.messages.append(message)
        return f"{self.name}:{message}"


class StubEnvironment(Environment):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._context_calls = 0

    def context(self) -> str:
        self._context_calls += 1
        return f"context-{self._context_calls}"


def test_scheduler_collects_all_agents(monkeypatch):
    scheduler = RoundRobinScheduler()
    env = StubEnvironment(scheduler)
    agent_a = StubAgent("A")
    agent_b = StubAgent("B")
    env.register_agent(agent_a)
    env.register_agent(agent_b)

    collected = []

    def fake_collect(agents, environment=None):
        collected.append((list(agents), environment))

    scheduler.simulation_observer.collect_data = fake_collect

    next_agent = scheduler.get_next_agent()
    assert next_agent in {agent_a, agent_b}
    assert collected and collected[0][0] == [agent_a, agent_b]
    assert collected[0][1] is env


def test_environment_step_runs_agents():
    scheduler = RoundRobinScheduler()
    env = StubEnvironment(scheduler)
    agent = StubAgent("Solo")
    env.register_agent(agent)

    responses = env.run(3)
    assert responses == ["Solo:context-1", "Solo:context-2", "Solo:context-3"]
    assert agent.messages == ["context-1", "context-2", "context-3"]


def test_skip_policy_continues_after_agent_error():
    class BoomAgent(AIAgent):
        def __init__(self):
            super().__init__(name="Boom")

        def respond(self, message: str) -> str:
            raise RuntimeError("backend down")

    scheduler = RoundRobinScheduler()
    env = StubEnvironment(scheduler)
    env.register_agent(BoomAgent())
    env.register_agent(StubAgent("OK"))

    env.error_policy = "return"
    assert env.step() is None
    snapshot = scheduler.simulation_observer.latest_snapshot()
    assert snapshot["failed_turn_count"] == 1
    assert snapshot["turn_count"] == 0
    assert env.step() == "OK:context-2"
    assert scheduler.simulation_observer.latest_snapshot()["turn_count"] == 1


def test_invalid_error_policy_rejected():
    import pytest

    with pytest.raises(ValueError, match="error_policy"):
        Environment(RoundRobinScheduler(), error_policy="explode")


def test_on_turn_complete_hook_feeds_same_turn_metrics():
    class TranscriptEnv(StubEnvironment):
        def __init__(self, scheduler):
            super().__init__(scheduler)
            self.transcript = []

        def on_turn_complete(self, response: str) -> None:
            self.transcript.append(response)

    env = TranscriptEnv(RoundRobinScheduler())
    env.register_agent(StubAgent("A"))
    env.register_agent(StubAgent("B"))
    env.run(2)
    snapshot = env.scheduler.simulation_observer.latest_snapshot()
    assert snapshot["dialogue_length"] == 2
    assert snapshot["turn_count"] == 2


def test_environment_snapshot_and_restore(tmp_path):
    scheduler = RoundRobinScheduler()
    env = StubEnvironment(scheduler)
    agent = StubAgent("Recorder")
    env.register_agent(agent)
    env.state["mood"] = "focused"

    agent.receive("initial", sender="tester")
    snapshot = env.snapshot()

    env.state["mood"] = "distracted"
    agent.receive("another", sender="tester")

    env.restore(snapshot)
    assert env.state["mood"] == "focused"
    assert len(agent.conversation_state.turns) == 2
