import pytest

from neva.agents import AIAgent
from neva.schedulers import (
    CompositeScheduler,
    ConditionalScheduler,
    EventDrivenScheduler,
    LeastRecentlyUsedScheduler,
    PriorityScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    WeightedRandomScheduler,
    create_scheduler,
    register_scheduler,
    unregister_scheduler,
)
from neva.utils.exceptions import SchedulingError


class StubAgent(AIAgent):
    def __init__(self, name: str):
        super().__init__(name=name)

    def respond(self, message: str) -> str:  # pragma: no cover - not used here.
        return message


def test_round_robin_scheduler_cycles_agents_in_order():
    scheduler = RoundRobinScheduler()
    agents = [StubAgent("A"), StubAgent("B"), StubAgent("C")]
    for agent in agents:
        scheduler.add(agent)

    selected = [scheduler.get_next_agent() for _ in range(5)]
    assert [agent.name for agent in selected] == ["A", "B", "C", "A", "B"]


def test_priority_scheduler_picks_highest_priority_first():
    scheduler = PriorityScheduler()
    low = StubAgent("low")
    high = StubAgent("high")

    scheduler.add(low, priority=1)
    scheduler.add(high, priority=5)

    assert scheduler.get_next_agent() is high

    # After the first call the highest priority agent is reinserted, so the
    # queue still prioritises it even when more agents are added.
    another_low = StubAgent("another_low")
    scheduler.add(another_low, priority=2)
    assert scheduler.get_next_agent() is high


def test_scheduler_pause_and_resume_controls_activation():
    scheduler = RoundRobinScheduler()
    agents = [StubAgent("A"), StubAgent("B"), StubAgent("C")]
    for agent in agents:
        scheduler.add(agent)

    scheduler.pause(agents[1])
    cycle = [scheduler.get_next_agent() for _ in range(4)]
    assert [agent.name for agent in cycle] == ["A", "C", "A", "C"]

    scheduler.resume(agents[1])
    resumed_cycle = [scheduler.get_next_agent() for _ in range(3)]
    assert [agent.name for agent in resumed_cycle] == ["A", "B", "C"]
    assert agents[1] in resumed_cycle


def test_scheduler_termination_hooks_execute_on_remove():
    scheduler = PriorityScheduler()
    agent = StubAgent("hooked")
    events = []

    scheduler.register_termination_hook(lambda removed: events.append(removed.name))
    scheduler.add(agent, priority=1)
    scheduler.terminate(agent)

    assert events == ["hooked"]
    assert agent not in scheduler.agents


def test_composite_scheduler_balances_groups():
    scheduler = CompositeScheduler()
    group_one_agents = [StubAgent("A"), StubAgent("B")]
    group_two_agents = [StubAgent("C")]

    for agent in group_one_agents:
        scheduler.add(agent, group="alpha")
    for agent in group_two_agents:
        scheduler.add(agent, group="beta")

    turns = [scheduler.get_next_agent() for _ in range(5)]
    assert [agent.name for agent in turns] == ["A", "C", "B", "C", "A"]


def test_event_driven_scheduler_requires_event_signal():
    scheduler = EventDrivenScheduler()
    agent = StubAgent("triggered")
    scheduler.add(agent)

    with pytest.raises(SchedulingError):
        scheduler.get_next_agent()

    scheduler.notify_event(agent)
    assert scheduler.get_next_agent() is agent


def test_conditional_scheduler_respects_agent_state():
    scheduler = ConditionalScheduler()
    ready = StubAgent("ready")
    waiting = StubAgent("waiting")
    ready.ready = True
    waiting.ready = False

    predicate = lambda agent: getattr(agent, "ready", False)
    scheduler.add(ready, condition=predicate)
    scheduler.add(waiting, condition=predicate)

    assert scheduler.get_next_agent() is ready

    waiting.ready = True
    scheduler.set_condition(waiting, predicate)
    assert scheduler.get_next_agent() is waiting


def test_scheduler_registry_supports_custom_registration():
    class CustomScheduler(RoundRobinScheduler):
        pass

    register_scheduler("custom_round_robin", CustomScheduler, overwrite=True)
    try:
        created = create_scheduler("custom_round_robin")
        assert isinstance(created, CustomScheduler)
        assert isinstance(create_scheduler("CustomScheduler"), CustomScheduler)
    finally:
        unregister_scheduler("custom_round_robin")


def test_least_recently_used_scheduler_cycles_and_handles_removal():
    scheduler = LeastRecentlyUsedScheduler()
    alpha = StubAgent("alpha")
    beta = StubAgent("beta")
    scheduler.add(alpha)
    scheduler.add(beta)

    assert scheduler.get_next_agent() is alpha
    assert scheduler.get_next_agent() is beta

    scheduler.pause(beta)
    scheduler.terminate(beta)
    assert scheduler.get_next_agent() is alpha

    scheduler.terminate(alpha)
    with pytest.raises(SchedulingError):
        scheduler.get_next_agent()


def test_weighted_random_scheduler_respects_weights(monkeypatch):
    scheduler = WeightedRandomScheduler()
    light = StubAgent("light")
    heavy = StubAgent("heavy")
    scheduler.add(light, weight=1.0)
    scheduler.add(heavy, weight=4.0)

    samples = iter([0.5, 3.5])
    monkeypatch.setattr(
        "neva.schedulers.weighted_random.random.uniform", lambda _a, _b: next(samples)
    )

    assert scheduler.get_next_agent() is light
    scheduler.pause(light)
    assert scheduler.get_next_agent() is heavy


def test_random_scheduler_requires_active_agents(monkeypatch):
    scheduler = RandomScheduler()
    agent = StubAgent("solo")
    scheduler.add(agent)

    monkeypatch.setattr("neva.schedulers.random.random.choice", lambda agents: agents[0])
    assert scheduler.get_next_agent() is agent

    scheduler.pause(agent)
    with pytest.raises(SchedulingError):
        scheduler.get_next_agent()
