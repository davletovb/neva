"""Focused edge-case coverage for Composite and Conditional schedulers."""

import pytest

from neva.agents.base import AIAgent
from neva.environments import Environment
from neva.schedulers import CompositeScheduler, ConditionalScheduler, RoundRobinScheduler
from neva.utils.exceptions import ConfigurationError, SchedulingError


def test_composite_add_requires_string_group():
    scheduler = CompositeScheduler()
    with pytest.raises(ConfigurationError, match="group must be provided"):
        scheduler.add(StubAgent("bad-group"), group=123)


def test_composite_add_rejects_non_scheduler_override():
    scheduler = CompositeScheduler()
    with pytest.raises(ConfigurationError, match="scheduler override"):
        scheduler.add(StubAgent("A"), group="alpha", scheduler="round-robin")


def test_composite_rejects_invalid_factory_result():
    scheduler = CompositeScheduler(group_scheduler_factory=lambda: object())
    with pytest.raises(ConfigurationError, match="must be an instance of Scheduler"):
        scheduler.add(StubAgent("A"), group="alpha")
    assert scheduler.agents == []


def test_composite_moving_agent_between_groups_detaches_old_group():
    scheduler = CompositeScheduler()
    agent = StubAgent("wanderer")
    scheduler.add(agent, group="alpha")
    scheduler.add(agent, group="beta")

    assert scheduler._group_membership[agent] == "beta"
    assert scheduler._group_schedulers["beta"].agents == [agent]
    assert scheduler.agents == [agent]
    assert scheduler.get_next_agent() is agent
    # The old group had only this agent, so it must be gone entirely.
    assert "alpha" not in scheduler._group_schedulers
    assert "alpha" not in scheduler._group_order


def test_composite_moving_paused_agent_propagates_pause_to_new_child():
    scheduler = CompositeScheduler()
    agent = StubAgent("paused-wanderer")
    scheduler.add(agent, group="alpha")
    scheduler.pause(agent)

    scheduler.add(agent, group="beta")

    assert scheduler.is_paused(agent)
    assert scheduler._group_membership[agent] == "beta"
    assert scheduler._group_schedulers["beta"].is_paused(agent)


def test_composite_empty_group_raises_scheduling_error():
    scheduler = CompositeScheduler()
    with pytest.raises(SchedulingError, match="no groups"):
        scheduler.get_next_agent()


def test_composite_all_groups_paused_raises_scheduling_error():
    scheduler = CompositeScheduler()
    agent = StubAgent("idle")
    scheduler.add(agent, group="alpha")
    scheduler.pause(agent)

    with pytest.raises(SchedulingError, match="no active agents"):
        scheduler.get_next_agent()


def test_composite_skips_paused_agent_and_schedules_other_group():
    scheduler = CompositeScheduler()
    alpha = StubAgent("alpha")
    beta = StubAgent("beta")
    scheduler.add(alpha, group="alpha")
    scheduler.add(beta, group="beta")
    scheduler.pause(alpha)

    assert scheduler.get_next_agent() is beta


def test_composite_removing_last_group_member_prunes_group():
    scheduler = CompositeScheduler()
    alpha = StubAgent("alpha")
    beta = StubAgent("beta")
    scheduler.add(alpha, group="alpha")
    scheduler.add(beta, group="beta")

    scheduler.terminate(alpha)
    assert "alpha" not in scheduler._group_schedulers
    assert "alpha" not in scheduler._group_order
    # Group index stays valid after pruning.
    assert scheduler.get_next_agent() is beta


def test_composite_pause_after_removal_keeps_index_in_range():
    scheduler = CompositeScheduler()
    agents = [StubAgent(name) for name in ("a1", "a2", "b1")]
    scheduler.add(agents[0], group="alpha")
    scheduler.add(agents[1], group="alpha")
    scheduler.add(agents[2], group="beta")

    # One selection leaves the group index at 1 ("beta"); pruning "alpha"
    # then shrinks the order to length 1, so the final selection only works
    # if removal normalizes the index instead of raising IndexError.
    scheduler.get_next_agent()
    scheduler.terminate(agents[0])
    scheduler.terminate(agents[1])
    assert "alpha" not in scheduler._group_order
    assert scheduler._group_index == 0
    # No IndexError after pruning the current group.
    assert scheduler.get_next_agent() is agents[2]


def test_conditional_add_rejects_non_callable_condition():
    scheduler = ConditionalScheduler()
    with pytest.raises(ConfigurationError, match="callable condition"):
        scheduler.add(StubAgent("blocked"), condition="not-callable")


def test_conditional_set_condition_requires_registered_agent():
    scheduler = ConditionalScheduler()
    with pytest.raises(ConfigurationError, match="must be registered"):
        scheduler.set_condition(StubAgent("ghost"), lambda agent: True)


def test_conditional_predicate_exception_wrapped_as_scheduling_error():
    scheduler = ConditionalScheduler()
    agent = StubAgent("boom")

    def exploding(agent):
        raise ValueError("predicate failure")

    scheduler.add(agent, condition=exploding)
    with pytest.raises(SchedulingError, match="Condition callable raised"):
        scheduler.get_next_agent()


def test_conditional_scheduler_pause_skips_condition_evaluation():
    scheduler = ConditionalScheduler()
    paused = StubAgent("paused")
    active = StubAgent("active")
    calls = []

    def counting_condition(agent):
        calls.append(agent.name)
        return True

    scheduler.add(paused, condition=counting_condition)
    scheduler.add(active, condition=counting_condition)
    scheduler.pause(paused)

    assert scheduler.get_next_agent() is active
    assert calls == ["active"]


def test_conditional_scheduler_remove_last_agent_resets_index():
    scheduler = ConditionalScheduler()
    agent = StubAgent("solo")
    scheduler.add(agent)
    scheduler.get_next_agent()

    scheduler.terminate(agent)
    assert scheduler._current_index == 0
    with pytest.raises(SchedulingError, match="no agents"):
        scheduler.get_next_agent()


def test_conditional_scheduler_removal_keeps_index_valid_for_remaining_agents():
    scheduler = ConditionalScheduler()
    agents = [StubAgent(name) for name in ("a", "b", "c")]
    for agent in agents:
        scheduler.add(agent)

    scheduler.get_next_agent()  # -> a (index now 1)
    scheduler.get_next_agent()  # -> b (index now 2)
    scheduler.terminate(agents[2])
    assert scheduler.get_next_agent() is agents[0]


def test_composite_environment_set_environment_propagates():
    composite = CompositeScheduler()
    agent = StubAgent("shared")

    # Registering the group scheduler happens on first add; set_environment
    # must reach it even when the environment is attached afterwards.
    composite.add(agent, group="alpha")
    env = StubEnvironment(composite)

    group_scheduler = composite._group_schedulers["alpha"]
    assert group_scheduler.environment is env


def test_composite_set_environment_before_add_propagates_later():
    composite = CompositeScheduler()
    env = StubEnvironment(composite)
    agent = StubAgent("early")
    composite.add(agent, group="alpha")

    group_scheduler = composite._group_schedulers["alpha"]
    assert group_scheduler.environment is env


def test_composite_per_group_scheduler_override():
    composite = CompositeScheduler()
    override = RoundRobinScheduler()
    agent_default = StubAgent("factory")
    agent_override = StubAgent("override")

    composite.add(agent_default, group="factory-group")
    composite.add(agent_override, group="override-group", scheduler=override)

    assert isinstance(composite._group_schedulers["factory-group"], RoundRobinScheduler)
    assert composite._group_schedulers["override-group"] is override
    assert composite.get_next_agent() is agent_default
    assert composite.get_next_agent() is agent_override


@pytest.mark.parametrize("outcome", ["empty", "none"])
def test_composite_skips_unavailable_child(outcome):
    class UnavailableScheduler(RoundRobinScheduler):
        def get_next_agent(self):
            if outcome == "empty":
                raise SchedulingError("not ready")
            return None

    scheduler = CompositeScheduler()
    scheduler.add(StubAgent("waiting"), group="waiting", scheduler=UnavailableScheduler())
    ready = StubAgent("ready")
    scheduler.add(ready, group="ready")
    assert scheduler.get_next_agent() is ready
    assert scheduler.simulation_observer.latest_snapshot()["scheduled_turn_count"] == 1


def test_conditional_false_predicate_can_be_updated():
    scheduler = ConditionalScheduler()
    agent = StubAgent("waiting")
    scheduler.add(agent, predicate=lambda _: False)
    with pytest.raises(SchedulingError, match="no agents meeting"):
        scheduler.get_next_agent()
    with pytest.raises(ConfigurationError, match="callable condition"):
        scheduler.set_condition(agent, "not-callable")
    scheduler.set_condition(agent, lambda _: True)
    assert scheduler.get_next_agent() is agent
    scheduler.pause(agent)
    with pytest.raises(SchedulingError):
        scheduler.get_next_agent()
    scheduler.resume(agent)
    assert scheduler.get_next_agent() is agent


def test_composite_removal_calls_parent_and_child_hooks_once():
    scheduler = CompositeScheduler()
    child = RoundRobinScheduler()
    agent = StubAgent("removed")
    events = []
    scheduler.register_termination_hook(lambda a: events.append(("parent", a)))
    child.register_termination_hook(lambda a: events.append(("child", a)))
    scheduler.add(agent, scheduler=child)
    scheduler.terminate(agent)
    scheduler.terminate(agent)
    assert events == [("child", agent), ("parent", agent)]
    assert scheduler.agents == child.agents == []
    with pytest.raises(SchedulingError, match="no groups"):
        scheduler.get_next_agent()
    replacement = StubAgent("replacement")
    scheduler.add(replacement)
    assert scheduler.get_next_agent() is replacement


def test_composite_group_fairness_is_independent_of_group_size():
    scheduler = CompositeScheduler()
    alpha = [StubAgent(f"a{index}") for index in range(3)]
    beta = [StubAgent("b0")]
    gamma = [StubAgent(f"g{index}") for index in range(2)]

    for agent in alpha:
        scheduler.add(agent, group="alpha")
    for agent in beta:
        scheduler.add(agent, group="beta")
    for agent in gamma:
        scheduler.add(agent, group="gamma")

    selected = [scheduler.get_next_agent().name for _ in range(18)]
    groups = [name[0] for name in selected]

    assert groups == ["a", "b", "g"] * 6
    assert [name for name in selected if name.startswith("a")] == [
        "a0",
        "a1",
        "a2",
        "a0",
        "a1",
        "a2",
    ]
    assert [name for name in selected if name.startswith("g")] == [
        "g0",
        "g1",
        "g0",
        "g1",
        "g0",
        "g1",
    ]


def test_nested_composite_lifecycle_propagates_environment_pause_resume_and_termination():
    inner = CompositeScheduler()
    alpha = StubAgent("alpha")
    beta = StubAgent("beta")
    direct = StubAgent("direct")

    outer = CompositeScheduler()
    outer.add(alpha, group="nested", scheduler=inner)
    outer.add(beta, group="nested", scheduler=inner)
    outer.add(direct, group="direct")
    env = StubEnvironment(outer)

    assert inner.environment is env
    assert inner._group_order == ["default"]
    assert all(child.environment is env for child in inner._group_schedulers.values())
    assert [outer.get_next_agent().name for _ in range(4)] == [
        "alpha",
        "direct",
        "beta",
        "direct",
    ]

    outer.pause(alpha)
    assert outer.is_paused(alpha)
    assert inner.is_paused(alpha)
    assert inner._group_schedulers["default"].is_paused(alpha)
    assert [outer.get_next_agent().name for _ in range(2)] == ["beta", "direct"]

    outer.resume(alpha)
    assert not outer.is_paused(alpha)
    assert not inner.is_paused(alpha)
    assert not inner._group_schedulers["default"].is_paused(alpha)
    assert "alpha" in [outer.get_next_agent().name for _ in range(4)]

    events = []
    outer.register_termination_hook(lambda agent: events.append(("outer", agent.name)))
    inner.register_termination_hook(lambda agent: events.append(("inner", agent.name)))
    outer.terminate(alpha)

    assert events == [("inner", "alpha"), ("outer", "alpha")]
    assert alpha not in outer.agents
    assert alpha not in inner.agents
    assert inner._group_schedulers["default"].agents == [beta]
    assert all(outer.get_next_agent() is not alpha for _ in range(6))


def test_conditional_scheduler_round_robin_fairness_tracks_dynamic_eligibility():
    scheduler = ConditionalScheduler()
    allowed = {"a": False, "b": True, "c": True}
    agents = [StubAgent(name) for name in ("a", "b", "c")]
    for agent in agents:
        scheduler.add(agent, condition=lambda current, allowed=allowed: allowed[current.name])

    assert [scheduler.get_next_agent().name for _ in range(6)] == [
        "b",
        "c",
        "b",
        "c",
        "b",
        "c",
    ]

    allowed["a"] = True
    assert [scheduler.get_next_agent().name for _ in range(6)] == [
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
    ]

    scheduler.pause(agents[1])
    assert [scheduler.get_next_agent().name for _ in range(4)] == ["a", "c", "a", "c"]
    scheduler.resume(agents[1])
    scheduler.terminate(agents[2])
    assert [scheduler.get_next_agent().name for _ in range(4)] == ["a", "b", "a", "b"]


class StubAgent(AIAgent):
    def __init__(self, name):
        super().__init__(name=name)

    def respond(self, message):
        return message


class StubEnvironment(Environment):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._context_calls = 0

    def context(self):
        self._context_calls += 1
        return f"context-{self._context_calls}"
