import pytest

from neva.agents import TransformerAgent
from neva.environments import Environment
from neva.schedulers import (
    EventDrivenScheduler,
    LeastRecentlyUsedScheduler,
    PriorityScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    WeightedRandomScheduler,
)
from neva.utils.state_management import SimulationSnapshot


@pytest.mark.parametrize(
    "scheduler_type",
    [
        RoundRobinScheduler,
        LeastRecentlyUsedScheduler,
        PriorityScheduler,
        EventDrivenScheduler,
        RandomScheduler,
        WeightedRandomScheduler,
    ],
)
def test_scheduler_continuation_after_json_roundtrip(scheduler_type):
    def make():
        env = Environment(scheduler_type())
        for name in ("A", "B", "C"):
            env.register_agent(TransformerAgent(name=name, llm_backend=lambda p, n=name: n))
        return env

    original = make()
    scheduler = original.scheduler
    if isinstance(scheduler, EventDrivenScheduler):
        for agent in original.agents:
            scheduler.notify_event(agent)
    original.step()
    snapshot = SimulationSnapshot.from_json(original.snapshot().to_json())
    expected = original.step()
    restored = make()
    restored.restore(snapshot)
    assert restored.step() == expected
