import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import AIAgent
from schedulers import PriorityScheduler, RoundRobinScheduler


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
