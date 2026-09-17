import asyncio
import threading

from neva.agents import AgentManager
from neva.agents.base import ParallelExecutionConfig
from neva.utils.observer import SimulationObserver


class _NamedStubAgent:
    """Minimal stand-in exposing only what the observer touches."""

    def __init__(self, name: str):
        self.name = name


def test_observer_is_thread_safe_under_concurrent_collection():
    observer = SimulationObserver()
    agents = [_NamedStubAgent(f"agent-{i}") for i in range(4)]
    errors = []

    def hammer(agent_index: int) -> None:
        try:
            for _ in range(250):
                observer.collect_data(agents, None, active_agent=agents[agent_index])
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(i % len(agents),)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert observer._turn_count == 8 * 250
    assert sum(observer._participation.values()) == 8 * 250


def test_sync_batches_can_reuse_manager_across_event_loops():
    import time

    manager = AgentManager(ParallelExecutionConfig(enabled=True, max_concurrency=1, batch_size=2))
    sender = manager.create_agent("transformer", name="S", llm_backend=lambda p: "s")

    def slow_backend(prompt):
        time.sleep(0.01)
        return "ok"

    receivers = [manager.create_agent("transformer", llm_backend=slow_backend) for _ in range(4)]
    ids = [str(agent.id) for agent in receivers]
    for _ in range(2):
        assert len(manager.batch_communicate(str(sender.id), ids, "hi")) == 4


def test_batch_communicate_async_reuses_one_concurrency_semaphore():
    manager = AgentManager(parallel_config=ParallelExecutionConfig(enabled=True, max_concurrency=2))
    sender = manager.create_agent("transformer", name="S", llm_backend=lambda prompt: "x")
    for i in range(3):
        manager.create_agent("transformer", name=f"R{i}", llm_backend=lambda prompt: "y")
    receiver_ids = [agent_id for agent_id, agent in manager.agents.items() if agent is not sender]

    async def scenario() -> None:
        await manager.batch_communicate_async(str(sender.id), receiver_ids, "hi")
        first = manager._concurrency_semaphore
        await manager.batch_communicate_async(str(sender.id), receiver_ids, "hi")
        second = manager._concurrency_semaphore
        return first, second

    first, second = asyncio.run(scenario())
    assert isinstance(first, asyncio.Semaphore)
    assert first is second
