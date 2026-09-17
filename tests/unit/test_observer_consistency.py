import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from neva.utils.observer import SimulationObserver


def test_collection_publishes_one_coherent_snapshot():
    observer = SimulationObserver()
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    agent = SimpleNamespace(name="A")

    def pause_first(agents, environment, *, context):
        if context["turn_count"] == 1:
            entered.set()
            assert release.wait(2)
        return context["turn_count"]

    observer.add_metric("coherent_turn", pause_first)

    def second():
        second_started.set()
        observer.collect_data([agent], active_agent=agent)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(observer.collect_data, [agent], active_agent=agent)
        assert entered.wait(2)
        later = pool.submit(second)
        try:
            assert second_started.wait(2)
            # A snapshot publication must not be overtaken by another writer.
            assert not threading.Event().wait(0.05)
            assert not later.done()
        finally:
            release.set()
        first.result(timeout=2)
        later.result(timeout=2)
    snapshot = observer.latest_snapshot()
    assert snapshot["coherent_turn"] == snapshot["turn_count"] == 2
    assert observer.to_dict()["coherent_turn"] == [1, 2]


def test_concurrent_tool_records_survive_checkpoint_roundtrip():
    observer = SimulationObserver()
    agent = SimpleNamespace(name="A")

    class TestTool:
        name = "tool"

        def use(self, *args, **kwargs):
            return "ok"

    tool = TestTool()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: observer.record_tool_usage(agent, tool), range(200)))
    state = observer.checkpoint_state()
    restored = SimulationObserver()
    restored.restore_checkpoint_state(state)
    assert restored.checkpoint_state()["tool_usage"] == {"A": {"tool": 200}}
    assert len(restored.checkpoint_state()["tool_usage_events"]) == 200
