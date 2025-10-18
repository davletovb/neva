import gc
import json
import sys
import weakref
from types import SimpleNamespace

import pytest

from neva.utils.exceptions import MissingDependencyError
from neva.utils.observer import SimulationObserver


def test_collects_metrics():
    obs = SimulationObserver(enable_builtin_metrics=False)
    # metric returns number of agents
    obs.add_metric("count", lambda agents, env: len(agents))
    agents = [1, 2, 3]
    obs.collect_data(agents, None)
    assert obs.data["count"] == [3]


def test_export_to_csv(tmp_path):
    obs = SimulationObserver(enable_builtin_metrics=False)
    obs.add_metric("turns", lambda agents, env: len(agents))
    obs.collect_data(["agent"], None)

    csv_path = tmp_path / "metrics.csv"
    obs.export_to_csv(csv_path)

    assert csv_path.read_text().strip() == "turns,1"


def test_export_to_json(tmp_path):
    obs = SimulationObserver(enable_builtin_metrics=False)
    obs.add_metric("messages", lambda agents, env: [agent for agent in agents])
    obs.collect_data(["A", "B"], None)

    json_path = tmp_path / "metrics.json"
    obs.export_to_json(json_path)

    assert json.loads(json_path.read_text()) == {"messages": [["A", "B"]]}


def test_builtin_metrics_compute_turns_and_participation():
    obs = SimulationObserver()
    agent = SimpleNamespace(name="Explorer")
    environment = SimpleNamespace(transcript=["Hello", "How are you?"], state={})

    obs.watch_agent(agent)
    obs.collect_data([agent], environment, active_agent=agent)

    snapshot = obs.latest_snapshot()
    assert snapshot["turn_count"] == 1
    assert snapshot["per_agent_participation"]["Explorer"] == 1
    assert snapshot["dialogue_length"] == 2
    assert snapshot["recent_intent"] == "question"


def test_tool_usage_is_instrumented():
    class DummyTool:
        name = "dummy"

        def use(self, task):
            return task.upper()

    agent = SimpleNamespace(name="Explorer", tools=[])
    env = SimpleNamespace(transcript=[], state={})

    obs = SimulationObserver()
    obs.watch_agent(agent)
    tool = DummyTool()
    agent.tools.append(tool)
    obs.watch_tool(agent, tool)

    tool.use("ping")
    obs.collect_data([agent], env, active_agent=agent)

    snapshot = obs.latest_snapshot()
    assert snapshot["tool_usage_counts"]["Explorer"]["dummy"] == 1


def test_log_to_mlflow_requires_dependency(monkeypatch):
    obs = SimulationObserver()
    obs.collect_data([], None)

    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(MissingDependencyError) as exc:
        obs.log_to_mlflow()

    assert "MLflow is not installed" in str(exc.value)


def test_tool_instrumentation_allows_garbage_collection():
    class DummyTool:
        name = "dummy"

        def use(self, *_args, **_kwargs):
            return "ok"

    observer = SimulationObserver()
    agent = SimpleNamespace(name="Observer", tools=[])
    tool = DummyTool()
    ref = weakref.ref(tool)

    observer.watch_tool(agent, tool)
    agent.tools.clear()
    del tool

    gc.collect()

    assert ref() is None
