import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from observer import SimulationObserver


def test_collects_metrics():
    obs = SimulationObserver()
    # metric returns number of agents
    obs.add_metric("count", lambda agents, env: len(agents))
    agents = [1, 2, 3]
    obs.collect_data(agents, None)
    assert obs.data["count"] == [3]


def test_export_to_csv(tmp_path):
    obs = SimulationObserver()
    obs.add_metric("turns", lambda agents, env: len(agents))
    obs.collect_data(["agent"], None)

    csv_path = tmp_path / "metrics.csv"
    obs.export_to_csv(csv_path)

    assert csv_path.read_text().strip() == "turns,1"


def test_export_to_json(tmp_path):
    obs = SimulationObserver()
    obs.add_metric("messages", lambda agents, env: [agent for agent in agents])
    obs.collect_data(["A", "B"], None)

    json_path = tmp_path / "metrics.json"
    obs.export_to_json(json_path)

    import json

    assert json.loads(json_path.read_text()) == {"messages": [["A", "B"]]}
