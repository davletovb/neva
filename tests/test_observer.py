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
