"""
This class captures the agents' states, environment conditions, and any other relevant metrics at each step of the simulation.
"""

import csv
import json


class SimulationObserver:
    """Collect and store metrics over a simulation run."""

    def __init__(self):
        # ``metrics`` holds the functions used to compute each metric while
        # ``data`` stores the collected values.  Previously both were mixed in
        # ``data``, leading to type errors when collecting.
        self.metrics = {}
        self.data = {}

    def add_metric(self, metric_name, metric_function):
        """Register a metric to be tracked.

        Parameters
        ----------
        metric_name: str
            Name of the metric to record.
        metric_function: Callable[[Any, Any], Any]
            Function that computes the metric given the agents and
            environment.
        """
        self.metrics[metric_name] = metric_function
        self.data[metric_name] = []

    def collect_data(self, agents, environment=None):
        """Collect the current value for each registered metric."""
        for metric_name, metric_function in self.metrics.items():
            value = metric_function(agents, environment)
            self.data[metric_name].append(value)
            
    def export_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in self.data.items():
                writer.writerow([key] + value)

    def export_to_json(self, filename):
        with open(filename, 'w') as jsonfile:
            json.dump(self.data, jsonfile)
