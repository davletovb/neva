"""
This class captures the agents' states, environment conditions, and any other relevant metrics at each step of the simulation.
"""

import csv
import json

class SimulationObserver:
    def __init__(self):
        self.data = {}
        
    def add_metric(self, metric_name, metric_function):
        self.data[metric_name] = []
        self.data[metric_name].append(metric_function)
        
    def collect_data(self, agents, environment):
        for metric_name, metric_function in self.data.items():
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
