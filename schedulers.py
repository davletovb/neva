import random
from models import Scheduler


class RandomScheduler(Scheduler):
    """
    A scheduler that activates agents in a random order.
    """

    def add(self, agent):
        self.agents.append(agent)

    def get_next_agent(self):
        return random.choice(self.agents)


class RoundRobinScheduler(Scheduler):
    """
    A scheduler that activates agents in a round-robin order.
    """

    def __init__(self):
        super().__init__()
        self.current_index = 0

    def add(self, agent):
        self.agents.append(agent)

    def get_next_agent(self):
        agent = self.agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.agents)
        return agent


class PriorityScheduler(Scheduler):
    """
    A scheduler that activates agents based on their priority.
    """

    def add(self, agent, priority=1):
        self.agents.append((priority, agent))

    def get_next_agent(self):
        self.agents.sort(reverse=True)  # Sort agents by priority
        # Return the agent with the highest priority
        return self.agents.pop()[1]


class LeastRecentlyUsedScheduler(Scheduler):
    """
    A scheduler that activates the agent that has been waiting the longest.
    """

    def add(self, agent):
        self.agents.append(agent)

    def get_next_agent(self):
        return self.agents.pop(0)


class WeightedRandomScheduler(Scheduler):
    """
    A scheduler that activates agents randomly, but with weights affecting the likelihood of being chosen.
    """

    def add(self, agent, weight=1):
        self.agents.append((weight, agent))

    def get_next_agent(self):
        total_weight = sum(weight for weight, agent in self.agents)
        random_weight = random.uniform(0, total_weight)
        for weight, agent in self.agents:
            if random_weight < weight:
                return agent
            random_weight -= weight
