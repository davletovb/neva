from models import Environment


class BasicEnvironment(Environment):
    """
    A basic environment for AI agents.
    """

    def __init__(self, name, description, scheduler):
        super().__init__(scheduler)
        self.name = name
        self.description = description

    def context(self):
        """
        Return a string that describes the environment.
        """
        return f"This is a {self.name}. {self.description}"
