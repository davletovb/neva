from __future__ import annotations

from typing import Optional

from models import Environment, Scheduler


class BasicEnvironment(Environment):
    """A minimal environment implementation suitable for smoke tests."""

    def __init__(
        self,
        name: str,
        description: str,
        scheduler: Optional[Scheduler] = None,
    ) -> None:
        super().__init__(scheduler)
        self.name = name
        self.description = description

    def context(self) -> str:
        """Return a descriptive string describing the environment state."""

        return f"This is a {self.name}. {self.description}"
