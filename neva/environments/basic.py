"""Basic environment implementation used for smoke tests."""

from __future__ import annotations

from typing import Optional

from neva.environments.base import Environment
from neva.schedulers.base import Scheduler


class BasicEnvironment(Environment):
    """A minimal environment implementation suitable for smoke tests."""

    def __init__(
        self,
        name: str,
        description: str,
        scheduler: Optional[Scheduler] = None,
        *,
        error_policy: str = "raise",
        error_value: Optional[str] = None,
    ) -> None:
        super().__init__(scheduler, error_policy=error_policy, error_value=error_value)
        self.name = name
        self.description = description

    def context(self) -> str:
        """Return a descriptive string describing the environment state."""

        return f"This is a {self.name}. {self.description}"


__all__ = ["BasicEnvironment"]
