"""Shared infrastructure for the Neva showcase examples."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import neva.tools as tool_module
from neva.environments import BasicEnvironment
from neva.schedulers import RoundRobinScheduler


class TranscriptEnvironment(BasicEnvironment):
    """Environment that tracks a rolling transcript of agent outputs."""

    def __init__(
        self,
        name: str,
        description: str,
        scheduler: RoundRobinScheduler,
    ) -> None:
        super().__init__(name, description, scheduler)
        self.transcript: List[str] = []

    def recent_dialogue(self, limit: int = 3) -> str:
        if not self.transcript:
            return "No dialogue yet."
        return " | ".join(self.transcript[-limit:])

    def step(self) -> Optional[str]:  # type: ignore[override]
        message = super().step()
        if message:
            self.transcript.append(message)
        return message


def install_wikipedia_stub() -> None:
    """Ensure the Wikipedia tool works without the optional dependency."""

    wikipedia_module = import_module("neva.tools.wikipedia")

    if getattr(wikipedia_module, "wikipedia", None) is not None:
        return

    class _OfflineWikipedia:
        @staticmethod
        def summary(query: str, sentences: int = 2) -> str:
            return (
                f"Offline summary for '{query}' with {sentences} sentences. "
                "Replace this stub with the real wikipedia package for live data."
            )

    wikipedia_module.wikipedia = _OfflineWikipedia()
    tool_module.wikipedia = wikipedia_module.wikipedia  # type: ignore[attr-defined]


def make_persona_backend(
    name: str,
    persona: str,
    formatter: Callable[[str], str],
) -> Callable[[str], str]:
    """Return a stub LLM backend that formats responses consistently."""

    def backend(prompt: str) -> str:
        return f"{name} ({persona}) {formatter(prompt)}"

    return backend


def run_simulation(
    title: str,
    environment: TranscriptEnvironment,
    steps: int,
) -> None:
    """Execute ``steps`` environment turns and print a compact transcript."""

    print(f"\n=== {title} ===")
    messages = environment.run(steps)
    for index, message in enumerate(messages, start=1):
        if message is None:
            continue
        print(f"Turn {index}: {message}")

    observer = environment.scheduler.simulation_observer
    snapshot = observer.latest_snapshot()
    interesting_metrics = {
        key: snapshot[key]
        for key in (
            "turn_count",
            "per_agent_participation",
            "tool_usage_counts",
        )
        if key in snapshot
    }
    print("Metrics:", interesting_metrics)


__all__ = [
    "TranscriptEnvironment",
    "install_wikipedia_stub",
    "make_persona_backend",
    "run_simulation",
]
