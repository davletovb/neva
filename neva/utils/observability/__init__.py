"""Observability: OpenTelemetry instrumentation and simulation observers.

``telemetry`` exports spans, metrics, and structured logs through the optional
OpenTelemetry integration (with a dependency-free fallback), and ``observer``
turns simulation events into observer callbacks and tool-call telemetry. Both
describe the same concern - reporting what agents, tools, and schedulers do -
so they live together instead of among the general-purpose ``neva.utils``
modules.

``neva.utils.telemetry`` and ``neva.utils.observer`` remain importable as
compatibility shims that re-export the public objects defined here.
"""

from .observer import SimulationObserver
from .telemetry import TelemetryManager, configure_telemetry, get_telemetry, reset_telemetry

__all__ = [
    "SimulationObserver",
    "TelemetryManager",
    "configure_telemetry",
    "get_telemetry",
    "reset_telemetry",
]
