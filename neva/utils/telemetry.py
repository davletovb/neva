"""Compatibility shim: the implementation moved to :mod:`neva.utils.observability.telemetry`.

Existing ``from neva.utils.telemetry import ...`` imports keep working, including
the module-level helpers used by tests; new code should import from
:mod:`neva.utils.observability.telemetry`.
"""

from .observability.telemetry import (  # noqa: F401
    TelemetryManager,
    _content_fields,
    _estimate_tokens,
    _extract_reasoning_steps,
    _normalise_attributes,
    _require_opentelemetry,
    configure_telemetry,
    get_telemetry,
    reset_telemetry,
)

__all__ = [
    "TelemetryManager",
    "configure_telemetry",
    "get_telemetry",
    "reset_telemetry",
]
