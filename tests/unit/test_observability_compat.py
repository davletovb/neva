"""Compatibility shims for the observability modules.

The implementations moved into ``neva.utils.observability``; the old module
paths must keep re-exporting the same objects so existing imports (including
the monkeypatch targets used by the telemetry tests) keep working.
"""

from __future__ import annotations

import neva.utils.observability as observability
import neva.utils.observer as observer_shim
import neva.utils.telemetry as telemetry_shim
from neva.utils.observability import observer as observer_impl
from neva.utils.observability import telemetry as telemetry_impl


def test_legacy_telemetry_path_reexports_the_implementation():
    assert telemetry_shim.TelemetryManager is telemetry_impl.TelemetryManager
    assert telemetry_shim.configure_telemetry is telemetry_impl.configure_telemetry
    assert telemetry_shim.get_telemetry is telemetry_impl.get_telemetry
    assert telemetry_shim.reset_telemetry is telemetry_impl.reset_telemetry
    assert telemetry_shim._require_opentelemetry is telemetry_impl._require_opentelemetry
    assert telemetry_shim.__all__ == [
        "TelemetryManager",
        "configure_telemetry",
        "get_telemetry",
        "reset_telemetry",
    ]


def test_legacy_observer_path_reexports_the_implementation():
    assert observer_shim.SimulationObserver is observer_impl.SimulationObserver
    assert observer_shim.__all__ == ["SimulationObserver"]


def test_observability_package_exports_the_public_objects():
    assert observability.TelemetryManager is telemetry_impl.TelemetryManager
    assert observability.SimulationObserver is observer_impl.SimulationObserver
    assert set(observability.__all__) == {
        "SimulationObserver",
        "TelemetryManager",
        "configure_telemetry",
        "get_telemetry",
        "reset_telemetry",
    }
