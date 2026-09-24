"""Compatibility shims for the observability modules.

The implementations moved into ``neva.utils.observability``; the old module
paths must stay usable, including module-level attribute mutation
(``monkeypatch.setattr`` in downstream suites), so they alias the implementation
modules rather than copying names out of them.
"""

from __future__ import annotations

import neva.utils.observability as observability
import neva.utils.observer as observer_shim
import neva.utils.telemetry as telemetry_shim
from neva.utils.exceptions import MissingDependencyError
from neva.utils.observability import observer as observer_impl, telemetry as telemetry_impl


def test_legacy_paths_alias_the_implementation_modules():
    assert telemetry_shim is telemetry_impl
    assert observer_shim is observer_impl


def test_legacy_telemetry_path_reexports_the_implementation():
    assert telemetry_shim.TelemetryManager is telemetry_impl.TelemetryManager
    assert telemetry_shim.configure_telemetry is telemetry_impl.configure_telemetry
    assert telemetry_shim.get_telemetry is telemetry_impl.get_telemetry
    assert telemetry_shim.reset_telemetry is telemetry_impl.reset_telemetry
    assert telemetry_shim._require_opentelemetry is telemetry_impl._require_opentelemetry


def test_legacy_path_attribute_mutation_reaches_the_implementation(monkeypatch):
    """A patch on the legacy path must reach the implementation's own globals."""

    def missing() -> None:
        raise MissingDependencyError("opentelemetry unavailable")

    monkeypatch.setattr(telemetry_shim, "_require_opentelemetry", missing)
    telemetry = telemetry_impl.TelemetryManager(service_name="neva-legacy-patch")

    assert telemetry._trace_api is None and telemetry._tracer_provider is None
    telemetry.shutdown()


def test_legacy_observer_path_reexports_the_implementation():
    assert observer_shim.SimulationObserver is observer_impl.SimulationObserver


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
