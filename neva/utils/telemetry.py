"""Compatibility shim: ``neva.utils.telemetry`` *is* the implementation module.

The implementation moved to :mod:`neva.utils.observability.telemetry`; this
module rebinds its own name in :data:`sys.modules` to that module object, so
existing imports *and* module-level attribute mutation on the legacy path
(``monkeypatch.setattr("neva.utils.telemetry._require_opentelemetry", ...)`` in
downstream test suites) keep working unchanged.
"""

import sys

from .observability import telemetry as _implementation

sys.modules[__name__] = _implementation
