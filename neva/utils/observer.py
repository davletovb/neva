"""Compatibility shim: the implementation moved to :mod:`neva.utils.observability.observer`.

Existing ``from neva.utils.observer import ...`` imports keep working; new code
should import from :mod:`neva.utils.observability.observer`.
"""

from .observability.observer import SimulationObserver  # noqa: F401

__all__ = ["SimulationObserver"]
