"""Compatibility shim: ``neva.utils.observer`` *is* the implementation module.

The implementation moved to :mod:`neva.utils.observability.observer`; this
module rebinds its own name in :data:`sys.modules` to that module object, so
existing imports and module-level attribute mutation on the legacy path keep
working unchanged.
"""

import sys

from .observability import observer as _implementation

sys.modules[__name__] = _implementation
