"""Backward-compatible shim for relocated module."""

from neva.utils import metrics as _module

from neva.utils.metrics import *  # noqa: F401,F403

__all__ = getattr(_module, "__all__", [name for name in globals() if not name.startswith("_")])
