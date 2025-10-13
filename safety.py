"""Backward-compatible shim for relocated module."""

from neva.utils import safety as _module

from neva.utils.safety import *  # noqa: F401,F403

__all__ = getattr(_module, "__all__", [name for name in globals() if not name.startswith("_")])
