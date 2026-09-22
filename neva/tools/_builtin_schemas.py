"""Argument schemas shared by Neva's built-in text tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from neva.agents.base import TOOL_TEXT_ARGUMENT_KEYS


class _BuiltInTextArgumentSchema:
    """Validate mapping shapes that resolve to one text payload.

    The alias set comes from the same constant used by
    AIAgent._normalise_tool_input so the schema cannot silently drift from
    the normalizer. Mapping shapes that would fall through to JSON serialization
    are rejected before a built-in tool executes.
    """

    def validate(self, arguments: Mapping[str, Any]) -> Optional[str]:
        if not isinstance(arguments, Mapping):
            return "arguments must be a mapping of argument names to values"

        for key in TOOL_TEXT_ARGUMENT_KEYS:
            if isinstance(arguments.get(key), str):
                return None

        if len(arguments) == 1 and isinstance(next(iter(arguments.values())), str):
            return None

        aliases = ", ".join(repr(key) for key in TOOL_TEXT_ARGUMENT_KEYS)
        return (
            f"one of {aliases} must contain a string, or a single mapping value "
            + "must be a string"
        )


BUILTIN_TEXT_ARGUMENT_SCHEMA = _BuiltInTextArgumentSchema()
