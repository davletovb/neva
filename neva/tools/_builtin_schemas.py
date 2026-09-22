"""Argument schemas shared by Neva's built-in text tools."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .schemas import ArgumentSchema, ArgumentSpec

_TEXT_ARGUMENT_KEYS = ("input", "task", "query", "text")


class _BuiltInTextArgumentSchema:
    """Validate the mapping shapes accepted by AIAgent.call_tool.

    call_tool normalises mapping arguments by selecting the first string
    value under input, task, query, or text. The built-in text tools share
    that contract, so their schema mirrors the existing normalisation
    semantics instead of narrowing callers to one alias.
    """

    def __init__(self) -> None:
        self._shape = ArgumentSchema(
            {
                key: ArgumentSpec(type=object, required=False) for key in _TEXT_ARGUMENT_KEYS
            },
            allow_extra=True,
        )

    def validate(self, arguments: Mapping[str, Any]) -> Optional[str]:
        reason = self._shape.validate(arguments)
        if reason is not None:
            return reason

        for key in _TEXT_ARGUMENT_KEYS:
            if isinstance(arguments.get(key), str):
                return None

        if len(arguments) == 1 and isinstance(next(iter(arguments.values())), str):
            return None

        aliases = ", ".join(repr(key) for key in _TEXT_ARGUMENT_KEYS)
        return (
            f"one of {aliases} must contain a string, or a single mapping "
            "value must be a string"
        )


BUILTIN_TEXT_ARGUMENT_SCHEMA = _BuiltInTextArgumentSchema()
