"""Declarative argument validation for tool invocations.

Tools may declare an :class:`ArgumentSchema` describing the arguments they
accept; :meth:`AIAgent.call_tool` validates ``ToolCall.arguments`` against it
before executing the tool, so malformed calls fail closed with a reason
instead of reaching the tool body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, Union

from neva.utils.exceptions import ToolSchemaConfigurationError

__all__ = ["ArgumentSchema", "ArgumentSpec"]

_TypeSpec = Union[type, Tuple[type, ...]]


def _type_names(expected: _TypeSpec) -> str:
    if isinstance(expected, tuple):
        return " or ".join(getattr(item, "__name__", str(item)) for item in expected)
    return getattr(expected, "__name__", str(expected))


_UNSIZED_TYPES = (int, float, complex, bool)


def _format_value(value: Any) -> str:
    """Render ``value`` safely: never raises, never unbounded."""

    try:
        text = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _positive_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolSchemaConfigurationError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolSchemaConfigurationError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ToolSchemaConfigurationError(
            f"{label} must be a finite number within the float range"
        )
    return float(value)


@dataclass(frozen=True)
class ArgumentSpec:
    """Validation rules for a single tool argument.

    Checks run in order: type, length bounds (``len(value)`` for sized
    values), value bounds (for numbers; booleans never count as numbers),
    then ``choices``. Bounds that do not apply to the value's type disqualify
    the call rather than being skipped, and non-finite numbers (NaN, inf) are
    rejected whenever value bounds are configured — a comparison bound does
    not constrain NaN. ``type=float`` matches floats only; use
    ``(int, float)`` to accept integral numbers from parsed payloads.
    """

    type: _TypeSpec = str
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = field(default=None)

    def __post_init__(self) -> None:
        expected = self.type
        candidates = expected if isinstance(expected, tuple) else (expected,)
        if not candidates or not all(isinstance(item, type) for item in candidates):
            raise ToolSchemaConfigurationError("type must be a type or a tuple of types")
        if not isinstance(self.required, bool):
            raise ToolSchemaConfigurationError("required must be a boolean")
        min_length = _positive_int(self.min_length, "min_length")
        max_length = _positive_int(self.max_length, "max_length")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ToolSchemaConfigurationError("min_length must not exceed max_length")
        min_value = _finite_number(self.min_value, "min_value")
        max_value = _finite_number(self.max_value, "max_value")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ToolSchemaConfigurationError("min_value must not exceed max_value")
        if (min_length is not None or max_length is not None) and all(
            issubclass(item, _UNSIZED_TYPES) for item in candidates
        ):
            raise ToolSchemaConfigurationError(
                "min_length/max_length cannot apply to the declared type, " "which has no length"
            )
        if self.choices is not None:
            if not isinstance(self.choices, tuple) or not self.choices:
                raise ToolSchemaConfigurationError("choices must be a non-empty tuple")
            seen: list = []
            for choice in self.choices:
                if isinstance(choice, float) and math.isnan(choice):
                    raise ToolSchemaConfigurationError("choices must not contain NaN")
                if any(choice == previous for previous in seen):
                    raise ToolSchemaConfigurationError("choices must not contain duplicates")
                seen.append(choice)
                if not self._type_matches(choice):
                    raise ToolSchemaConfigurationError("choices must match the declared type")
        object.__setattr__(self, "min_length", min_length)
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "min_value", min_value)
        object.__setattr__(self, "max_value", max_value)

    def validate(self, name: str, value: Any) -> Optional[str]:
        """Return a human-readable violation, or ``None`` when ``value`` is valid."""

        if not self._type_matches(value):
            return (
                f"argument '{name}' must be of type {_type_names(self.type)}, "
                f"got {type(value).__name__}"
            )
        if self.min_length is not None or self.max_length is not None:
            try:
                length = len(value)
            except Exception:
                return (
                    f"argument '{name}' does not support the configured "
                    "min_length/max_length validation"
                )
            if self.min_length is not None and length < self.min_length:
                return (
                    f"argument '{name}' must be at least {self.min_length} "
                    f"characters, got {length}"
                )
            if self.max_length is not None and length > self.max_length:
                return (
                    f"argument '{name}' must be at most {self.max_length} "
                    f"characters, got {length}"
                )
        if self.min_value is not None or self.max_value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return (
                    f"argument '{name}' does not support the configured "
                    "min_value/max_value validation"
                )
            if isinstance(value, float) and not math.isfinite(value):
                return (
                    f"argument '{name}' must be a finite number, "
                    f"got {_format_value(value)}"
                )
            if self.min_value is not None and value < self.min_value:
                return (
                    f"argument '{name}' must be >= {self.min_value:g}, "
                    f"got {_format_value(value)}"
                )
            if self.max_value is not None and value > self.max_value:
                return (
                    f"argument '{name}' must be <= {self.max_value:g}, "
                    f"got {_format_value(value)}"
                )
        if self.choices is not None and value not in self.choices:
            return (
                f"argument '{name}' must be one of "
                f"{_format_value(self.choices)}, got {_format_value(value)}"
            )
        return None

    def _type_matches(self, value: Any) -> bool:
        if isinstance(value, bool):
            candidates = self.type if isinstance(self.type, tuple) else (self.type,)
            return bool in candidates
        return isinstance(value, self.type)


class ArgumentSchema:
    """Schema for a tool's arguments, validated before execution.

    Unknown keys are rejected unless ``allow_extra=True`` (extra keys are
    never validated, and an extra key can still become the executed payload
    when it is one of the payload keys ``call_tool`` recognises — declare
    every key the tool may consume). Required fields must be present and each
    present field is checked against its :class:`ArgumentSpec`. Validation is
    deterministic: unexpected keys first, then the declared fields in
    definition order with each field's checks in place.

    ``call_tool`` validates mapping arguments; a raw-string payload is
    validated as ``{"input": payload}``.
    """

    def __init__(
        self,
        fields: Mapping[str, ArgumentSpec],
        *,
        allow_extra: bool = False,
    ) -> None:
        if not isinstance(fields, Mapping):
            raise ToolSchemaConfigurationError(
                "fields must be a mapping of argument names to ArgumentSpec"
            )
        for name, spec in fields.items():
            if not isinstance(name, str):
                raise ToolSchemaConfigurationError("fields must use string argument names")
            if not isinstance(spec, ArgumentSpec):
                raise ToolSchemaConfigurationError(
                    "fields must map argument names to ArgumentSpec instances"
                )
        if not isinstance(allow_extra, bool):
            raise ToolSchemaConfigurationError("allow_extra must be a boolean")
        self.fields: Mapping[str, ArgumentSpec] = MappingProxyType(dict(fields))
        self.allow_extra = allow_extra

    def validate(self, arguments: Mapping[str, Any]) -> Optional[str]:
        """Return the first violation found, or ``None`` when ``arguments`` is valid."""

        if not isinstance(arguments, Mapping):
            return "arguments must be a mapping of argument names to values"
        if not self.allow_extra:
            unexpected = sorted(
                (key for key in arguments if key not in self.fields),
                key=_format_value,
            )
            if unexpected:
                rendered = ", ".join(_format_value(key) for key in unexpected)
                return f"unexpected argument(s): {rendered}"
        for name, spec in self.fields.items():
            if name not in arguments:
                if spec.required:
                    return f"missing required argument '{name}'"
                continue
            reason = spec.validate(name, arguments[name])
            if reason is not None:
                return reason
        return None
