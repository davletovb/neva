"""Utilities for persisting and restoring simulation state."""

from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, cast, runtime_checkable


def _utcnow_naive() -> datetime:
    """Return naive UTC without relying on the deprecated ``datetime.utcnow``."""

    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class ConversationTurn:
    speaker: str
    message: str
    timestamp: datetime = field(default_factory=_utcnow_naive)

    def to_dict(self) -> Dict[str, str]:
        return {
            "speaker": self.speaker,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, str]) -> "ConversationTurn":
        timestamp = datetime.fromisoformat(payload["timestamp"])
        return cls(speaker=payload["speaker"], message=payload["message"], timestamp=timestamp)


_TRUNCATION_MARKER = "...[truncated]"


@dataclass(frozen=True)
class CheckpointLimits:
    """Optional in-memory resource ceilings for checkpoint graphs.

    max_depth bounds nested mapping/sequence depth, max_nodes counts
    containers, scalar values, and mapping keys, max_string_bytes bounds
    any individual serialized string/key in UTF-8 bytes, and
    max_total_string_bytes bounds their aggregate UTF-8 bytes.

    Limits are opt-in so existing callers remain backward compatible. File
    size remains governed independently by max_bytes on save/load.
    """

    max_depth: Optional[int] = None
    max_nodes: Optional[int] = None
    max_string_bytes: Optional[int] = None
    max_total_string_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "max_nodes",
            "max_string_bytes",
            "max_total_string_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")


class _CheckpointLimitExceeded(ValueError):
    """Internal marker for an explicitly configured checkpoint resource ceiling."""


def _json_key_text(key: object) -> str:
    """Normalize a JSON mapping key the same way json.dumps/json.loads would."""

    if isinstance(key, str):
        return str.__str__(key)
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, int):
        return str(int(key))
    if isinstance(key, float):
        numeric = float(key)
        if not math.isfinite(numeric):
            raise ValueError("Checkpoint contains a non-finite mapping key")
        return str(numeric)
    raise TypeError("Checkpoint mapping keys must be str, int, float, bool, or None")


class _CheckpointGraphBudget:
    def __init__(self, limits: CheckpointLimits) -> None:
        self.limits = limits
        self.nodes = 0
        self.total_string_bytes = 0
        self._active: Set[int] = set()

    def _check_node(self) -> None:
        self.nodes += 1
        if self.limits.max_nodes is not None and self.nodes > self.limits.max_nodes:
            raise _CheckpointLimitExceeded("Checkpoint exceeds max_nodes")

    def _check_depth(self, depth: int) -> None:
        if self.limits.max_depth is not None and depth > self.limits.max_depth:
            raise _CheckpointLimitExceeded("Checkpoint exceeds max_depth")

    def _check_string(self, value: str) -> None:
        self._check_node()
        size = len(value.encode("utf-8", errors="replace"))
        if self.limits.max_string_bytes is not None and size > self.limits.max_string_bytes:
            raise _CheckpointLimitExceeded("Checkpoint string exceeds max_string_bytes")
        self.total_string_bytes += size
        if (
            self.limits.max_total_string_bytes is not None
            and self.total_string_bytes > self.limits.max_total_string_bytes
        ):
            raise _CheckpointLimitExceeded("Checkpoint exceeds max_total_string_bytes")

    def walk(self, value: object, *, depth: int = 0, native_only: bool = False) -> None:
        if isinstance(value, str):
            self._check_string(value)
            return
        if value is None or isinstance(value, (bool, int)):
            self._check_node()
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Checkpoint contains a non-finite number")
            self._check_node()
            return
        if isinstance(value, datetime):
            if native_only:
                raise TypeError("Checkpoint runtime state must contain only native JSON values")
            self._check_string(value.isoformat())
            return

        if isinstance(value, dict):
            self._check_node()
            container_depth = depth + 1
            self._check_depth(container_depth)
            marker = id(value)
            if marker in self._active:
                raise ValueError("Checkpoint contains a circular reference")
            self._active.add(marker)
            try:
                for key, item in value.items():
                    self._check_string(_json_key_text(key))
                    self.walk(item, depth=container_depth, native_only=native_only)
            finally:
                self._active.remove(marker)
            return

        if isinstance(value, (list, tuple)):
            self._check_node()
            container_depth = depth + 1
            self._check_depth(container_depth)
            marker = id(value)
            if marker in self._active:
                raise ValueError("Checkpoint contains a circular reference")
            self._active.add(marker)
            try:
                for item in value:
                    self.walk(item, depth=container_depth, native_only=native_only)
            finally:
                self._active.remove(marker)
            return

        if native_only:
            raise TypeError("Checkpoint runtime state must contain only native JSON values")
        if isinstance(value, _SupportsToDict):
            self.walk(value.to_dict(), depth=depth, native_only=False)
            return
        if is_dataclass(value):
            self._check_node()
            container_depth = depth + 1
            self._check_depth(container_depth)
            marker = id(value)
            if marker in self._active:
                raise ValueError("Checkpoint contains a circular reference")
            self._active.add(marker)
            try:
                for item in fields(value):
                    self._check_string(item.name)
                    self.walk(getattr(value, item.name), depth=container_depth, native_only=False)
            finally:
                self._active.remove(marker)
            return
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _validate_checkpoint_value(
    value: object,
    limits: Optional[CheckpointLimits],
    *,
    native_only: bool = False,
    budget: Optional[_CheckpointGraphBudget] = None,
) -> Optional[_CheckpointGraphBudget]:
    if limits is None:
        return budget
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits instance or None")
    active_budget = budget or _CheckpointGraphBudget(limits)
    active_budget.walk(value, native_only=native_only)
    return active_budget


def _preflight_json_bytes(raw: bytes | bytearray, limits: Optional[CheckpointLimits]) -> None:
    """Reject oversized JSON structure/string tokens before UTF-8 decode/parse.

    JSON escape sequences are charged by their decoded UTF-8 size so the same
    CheckpointLimits envelope is symmetric between save and load. The scanner
    never materialises a decoded string.
    """

    if limits is None:
        return
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits instance or None")

    nodes = 0
    depth = 0
    total_string_bytes = 0
    i = 0
    length = len(raw)

    def add_node() -> None:
        nonlocal nodes
        nodes += 1
        if limits.max_nodes is not None and nodes > limits.max_nodes:
            raise _CheckpointLimitExceeded("Checkpoint exceeds max_nodes")

    def decoded_codepoint_bytes(codepoint: int) -> int:
        if 0xD800 <= codepoint <= 0xDFFF:
            # json.loads can preserve lone surrogates; graph accounting encodes
            # them with errors="replace", which is one '?' byte per code unit.
            return 1
        if codepoint <= 0x7F:
            return 1
        if codepoint <= 0x7FF:
            return 2
        return 3

    def parse_hex_codepoint(offset: int) -> Optional[int]:
        if offset + 4 > length:
            return None
        value = 0
        for current in raw[offset : offset + 4]:
            if 48 <= current <= 57:
                digit = current - 48
            elif 65 <= current <= 70:
                digit = current - 55
            elif 97 <= current <= 102:
                digit = current - 87
            else:
                return None
            value = (value << 4) | digit
        return value

    def scan_string(offset: int) -> int:
        nonlocal total_string_bytes
        string_bytes = 0
        cursor = offset

        def charge(amount: int) -> None:
            nonlocal string_bytes
            string_bytes += amount
            if limits.max_string_bytes is not None and string_bytes > limits.max_string_bytes:
                raise _CheckpointLimitExceeded("Checkpoint string exceeds max_string_bytes")
            if (
                limits.max_total_string_bytes is not None
                and total_string_bytes + string_bytes > limits.max_total_string_bytes
            ):
                raise _CheckpointLimitExceeded("Checkpoint exceeds max_total_string_bytes")

        while cursor < length:
            current = raw[cursor]
            if current == 34:
                total_string_bytes += string_bytes
                return cursor + 1

            if current != 92:
                # Raw non-ASCII JSON is UTF-8, so counting its bytes directly is
                # exactly its decoded UTF-8 byte size.
                charge(1)
                cursor += 1
                continue

            if cursor + 1 >= length:
                charge(1)
                cursor += 1
                continue

            escape = raw[cursor + 1]
            if escape != 117:  # quote, slash, backslash, b/f/n/r/t, or invalid escape
                charge(1)
                cursor += 2
                continue

            codepoint = parse_hex_codepoint(cursor + 2)
            if codepoint is None:
                charge(1)
                cursor += min(6, length - cursor)
                continue

            if 0xD800 <= codepoint <= 0xDBFF and cursor + 12 <= length:
                if raw[cursor + 6] == 92 and raw[cursor + 7] == 117:
                    low = parse_hex_codepoint(cursor + 8)
                    if low is not None and 0xDC00 <= low <= 0xDFFF:
                        charge(4)
                        cursor += 12
                        continue

            charge(decoded_codepoint_bytes(codepoint))
            cursor += 6

        total_string_bytes += string_bytes
        return cursor

    while i < length:
        byte = raw[i]
        if byte in (9, 10, 13, 32, 44, 58):
            i += 1
            continue
        if byte == 34:
            add_node()
            i = scan_string(i + 1)
            continue
        if byte in (123, 91):
            add_node()
            depth += 1
            if limits.max_depth is not None and depth > limits.max_depth:
                raise _CheckpointLimitExceeded("Checkpoint exceeds max_depth")
            i += 1
            continue
        if byte in (125, 93):
            depth = max(0, depth - 1)
            i += 1
            continue
        if byte in b"-0123456789":
            add_node()
            i += 1
            while i < length and raw[i] not in b" \t\r\n,]}":
                i += 1
            continue
        if byte in (116, 102, 110):
            add_node()
            i += 1
            while i < length and raw[i] not in b" \t\r\n,]}":
                i += 1
            continue
        i += 1


def _truncate_utf8(message: str, max_bytes: Optional[int]) -> str:
    """Normalize and bound a stored message when a UTF-8 byte ceiling is configured.

    Python's UTF-8 encoder uses ``?`` for each surrogate code unit when
    ``errors="replace"``. With no ceiling, the original string is returned
    byte-for-byte unchanged.
    """

    if max_bytes is None:
        return message

    # Python strings can contain isolated surrogate code points (for example
    # after JSON decoding). Normalise those explicitly so enabling a storage
    # ceiling cannot turn an otherwise successful agent response into an
    # encoding failure.
    normalized = message.encode("utf-8", errors="replace").decode("utf-8")
    raw = normalized.encode("utf-8")
    if len(raw) <= max_bytes:
        return normalized

    marker = _TRUNCATION_MARKER.encode("ascii")
    if max_bytes <= len(marker):
        return marker[:max_bytes].decode("ascii")

    prefix_budget = max_bytes - len(marker)
    prefix = raw[:prefix_budget].decode("utf-8", errors="ignore")
    return prefix + _TRUNCATION_MARKER


@dataclass
class ConversationState:
    """Track chronological history with optional count and byte ceilings.

    max_turn_bytes bounds each stored message. max_history_bytes bounds the
    aggregate UTF-8 bytes of retained messages and evicts oldest turns as
    needed. The effective per-turn ceiling is the stricter of the two, so one
    oversized newest message is truncated rather than immediately evicted.

    The supplied turn list is always copied. Existing ConversationTurn objects
    retain their identity only when both byte ceilings are disabled. Because
    turns remains a public mutable list for compatibility, direct edits are
    reconciled on the next record_turn() or to_dict() call.
    """

    agent_name: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: Optional[int] = None
    max_turn_bytes: Optional[int] = None
    max_history_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns <= 0):
            raise ValueError("max_turns must be a positive integer or None")
        for name in ("max_turn_bytes", "max_history_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")

        byte_limit = self._effective_turn_byte_limit()
        if byte_limit is None:
            self.turns = list(self.turns)
        else:
            self.turns = [
                ConversationTurn(
                    speaker=turn.speaker,
                    message=_truncate_utf8(turn.message, byte_limit),
                    timestamp=turn.timestamp,
                )
                for turn in self.turns
            ]
        self._trim()

    @staticmethod
    def _message_bytes(message: str) -> int:
        return len(message.encode("utf-8", errors="replace"))

    def _effective_turn_byte_limit(self) -> Optional[int]:
        limits = [
            limit for limit in (self.max_turn_bytes, self.max_history_bytes) if limit is not None
        ]
        return min(limits) if limits else None

    def _drop_prefix(self, count: int) -> None:
        if count > 0:
            del self.turns[:count]

    def _trim(self) -> None:
        if self.max_turns is not None:
            self._drop_prefix(max(0, len(self.turns) - self.max_turns))

        if self.max_history_bytes is None:
            return

        retained_bytes = sum(self._message_bytes(turn.message) for turn in self.turns)
        drop_count = 0
        while drop_count < len(self.turns) and retained_bytes > self.max_history_bytes:
            retained_bytes -= self._message_bytes(self.turns[drop_count].message)
            drop_count += 1
        self._drop_prefix(drop_count)

    def record_turn(self, speaker: str, message: str) -> None:
        stored = _truncate_utf8(message, self._effective_turn_byte_limit())
        self.turns.append(ConversationTurn(speaker=speaker, message=stored))
        self._trim()

    def to_dict(self) -> Dict[str, object]:
        # Reconcile any direct mutations of the public turns list before
        # persistence so retention guarantees cannot be bypassed by a stale
        # cached byte count.
        self._trim()
        return {
            "agent_name": self.agent_name,
            "turns": [turn.to_dict() for turn in self.turns],
            "max_turns": self.max_turns,
            "max_turn_bytes": self.max_turn_bytes,
            "max_history_bytes": self.max_history_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ConversationState":
        raw_turns = payload.get("turns", [])
        turns = []
        if isinstance(raw_turns, Iterable):
            for turn_payload in raw_turns:
                if isinstance(turn_payload, dict):
                    turns.append(ConversationTurn.from_dict(cast(Dict[str, str], turn_payload)))
        return cls(
            agent_name=str(payload["agent_name"]),
            turns=turns,
            max_turns=cast(Optional[int], payload.get("max_turns")),
            max_turn_bytes=cast(Optional[int], payload.get("max_turn_bytes")),
            max_history_bytes=cast(Optional[int], payload.get("max_history_bytes")),
        )


@dataclass
class SimulationSnapshot:
    """Serializable view over environment state for persistence."""

    created_at: datetime
    environment_state: Dict[str, object]
    agent_states: Dict[str, Dict[str, object]]
    version: int = 1
    runtime_state: Optional[Dict[str, Any]] = None

    def _serialisable(self) -> Dict[str, Any]:
        """Return the canonical mapping persisted by both JSON save paths.

        Subclasses that intentionally customize checkpoint representation
        should override this method rather than only overriding ``to_json()``.
        """
        return {
            "created_at": self.created_at.isoformat(),
            "environment_state": self.environment_state,
            "agent_states": self.agent_states,
            "version": self.version,
            "runtime_state": self.runtime_state,
        }

    def validate_limits(self, limits: Optional[CheckpointLimits]) -> None:
        _validate_checkpoint_value(self._serialisable(), limits)

    def to_json(self) -> str:
        return json.dumps(self._serialisable(), default=_json_default, indent=2)

    @classmethod
    def _from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        limits: Optional[CheckpointLimits] = None,
    ) -> "SimulationSnapshot":
        _validate_checkpoint_value(payload, limits, native_only=True)
        version = payload.get("version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported snapshot version: {version}")
        created_at = datetime.fromisoformat(payload["created_at"])
        return cls(
            created_at=created_at,
            environment_state=payload["environment_state"],
            agent_states=payload["agent_states"],
            version=version,
            runtime_state=payload.get("runtime_state"),
        )

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        limits: Optional[CheckpointLimits] = None,
    ) -> "SimulationSnapshot":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Snapshot root must be a JSON object")
        return cls._from_payload(payload, limits=limits)


def create_snapshot(
    *,
    environment_state: Optional[Dict[str, object]] = None,
    agent_states: Optional[Iterable[ConversationState]] = None,
    limits: Optional[CheckpointLimits] = None,
) -> SimulationSnapshot:
    environment_state = environment_state or {}

    agent_snapshot: Dict[str, Dict[str, object]] = {}
    if agent_states is None:
        agent_iter: Iterable[ConversationState] = ()
    else:
        agent_iter = agent_states
    for state in agent_iter:
        agent_snapshot[state.agent_name] = state.to_dict()

    # Assemble the source graph first, then validate it exactly once before the
    # environment-state deepcopy. This keeps the pre-copy resource boundary
    # without walking environment and agent state twice.
    snapshot = SimulationSnapshot(
        created_at=_utcnow_naive(),
        environment_state=environment_state,
        agent_states=agent_snapshot,
    )
    snapshot.validate_limits(limits)
    snapshot.environment_state = deepcopy(environment_state)
    return snapshot


def _validate_max_bytes(max_bytes: Optional[int]) -> None:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer or None")


def save_snapshot(
    snapshot: SimulationSnapshot,
    path: Path,
    *,
    max_bytes: Optional[int] = None,
    limits: Optional[CheckpointLimits] = None,
) -> None:
    """Atomically save UTF-8 JSON without materialising the complete snapshot.

    ``max_bytes`` must be a positive integer or None (unlimited). JSON is
    encoded incrementally into a temporary file beside the destination and
    atomically installed with :func:`os.replace` only after serialization,
    size validation, flush, and fsync succeed. Existing checkpoints therefore
    survive serialization, staging-write, and replacement failures.

    This bounds serialization-buffer memory, but it requires temporary disk
    space on the destination filesystem roughly equal to the new checkpoint.
    Pass limits=CheckpointLimits(...) to validate the in-memory graph before
    staging; decoded-load limits are enforced separately by load_snapshot().
    ``_serialisable()`` is the shared representation point for both this
    function and ``to_json()``.
    """
    _validate_max_bytes(max_bytes)
    snapshot.validate_limits(limits)
    encoder = json.JSONEncoder(default=_json_default, indent=2)
    temp_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as staged:
            temp_path = Path(staged.name)
            total_bytes = 0
            exceeds_limit = False

            for piece in encoder.iterencode(snapshot._serialisable()):
                chunk = piece.encode("utf-8")
                total_bytes += len(chunk)
                if max_bytes is not None and total_bytes > max_bytes:
                    exceeds_limit = True
                    continue
                staged.write(chunk)

            if exceeds_limit:
                raise ValueError("Snapshot exceeds max_bytes")

            staged.flush()
            os.fsync(staged.fileno())

        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def load_snapshot(
    path: Path,
    *,
    max_bytes: Optional[int] = None,
    limits: Optional[CheckpointLimits] = None,
) -> SimulationSnapshot:
    """Load UTF-8 JSON with optional file and in-memory graph ceilings.

    max_bytes bounds serialized input bytes. limits preflights nesting, node
    count, and raw JSON string-token bytes before UTF-8 decode/JSON parse, then
    validates the parsed graph before constructing the snapshot.

    The stdlib JSON parser still materialises decoded text while parsing. The
    file and graph ceilings bound that work rather than changing the checkpoint
    format into a streaming parser.
    """

    _validate_max_bytes(max_bytes)
    if limits is not None and not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits instance or None")

    raw = bytearray()
    with path.open("rb") as source:
        remaining = max_bytes + 1 if max_bytes is not None else None
        while remaining is None or remaining > 0:
            request = 65536 if remaining is None else min(remaining, 65536)
            chunk = source.read(request)
            if not chunk:
                break
            raw.extend(chunk)
            if remaining is not None:
                remaining -= len(chunk)

    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("Snapshot exceeds max_bytes")

    _preflight_json_bytes(raw, limits)
    decoded = raw.decode("utf-8")
    del raw
    payload = json.loads(decoded)
    del decoded
    if not isinstance(payload, dict):
        raise ValueError("Snapshot root must be a JSON object")
    return SimulationSnapshot._from_payload(payload, limits=limits)


@runtime_checkable
class _SupportsToDict(Protocol):
    def to_dict(self) -> Dict[str, Any]:
        ...


def _json_default(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, _SupportsToDict):
        return value.to_dict()
    if is_dataclass(value):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
