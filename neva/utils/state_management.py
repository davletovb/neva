"""Utilities for persisting and restoring simulation state."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, cast, runtime_checkable


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


def _truncate_utf8(message: str, max_bytes: Optional[int]) -> str:
    """Normalize and bound a stored message when a UTF-8 byte ceiling is configured.

    Python's UTF-8 encoder uses ``?`` for each isolated surrogate when
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
    """Track chronological history with optional turn-count and byte ceilings.

    The supplied turn list is always copied. Existing ``ConversationTurn``
    objects retain their identity when ``max_turn_bytes`` is disabled; enabling
    the byte ceiling creates normalized, independently owned turn objects.
    """

    agent_name: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: Optional[int] = None
    max_turn_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns <= 0):
            raise ValueError("max_turns must be a positive integer or None")
        if self.max_turn_bytes is not None and (
            type(self.max_turn_bytes) is not int or self.max_turn_bytes <= 0
        ):
            raise ValueError("max_turn_bytes must be a positive integer or None")
        if self.max_turn_bytes is None:
            self.turns = list(self.turns)
        else:
            self.turns = [
                ConversationTurn(
                    speaker=turn.speaker,
                    message=_truncate_utf8(turn.message, self.max_turn_bytes),
                    timestamp=turn.timestamp,
                )
                for turn in self.turns
            ]
        self._trim()

    def _trim(self) -> None:
        if self.max_turns is not None:
            del self.turns[: max(0, len(self.turns) - self.max_turns)]

    def record_turn(self, speaker: str, message: str) -> None:
        self.turns.append(
            ConversationTurn(
                speaker=speaker,
                message=_truncate_utf8(message, self.max_turn_bytes),
            )
        )
        self._trim()

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "turns": [turn.to_dict() for turn in self.turns],
            "max_turns": self.max_turns,
            "max_turn_bytes": self.max_turn_bytes,
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

    def to_json(self) -> str:
        return json.dumps(self._serialisable(), default=_json_default, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "SimulationSnapshot":
        payload = json.loads(raw)
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


def create_snapshot(
    *,
    environment_state: Optional[Dict[str, object]] = None,
    agent_states: Optional[Iterable[ConversationState]] = None,
) -> SimulationSnapshot:
    environment_state = environment_state or {}
    agent_snapshot: Dict[str, Dict[str, object]] = {}
    if agent_states is None:
        agent_iter: Iterable[ConversationState] = ()
    else:
        agent_iter = agent_states
    for state in agent_iter:
        agent_snapshot[state.agent_name] = state.to_dict()
    return SimulationSnapshot(
        created_at=_utcnow_naive(),
        environment_state=deepcopy(environment_state),
        agent_states=agent_snapshot,
    )


def _validate_max_bytes(max_bytes: Optional[int]) -> None:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer or None")


def save_snapshot(
    snapshot: SimulationSnapshot, path: Path, *, max_bytes: Optional[int] = None
) -> None:
    """Atomically save UTF-8 JSON without materialising the complete snapshot.

    ``max_bytes`` must be a positive integer or None (unlimited). JSON is
    encoded incrementally into a temporary file beside the destination and
    atomically installed with :func:`os.replace` only after serialization,
    size validation, flush, and fsync succeed. Existing checkpoints therefore
    survive serialization, staging-write, and replacement failures.

    This bounds serialization-buffer memory, but it requires temporary disk
    space on the destination filesystem roughly equal to the new checkpoint
    and does not bound the snapshot object graph, deep-copy creation, decoded
    loads, or the size of an individual JSON scalar. ``_serialisable()`` is
    the shared representation point for both this function and ``to_json()``.
    """
    _validate_max_bytes(max_bytes)
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


def load_snapshot(path: Path, *, max_bytes: Optional[int] = None) -> SimulationSnapshot:
    """Load UTF-8 JSON with an optional positive byte limit (None is unlimited).

    Limited loads read in 64 KiB chunks (at most limit + 1 bytes total) and
    reject overflow before decoding or parsing, so a generous limit never
    triggers a proportional preallocation. The limit does not bound the memory
    used by the decoded object graph.
    """
    _validate_max_bytes(max_bytes)
    raw = b"" if max_bytes is None else None
    with path.open("rb") as source:
        if max_bytes is None:
            raw = source.read()
        else:
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = source.read(min(remaining, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("Snapshot exceeds max_bytes")
    return SimulationSnapshot.from_json(raw.decode("utf-8"))


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
        return asdict(cast(Any, value))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
