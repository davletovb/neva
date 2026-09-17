"""Utilities for persisting and restoring simulation state."""

from __future__ import annotations

import json
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


@dataclass
class ConversationState:
    """Track the chronological conversation history for an agent."""

    agent_name: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns <= 0):
            raise ValueError("max_turns must be a positive integer or None")
        self.turns = list(self.turns)
        self._trim()

    def _trim(self) -> None:
        if self.max_turns is not None:
            del self.turns[: max(0, len(self.turns) - self.max_turns)]

    def record_turn(self, speaker: str, message: str) -> None:
        self.turns.append(ConversationTurn(speaker=speaker, message=message))
        self._trim()

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "turns": [turn.to_dict() for turn in self.turns],
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ConversationState":
        state = cls(
            agent_name=str(payload["agent_name"]),
            max_turns=cast(Optional[int], payload.get("max_turns")),
        )
        raw_turns = payload.get("turns", [])
        if isinstance(raw_turns, Iterable):
            for turn_payload in raw_turns:
                if isinstance(turn_payload, dict):
                    state.turns.append(
                        ConversationTurn.from_dict(cast(Dict[str, str], turn_payload))
                    )
            state._trim()
        return state


@dataclass
class SimulationSnapshot:
    """Serializable view over environment state for persistence."""

    created_at: datetime
    environment_state: Dict[str, object]
    agent_states: Dict[str, Dict[str, object]]
    version: int = 1
    runtime_state: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        serialisable = {
            "created_at": self.created_at.isoformat(),
            "environment_state": self.environment_state,
            "agent_states": self.agent_states,
            "version": self.version,
            "runtime_state": self.runtime_state,
        }
        return json.dumps(serialisable, default=_json_default, indent=2)

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
    """Save UTF-8 JSON, optionally rejecting oversized output before opening the file.

    ``max_bytes`` must be a positive integer or None (unlimited). Serialization
    still happens in memory; this is not a snapshot-creation or RAM limit.
    """
    _validate_max_bytes(max_bytes)
    raw = snapshot.to_json().encode("utf-8")
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("Snapshot exceeds max_bytes")
    path.write_bytes(raw)


def load_snapshot(path: Path, *, max_bytes: Optional[int] = None) -> SimulationSnapshot:
    """Load UTF-8 JSON with an optional positive byte limit (None is unlimited).

    Read at most limit + 1 bytes and reject overflow before decoding/parsing.
    The limit does not bound the memory used by the decoded object graph.
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
