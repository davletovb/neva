"""Utilities for persisting and restoring simulation state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ConversationTurn:
    speaker: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

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

    def record_turn(self, speaker: str, message: str) -> None:
        self.turns.append(ConversationTurn(speaker=speaker, message=message))

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ConversationState":
        state = cls(agent_name=str(payload["agent_name"]))
        raw_turns = payload.get("turns", [])
        if isinstance(raw_turns, list):
            for turn_payload in raw_turns:
                if isinstance(turn_payload, dict):
                    state.turns.append(ConversationTurn.from_dict(turn_payload))
        return state


@dataclass
class SimulationSnapshot:
    """Serializable view over environment state for persistence."""

    created_at: datetime
    environment_state: Dict[str, object]
    agent_states: Dict[str, Dict[str, object]]

    def to_json(self) -> str:
        serialisable = {
            "created_at": self.created_at.isoformat(),
            "environment_state": self.environment_state,
            "agent_states": self.agent_states,
        }
        return json.dumps(serialisable, default=_json_default, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "SimulationSnapshot":
        payload = json.loads(raw)
        created_at = datetime.fromisoformat(payload["created_at"])
        return cls(
            created_at=created_at,
            environment_state=payload["environment_state"],
            agent_states=payload["agent_states"],
        )


def create_snapshot(
    *,
    environment_state: Optional[Dict[str, object]] = None,
    agent_states: Optional[Iterable[ConversationState]] = None,
) -> SimulationSnapshot:
    environment_state = environment_state or {}
    if agent_states is None:
        agent_state_iter: Iterable[ConversationState] = ()
    else:
        agent_state_iter = agent_states

    agent_snapshot: Dict[str, Dict[str, object]] = {}
    for state in agent_state_iter:
        agent_snapshot[state.agent_name] = state.to_dict()
    return SimulationSnapshot(
        created_at=datetime.utcnow(),
        environment_state=environment_state,
        agent_states=agent_snapshot,
    )


def save_snapshot(snapshot: SimulationSnapshot, path: Path) -> None:
    path.write_text(snapshot.to_json(), encoding="utf-8")


def load_snapshot(path: Path) -> SimulationSnapshot:
    return SimulationSnapshot.from_json(path.read_text(encoding="utf-8"))


def _json_default(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
