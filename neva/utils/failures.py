"""Durable failure records for unattended runs.

Turn failures handled by an environment's error policy are easy to miss in
logs; :class:`FailureLog` persists them as append-only JSON lines so they
survive crashes and restarts and can be inspected or replayed later.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["FailureLog", "FailureRecord"]


@dataclass(frozen=True)
class FailureRecord:
    """One handled turn failure.

    ``policy`` is the effective disposition that handled the failure
    ("raise" or "return"). ``context`` is the prompt the agent was given and
    is only stored when the owning :class:`FailureLog` opts in.
    """

    timestamp: float
    environment: str
    error_type: str
    error_message: str
    policy: str
    conversation_id: Optional[str] = None
    agent_name: Optional[str] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FailureRecord":
        if not isinstance(payload, dict):
            raise TypeError("failure record payload must be a dict")
        policy = payload["policy"]
        if policy not in {"raise", "return"}:
            raise ValueError("failure record policy must be 'raise' or 'return'")
        timestamp = payload["timestamp"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("failure record timestamp must be a finite non-negative number")
        return cls(
            timestamp=float(timestamp),
            environment=payload["environment"],
            error_type=payload["error_type"],
            error_message=payload["error_message"],
            policy=policy,
            conversation_id=payload.get("conversation_id"),
            agent_name=payload.get("agent_name"),
            context=payload.get("context"),
        )


class FailureLog:
    """Append-only JSONL store for :class:`FailureRecord` entries.

    Each record is written as one JSON object per line and flushed (and
    fsynced unless disabled) before ``append`` returns, so entries survive
    process crashes. A file that ends mid-line (for example after a crash
    during a previous write) is newline-separated before the next record so
    the new entry cannot merge into the corrupt tail.

    Reading is tolerant: truncated, undecodable, or otherwise malformed
    lines are skipped with a warning and the remaining records still load.

    Raw context is dropped unless ``include_context=True`` (a write-side
    gate: ``load()`` returns whatever context the file already contains).
    Appends are thread-safe within one instance; separate processes
    appending the same path rely on O_APPEND semantics for small lines.
    Records live in this external file and are not part of simulation
    checkpoints.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        *,
        include_context: bool = False,
        fsync: bool = True,
    ) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("path must be a string or path-like object")
        candidate = Path(path)
        if candidate.exists() and candidate.is_dir():
            raise ValueError("path must point to a file, not a directory")
        self.path = candidate
        self.include_context = bool(include_context)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()

    def append(self, failure: FailureRecord) -> None:
        """Durably append one failure record."""

        if not isinstance(failure, FailureRecord):
            raise TypeError("failure must be a FailureRecord")
        if not self.include_context and failure.context is not None:
            failure = replace(failure, context=None)
        line = json.dumps(failure.to_dict(), sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() > 0:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
                handle.write(line.encode("utf-8"))
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())

    def load(self) -> List[FailureRecord]:
        """Return all readable records; malformed lines are skipped with a warning."""

        records: List[FailureRecord] = []
        with self._lock:
            if not self.path.exists():
                return records
            with self.path.open("rb") as handle:
                for number, raw in enumerate(handle, start=1):
                    try:
                        line = raw.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        logger.warning("Skipping undecodable failure record at line %d", number)
                        continue
                    if not line:
                        continue
                    try:
                        records.append(FailureRecord.from_dict(json.loads(line)))
                    except (ValueError, KeyError, TypeError):
                        logger.warning("Skipping malformed failure record at line %d", number)
                        continue
        return records
