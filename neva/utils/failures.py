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
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

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
    attempt: int = 1
    max_attempts: int = 1
    action: str = ""
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.action:
            object.__setattr__(self, "action", self.policy)

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
        attempt = payload.get("attempt", 1)
        max_attempts = payload.get("max_attempts", 1)
        action = payload.get("action", policy)
        truncated = payload.get("truncated", False)
        if type(attempt) is not int or attempt <= 0:
            raise ValueError("failure record attempt must be a positive integer")
        if type(max_attempts) is not int or max_attempts < attempt:
            raise ValueError("failure record max_attempts must be >= attempt")
        if action not in {"retry", "raise", "return"}:
            raise ValueError("failure record action must be 'retry', 'raise', or 'return'")
        if type(truncated) is not bool:
            raise ValueError("failure record truncated must be a boolean")
        return cls(
            timestamp=float(timestamp),
            environment=payload["environment"],
            error_type=payload["error_type"],
            error_message=payload["error_message"],
            policy=policy,
            conversation_id=payload.get("conversation_id"),
            agent_name=payload.get("agent_name"),
            context=payload.get("context"),
            attempt=attempt,
            max_attempts=max_attempts,
            action=action,
            truncated=truncated,
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
    Appends, loads, and size-based rotation are serialized by a sibling
    advisory lock file, so multiple processes using FailureLog coordinate the
    same active/backup set instead of racing rotation. Rotated files use
    ``<path>.1``, ``<path>.2``, ... . ``load()`` reads
    retained backups oldest-first only when the reader is also configured with
    rotation and a sufficient ``backup_count``; a default ``FailureLog(path)``
    intentionally reads only the active file.

    ``rotate_bytes`` controls file rotation. ``max_record_bytes`` is an
    independent optional hard ceiling: oversized records first drop raw context
    and then UTF-8-safe truncate the diagnostic message, marking
    ``truncated=True``. If the structural metadata alone cannot fit, append
    fails rather than writing an over-limit record.
    Records live in this external file and are not part of simulation
    checkpoints.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        *,
        include_context: bool = False,
        fsync: bool = True,
        rotate_bytes: Optional[int] = None,
        backup_count: int = 3,
        max_record_bytes: Optional[int] = None,
    ) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("path must be a string or path-like object")
        candidate = Path(path)
        if candidate.exists() and candidate.is_dir():
            raise ValueError("path must point to a file, not a directory")
        if rotate_bytes is not None and (type(rotate_bytes) is not int or rotate_bytes <= 0):
            raise ValueError("rotate_bytes must be a positive integer or None")
        if type(backup_count) is not int or backup_count < 0:
            raise ValueError("backup_count must be a non-negative integer")
        if max_record_bytes is not None and (
            type(max_record_bytes) is not int or max_record_bytes <= 0
        ):
            raise ValueError("max_record_bytes must be a positive integer or None")
        self.path = candidate
        self.include_context = bool(include_context)
        self.fsync = bool(fsync)
        self.rotate_bytes = rotate_bytes
        self.backup_count = backup_count
        self.max_record_bytes = max_record_bytes
        self._retention_pruned = False
        self._lock = threading.Lock()

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.lock")

    @contextmanager
    def _process_lock(self) -> Iterator[None]:
        """Serialize append/rotation/load across processes sharing this path."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as handle:
            if os.name == "nt":  # pragma: no cover - platform-specific branch.
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _encode_record(failure: FailureRecord) -> bytes:
        return (json.dumps(failure.to_dict(), sort_keys=True) + "\n").encode("utf-8")

    def _bounded_record(self, failure: FailureRecord) -> tuple[FailureRecord, bytes]:
        encoded = self._encode_record(failure)
        if self.max_record_bytes is None or len(encoded) <= self.max_record_bytes:
            return failure, encoded

        failure = replace(failure, context=None, truncated=True)
        encoded = self._encode_record(failure)
        if len(encoded) <= self.max_record_bytes:
            return failure, encoded

        original = failure.error_message
        suffix = "...[truncated]"
        low, high = 0, len(original)
        best_record: Optional[FailureRecord] = None
        best_bytes: Optional[bytes] = None
        while low <= high:
            middle = (low + high) // 2
            candidate = replace(failure, error_message=original[:middle] + suffix)
            candidate_bytes = self._encode_record(candidate)
            if len(candidate_bytes) <= self.max_record_bytes:
                best_record = candidate
                best_bytes = candidate_bytes
                low = middle + 1
            else:
                high = middle - 1
        if best_record is None or best_bytes is None:
            raise ValueError("failure record metadata exceeds max_record_bytes")
        return best_record, best_bytes

    def _rotated_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def _prune_backups(self) -> None:
        prefix = f"{self.path.name}."
        for candidate in self.path.parent.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            suffix = candidate.name[len(prefix) :]
            if suffix.isascii() and suffix.isdigit() and int(suffix) > self.backup_count:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    def _rotate(self) -> None:
        if not self.path.exists():
            return
        if self.backup_count == 0:
            self.path.unlink()
            return

        oldest = self._rotated_path(self.backup_count)
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source = self._rotated_path(index)
            if source.exists():
                os.replace(source, self._rotated_path(index + 1))
        os.replace(self.path, self._rotated_path(1))

    def _retained_paths(self) -> List[Path]:
        paths: List[Path] = []
        if self.rotate_bytes is not None:
            for index in range(self.backup_count, 0, -1):
                candidate = self._rotated_path(index)
                if candidate.exists():
                    paths.append(candidate)
        if self.path.exists():
            paths.append(self.path)
        return paths

    def append(self, failure: FailureRecord) -> None:
        """Durably append one failure record."""

        if not isinstance(failure, FailureRecord):
            raise TypeError("failure must be a FailureRecord")
        if not self.include_context and failure.context is not None:
            failure = replace(failure, context=None)
        failure, encoded = self._bounded_record(failure)

        with self._lock:
            with self._process_lock():
                if self.rotate_bytes is not None and not self._retention_pruned:
                    self._prune_backups()
                    self._retention_pruned = True

                separator = b""
                current_size = self.path.stat().st_size if self.path.exists() else 0
                if current_size > 0:
                    with self.path.open("rb") as existing:
                        existing.seek(-1, os.SEEK_END)
                        if existing.read(1) != b"\n":
                            separator = b"\n"

                if (
                    self.rotate_bytes is not None
                    and current_size > 0
                    and current_size + len(separator) + len(encoded) > self.rotate_bytes
                ):
                    self._rotate()
                    separator = b""

                with self.path.open("ab") as handle:
                    if separator:
                        handle.write(separator)
                    handle.write(encoded)
                    handle.flush()
                    if self.fsync:
                        os.fsync(handle.fileno())

    def load(self) -> List[FailureRecord]:
        """Return all readable records; malformed lines are skipped with a warning."""

        records: List[FailureRecord] = []
        with self._lock:
            with self._process_lock():
                sources = self._retained_paths()
                payloads = [(source, source.read_bytes()) for source in sources]

        for source, payload in payloads:
            for number, raw in enumerate(payload.splitlines(), start=1):
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    logger.warning(
                        "Skipping undecodable failure record at %s line %d",
                        source,
                        number,
                    )
                    continue
                if not line:
                    continue
                try:
                    records.append(FailureRecord.from_dict(json.loads(line)))
                except (ValueError, KeyError, TypeError):
                    logger.warning(
                        "Skipping malformed failure record at %s line %d",
                        source,
                        number,
                    )
                    continue
        return records
