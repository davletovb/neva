"""Shared provider/account resource coordination for LLM calls.

The coordinator combines FIFO request-rate admission, a concurrency ceiling, and
optional spend reservations. Instances are shared automatically inside one
process; when state_path is configured the same scope is coordinated across
Python processes through SQLite transactions.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from neva.utils.exceptions import (
    ConfigurationError,
    RateLimiterCancelledError,
    SpendBudgetExceededError,
)

_PathLike = Union[str, "os.PathLike[str]"]
_EPSILON = 1e-9


@dataclass(frozen=True)
class ProviderPermit:
    """Admission token returned for one provider attempt."""

    owner: str
    reserved_cost: float


class ProviderResourceCoordinator:
    """Coordinate provider rate, concurrency, and spend for one account scope.

    With no state_path this object is thread-safe and loop-agnostic inside a
    process. With state_path all admission state is stored in SQLite, so
    independent processes that use the same scope and configuration coordinate
    the same provider/account budget.

    SQLite leases expire after lease_ttl to recover from crashed processes.
    A process holding an unusually long provider call should configure a TTL
    longer than its maximum request/retry window.
    """

    def __init__(
        self,
        *,
        scope: str,
        rate: Optional[int] = 60,
        per: float = 60.0,
        max_concurrency: Optional[int] = 8,
        max_cost: Optional[float] = None,
        state_path: Optional[_PathLike] = None,
        poll_interval: float = 0.05,
        lease_ttl: float = 3600.0,
    ) -> None:
        if not isinstance(scope, str) or not scope.strip():
            raise ConfigurationError("provider resource scope must be a non-empty string")
        if rate is not None and (type(rate) is not int or rate <= 0):
            raise ConfigurationError("provider resource rate must be a positive integer or None")
        if not isinstance(per, (int, float)) or isinstance(per, bool) or per <= 0:
            raise ConfigurationError("provider resource period must be positive")
        if max_concurrency is not None and (
            type(max_concurrency) is not int or max_concurrency <= 0
        ):
            raise ConfigurationError("provider max_concurrency must be a positive integer or None")
        if max_cost is not None and (
            isinstance(max_cost, bool)
            or not isinstance(max_cost, (int, float))
            or not math.isfinite(max_cost)
            or max_cost <= 0
        ):
            raise ConfigurationError("provider max_cost must be a finite positive number or None")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or poll_interval <= 0
        ):
            raise ConfigurationError("poll_interval must be positive")
        if isinstance(lease_ttl, bool) or not isinstance(lease_ttl, (int, float)) or lease_ttl <= 0:
            raise ConfigurationError("lease_ttl must be positive")

        self.scope = scope.strip()
        self.rate = rate
        self.per = float(per)
        self.max_concurrency = max_concurrency
        self.max_cost = float(max_cost) if max_cost is not None else None
        self.poll_interval = float(poll_interval)
        self.lease_ttl = float(lease_ttl)
        self.state_path = Path(state_path).expanduser() if state_path is not None else None

        self._condition = threading.Condition()
        self._queue = []  # type: list[str]
        self._active = set()  # type: set[str]
        self._allowance = float(rate or 0)
        self._last_check = time.monotonic()
        self._spent = 0.0
        self._reservations = {}  # type: dict[str, float]

        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialise_sqlite()

    @property
    def config_key(self) -> Tuple[Optional[int], float, Optional[int], Optional[float]]:
        return self.rate, self.per, self.max_concurrency, self.max_cost

    def _connect(self) -> sqlite3.Connection:
        if self.state_path is None:
            raise RuntimeError("SQLite coordination is not configured")
        connection = sqlite3.connect(str(self.state_path), timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialise_sqlite(self) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_scope_config (
                    scope TEXT PRIMARY KEY,
                    rate INTEGER,
                    period REAL NOT NULL,
                    max_concurrency INTEGER,
                    max_cost REAL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_scope_state (
                    scope TEXT PRIMARY KEY,
                    allowance REAL NOT NULL,
                    last_check REAL NOT NULL,
                    spent REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_waiters (
                    ticket INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    owner TEXT NOT NULL UNIQUE,
                    expires REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_leases (
                    scope TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    expires REAL NOT NULL,
                    PRIMARY KEY (scope, owner)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_reservations (
                    scope TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    amount REAL NOT NULL,
                    expires REAL NOT NULL,
                    PRIMARY KEY (scope, owner)
                )
                """
            )
            row = connection.execute(
                "SELECT rate, period, max_concurrency, max_cost "
                "FROM provider_scope_config WHERE scope = ?",
                (self.scope,),
            ).fetchone()
            expected = self.config_key
            if row is None:
                connection.execute(
                    "INSERT INTO provider_scope_config "
                    "(scope, rate, period, max_concurrency, max_cost) VALUES (?, ?, ?, ?, ?)",
                    (self.scope, *expected),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO provider_scope_state "
                    "(scope, allowance, last_check, spent) VALUES (?, ?, ?, 0.0)",
                    (self.scope, float(self.rate or 0), now),
                )
            else:
                actual = (row[0], float(row[1]), row[2], row[3])
                if actual != expected:
                    connection.rollback()
                    raise ConfigurationError(
                        "provider resource scope already exists with different limits"
                    )
            connection.commit()

    def _check_cancel(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RateLimiterCancelledError("Provider resource acquisition cancelled")

    def _wait(self, cancel_event: Optional[threading.Event], timeout: float) -> None:
        duration = max(0.001, min(timeout, self.poll_interval))
        if cancel_event is None:
            time.sleep(duration)
        elif cancel_event.wait(duration):
            raise RateLimiterCancelledError("Provider resource acquisition cancelled")

    def _refill_local(self, now: float) -> float:
        if self.rate is None:
            return 0.0
        elapsed = max(0.0, now - self._last_check)
        self._last_check = now
        self._allowance = min(
            float(self.rate),
            self._allowance + elapsed * (self.rate / self.per),
        )
        if self._allowance >= 1.0:
            return 0.0
        return (1.0 - self._allowance) * (self.per / self.rate)

    def _acquire_local(
        self,
        *,
        reserve_cost: float,
        cancel_event: Optional[threading.Event],
    ) -> ProviderPermit:
        owner = uuid.uuid4().hex
        with self._condition:
            self._queue.append(owner)

        while True:
            self._check_cancel(cancel_event)
            wait_for = self.poll_interval
            with self._condition:
                if cancel_event is not None and cancel_event.is_set():
                    if owner in self._queue:
                        self._queue.remove(owner)
                        self._condition.notify_all()
                    raise RateLimiterCancelledError("Provider resource acquisition cancelled")

                if self._queue and self._queue[0] == owner:
                    now = time.monotonic()
                    rate_wait = self._refill_local(now)
                    concurrency_available = (
                        self.max_concurrency is None or len(self._active) < self.max_concurrency
                    )
                    reserved_total = sum(self._reservations.values())
                    if (
                        self.max_cost is not None
                        and self._spent + reserved_total + reserve_cost > self.max_cost + _EPSILON
                    ):
                        self._queue.pop(0)
                        self._condition.notify_all()
                        raise SpendBudgetExceededError(
                            "provider spend reservation exceeds remaining shared budget"
                        )
                    rate_available = self.rate is None or self._allowance >= 1.0
                    if rate_available and concurrency_available:
                        if self.rate is not None:
                            self._allowance -= 1.0
                        self._active.add(owner)
                        if reserve_cost:
                            self._reservations[owner] = reserve_cost
                        self._queue.pop(0)
                        self._condition.notify_all()
                        return ProviderPermit(owner=owner, reserved_cost=reserve_cost)
                    if not rate_available:
                        wait_for = max(0.001, rate_wait)
            self._wait(cancel_event, wait_for)

    def _sqlite_cleanup(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "DELETE FROM provider_waiters WHERE scope = ? AND expires < ?",
            (self.scope, now),
        )
        connection.execute(
            "DELETE FROM provider_leases WHERE scope = ? AND expires < ?",
            (self.scope, now),
        )
        connection.execute(
            "DELETE FROM provider_reservations WHERE scope = ? AND expires < ?",
            (self.scope, now),
        )

    def _remove_sqlite_waiter(self, owner: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM provider_waiters WHERE scope = ? AND owner = ?",
                    (self.scope, owner),
                )
        except sqlite3.Error:
            pass

    def _acquire_sqlite(
        self,
        *,
        reserve_cost: float,
        cancel_event: Optional[threading.Event],
    ) -> ProviderPermit:
        owner = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO provider_waiters (scope, owner, expires) VALUES (?, ?, ?)",
                (self.scope, owner, now + self.lease_ttl),
            )

        try:
            while True:
                self._check_cancel(cancel_event)
                now = time.time()
                wait_for = self.poll_interval
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._sqlite_cleanup(connection, now)
                    connection.execute(
                        "UPDATE provider_waiters SET expires = ? " "WHERE scope = ? AND owner = ?",
                        (now + self.lease_ttl, self.scope, owner),
                    )
                    ticket_row = connection.execute(
                        "SELECT ticket FROM provider_waiters " "WHERE scope = ? AND owner = ?",
                        (self.scope, owner),
                    ).fetchone()
                    if ticket_row is None:
                        connection.rollback()
                        raise RateLimiterCancelledError(
                            "Provider resource waiter expired before admission"
                        )
                    first = connection.execute(
                        "SELECT MIN(ticket) FROM provider_waiters WHERE scope = ?",
                        (self.scope,),
                    ).fetchone()[0]
                    if first == ticket_row[0]:
                        allowance, last_check, spent = connection.execute(
                            "SELECT allowance, last_check, spent "
                            "FROM provider_scope_state WHERE scope = ?",
                            (self.scope,),
                        ).fetchone()
                        rate_wait = 0.0
                        if self.rate is not None:
                            elapsed = max(0.0, now - float(last_check))
                            allowance = min(
                                float(self.rate),
                                float(allowance) + elapsed * (self.rate / self.per),
                            )
                            if allowance < 1.0:
                                rate_wait = (1.0 - allowance) * (self.per / self.rate)
                        active = connection.execute(
                            "SELECT COUNT(*) FROM provider_leases WHERE scope = ?",
                            (self.scope,),
                        ).fetchone()[0]
                        reserved = connection.execute(
                            "SELECT COALESCE(SUM(amount), 0.0) "
                            "FROM provider_reservations WHERE scope = ?",
                            (self.scope,),
                        ).fetchone()[0]
                        if (
                            self.max_cost is not None
                            and float(spent) + float(reserved) + reserve_cost
                            > self.max_cost + _EPSILON
                        ):
                            connection.execute(
                                "DELETE FROM provider_waiters " "WHERE scope = ? AND owner = ?",
                                (self.scope, owner),
                            )
                            connection.commit()
                            raise SpendBudgetExceededError(
                                "provider spend reservation exceeds remaining shared budget"
                            )
                        rate_available = self.rate is None or float(allowance) >= 1.0
                        concurrency_available = (
                            self.max_concurrency is None or int(active) < self.max_concurrency
                        )
                        if rate_available and concurrency_available:
                            if self.rate is not None:
                                allowance = float(allowance) - 1.0
                            connection.execute(
                                "UPDATE provider_scope_state "
                                "SET allowance = ?, last_check = ? WHERE scope = ?",
                                (float(allowance), now, self.scope),
                            )
                            connection.execute(
                                "INSERT INTO provider_leases (scope, owner, expires) "
                                "VALUES (?, ?, ?)",
                                (self.scope, owner, now + self.lease_ttl),
                            )
                            if reserve_cost:
                                connection.execute(
                                    "INSERT INTO provider_reservations "
                                    "(scope, owner, amount, expires) VALUES (?, ?, ?, ?)",
                                    (
                                        self.scope,
                                        owner,
                                        reserve_cost,
                                        now + self.lease_ttl,
                                    ),
                                )
                            connection.execute(
                                "DELETE FROM provider_waiters " "WHERE scope = ? AND owner = ?",
                                (self.scope, owner),
                            )
                            connection.commit()
                            return ProviderPermit(owner=owner, reserved_cost=reserve_cost)
                        if self.rate is not None:
                            connection.execute(
                                "UPDATE provider_scope_state "
                                "SET allowance = ?, last_check = ? WHERE scope = ?",
                                (float(allowance), now, self.scope),
                            )
                            if not rate_available:
                                wait_for = max(0.001, rate_wait)
                    connection.commit()
                self._wait(cancel_event, wait_for)
        except Exception:
            self._remove_sqlite_waiter(owner)
            raise

    def acquire(
        self,
        *,
        reserve_cost: float = 0.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> ProviderPermit:
        """Acquire rate/concurrency admission and optionally reserve spend."""

        if (
            isinstance(reserve_cost, bool)
            or not isinstance(reserve_cost, (int, float))
            or not math.isfinite(reserve_cost)
            or reserve_cost < 0
        ):
            raise ConfigurationError("reserve_cost must be a finite non-negative number")
        self._check_cancel(cancel_event)
        if self.state_path is None:
            return self._acquire_local(
                reserve_cost=float(reserve_cost),
                cancel_event=cancel_event,
            )
        return self._acquire_sqlite(
            reserve_cost=float(reserve_cost),
            cancel_event=cancel_event,
        )

    def release(
        self,
        permit: ProviderPermit,
        *,
        actual_cost: Optional[float] = None,
    ) -> None:
        """Release concurrency and settle/release the permit spend reservation."""

        if not isinstance(permit, ProviderPermit):
            raise TypeError("permit must be a ProviderPermit")
        if actual_cost is not None and (
            isinstance(actual_cost, bool)
            or not isinstance(actual_cost, (int, float))
            or not math.isfinite(actual_cost)
            or actual_cost < 0
        ):
            raise ConfigurationError("actual_cost must be finite and non-negative")

        if self.state_path is None:
            overflow = False
            with self._condition:
                known = permit.owner in self._active or permit.owner in self._reservations
                if not known:
                    return
                self._active.discard(permit.owner)
                self._reservations.pop(permit.owner, None)
                if actual_cost is not None and self.max_cost is not None:
                    self._spent += float(actual_cost)
                    overflow = self._spent > self.max_cost + _EPSILON
                self._condition.notify_all()
            if overflow:
                raise SpendBudgetExceededError(
                    "provider call exceeded the shared spend budget after settlement"
                )
            return

        overflow = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease_exists = connection.execute(
                "SELECT 1 FROM provider_leases WHERE scope = ? AND owner = ?",
                (self.scope, permit.owner),
            ).fetchone()
            reservation_exists = connection.execute(
                "SELECT 1 FROM provider_reservations WHERE scope = ? AND owner = ?",
                (self.scope, permit.owner),
            ).fetchone()
            if lease_exists is None and reservation_exists is None:
                connection.commit()
                return
            connection.execute(
                "DELETE FROM provider_leases WHERE scope = ? AND owner = ?",
                (self.scope, permit.owner),
            )
            connection.execute(
                "DELETE FROM provider_reservations WHERE scope = ? AND owner = ?",
                (self.scope, permit.owner),
            )
            if actual_cost is not None and self.max_cost is not None:
                spent = float(
                    connection.execute(
                        "SELECT spent FROM provider_scope_state WHERE scope = ?",
                        (self.scope,),
                    ).fetchone()[0]
                )
                spent += float(actual_cost)
                connection.execute(
                    "UPDATE provider_scope_state SET spent = ? WHERE scope = ?",
                    (spent, self.scope),
                )
                overflow = spent > self.max_cost + _EPSILON
            connection.commit()
        if overflow:
            raise SpendBudgetExceededError(
                "provider call exceeded the shared spend budget after settlement"
            )

    def reconcile_spend(self, actual_spent: float) -> None:
        """Replace completed-spend estimates with an authoritative billing total."""

        if (
            isinstance(actual_spent, bool)
            or not isinstance(actual_spent, (int, float))
            or not math.isfinite(actual_spent)
            or actual_spent < 0
        ):
            raise ConfigurationError("actual_spent must be a finite non-negative number")
        if self.max_cost is None:
            raise ConfigurationError("cannot reconcile spend without a configured max_cost")
        if self.state_path is None:
            with self._condition:
                self._spent = float(actual_spent)
                self._condition.notify_all()
            return
        with self._connect() as connection:
            connection.execute(
                "UPDATE provider_scope_state SET spent = ? WHERE scope = ?",
                (float(actual_spent), self.scope),
            )

    @property
    def spent(self) -> float:
        if self.state_path is None:
            with self._condition:
                return self._spent
        with self._connect() as connection:
            row = connection.execute(
                "SELECT spent FROM provider_scope_state WHERE scope = ?",
                (self.scope,),
            ).fetchone()
            return float(row[0])

    @property
    def reserved(self) -> float:
        if self.state_path is None:
            with self._condition:
                return sum(self._reservations.values())
        with self._connect() as connection:
            value = connection.execute(
                "SELECT COALESCE(SUM(amount), 0.0) " "FROM provider_reservations WHERE scope = ?",
                (self.scope,),
            ).fetchone()[0]
            return float(value)

    @property
    def remaining(self) -> Optional[float]:
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - self.spent - self.reserved)


_registry_lock = threading.Lock()
_registry: "weakref.WeakValueDictionary[Tuple[object, ...], ProviderResourceCoordinator]" = (
    weakref.WeakValueDictionary()
)


def account_scope(
    *,
    provider: str,
    api_key: Optional[str],
    api_base: Optional[str],
    explicit_scope: Optional[str] = None,
) -> str:
    """Return a non-secret stable scope for one provider/account combination."""

    if explicit_scope is not None:
        if not isinstance(explicit_scope, str) or not explicit_scope.strip():
            raise ConfigurationError("provider_scope must be a non-empty string")
        return explicit_scope.strip()
    material = f"{provider.lower()}\0{api_base or ''}\0{api_key or '<missing>'}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"{provider.lower()}:{digest}"


def shared_provider_resources(
    *,
    provider: str,
    api_key: Optional[str],
    api_base: Optional[str] = None,
    provider_scope: Optional[str] = None,
    rate: Optional[int] = 60,
    per: float = 60.0,
    max_concurrency: Optional[int] = 8,
    max_cost: Optional[float] = None,
    state_path: Optional[_PathLike] = None,
) -> ProviderResourceCoordinator:
    """Return the process-shared coordinator for a provider/account scope.

    If state_path is omitted, NEVA_PROVIDER_COORDINATION_DB is honoured.
    Supplying the same SQLite path in separate processes extends the same scope
    across those processes.
    """

    resolved_scope = account_scope(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        explicit_scope=provider_scope,
    )
    resolved_path = state_path
    if resolved_path is None:
        configured = os.getenv("NEVA_PROVIDER_COORDINATION_DB")
        resolved_path = configured or None
    path_key = str(Path(resolved_path).expanduser().resolve()) if resolved_path else None
    key = (
        resolved_scope,
        rate,
        float(per),
        max_concurrency,
        float(max_cost) if max_cost is not None else None,
        path_key,
    )
    with _registry_lock:
        existing = _registry.get(key)
        if existing is not None:
            return existing
        coordinator = ProviderResourceCoordinator(
            scope=resolved_scope,
            rate=rate,
            per=per,
            max_concurrency=max_concurrency,
            max_cost=max_cost,
            state_path=resolved_path,
        )
        _registry[key] = coordinator
        return coordinator


def _clear_provider_resource_registry_for_tests() -> None:
    with _registry_lock:
        _registry.clear()
