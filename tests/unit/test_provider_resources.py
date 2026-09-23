import multiprocessing
import sqlite3
import threading
import time

import pytest

from neva.utils.exceptions import (
    ConfigurationError,
    RateLimiterCancelledError,
    SpendBudgetExceededError,
)
from neva.utils.provider_resources import (
    ProviderResourceCoordinator,
    _clear_provider_resource_registry_for_tests,
    account_scope,
    shared_provider_resources,
)


def _child_acquire(path, queue):
    coordinator = ProviderResourceCoordinator(
        scope="shared",
        rate=None,
        max_concurrency=1,
        state_path=path,
        poll_interval=0.02,
        lease_ttl=10.0,
    )
    queue.put("started")
    permit = coordinator.acquire()
    queue.put("acquired")
    coordinator.release(permit)


@pytest.fixture(autouse=True)
def clear_registry():
    _clear_provider_resource_registry_for_tests()
    yield
    _clear_provider_resource_registry_for_tests()


def test_account_scope_is_stable_and_does_not_expose_key():
    first = account_scope(provider="openai", api_key="secret-key", api_base=None)
    second = account_scope(provider="OPENAI", api_key="secret-key", api_base=None)

    assert first == second
    assert "secret-key" not in first
    assert first.startswith("openai:")


def test_process_registry_reuses_same_account_coordinator():
    first = shared_provider_resources(
        provider="openai",
        api_key="same",
        max_concurrency=2,
    )
    second = shared_provider_resources(
        provider="openai",
        api_key="same",
        max_concurrency=2,
    )
    other = shared_provider_resources(
        provider="openai",
        api_key="different",
        max_concurrency=2,
    )

    assert first is second
    assert first is not other


def test_process_registry_rejects_conflicting_limits_for_same_account():
    shared_provider_resources(
        provider="openai",
        api_key="same",
        rate=10,
        max_concurrency=1,
        max_cost=1.0,
    )

    with pytest.raises(ConfigurationError, match="different limits"):
        shared_provider_resources(
            provider="openai",
            api_key="same",
            rate=10,
            max_concurrency=2,
            max_cost=1.0,
        )


def test_local_fifo_concurrency_order():
    coordinator = ProviderResourceCoordinator(
        scope="fifo",
        rate=None,
        max_concurrency=1,
        poll_interval=0.01,
    )
    first = coordinator.acquire()
    order = []
    entered = threading.Event()

    def worker(name):
        entered.set()
        permit = coordinator.acquire()
        order.append(name)
        time.sleep(0.01)
        coordinator.release(permit)

    a = threading.Thread(target=worker, args=("a",))
    b = threading.Thread(target=worker, args=("b",))
    a.start()
    assert entered.wait(timeout=1)
    time.sleep(0.03)
    b.start()
    time.sleep(0.03)
    coordinator.release(first)
    a.join(timeout=2)
    b.join(timeout=2)

    assert not a.is_alive()
    assert not b.is_alive()
    assert order == ["a", "b"]


def test_cancelled_waiter_does_not_block_queue():
    coordinator = ProviderResourceCoordinator(
        scope="cancel",
        rate=None,
        max_concurrency=1,
        poll_interval=0.01,
    )
    first = coordinator.acquire()
    cancel = threading.Event()
    outcome = []

    def cancelled_waiter():
        try:
            coordinator.acquire(cancel_event=cancel)
        except RateLimiterCancelledError:
            outcome.append("cancelled")

    worker = threading.Thread(target=cancelled_waiter)
    worker.start()
    time.sleep(0.03)
    cancel.set()
    worker.join(timeout=1)
    coordinator.release(first)

    assert outcome == ["cancelled"]
    second = coordinator.acquire()
    coordinator.release(second)


def test_spend_reservation_prevents_concurrent_oversubscription():
    coordinator = ProviderResourceCoordinator(
        scope="spend",
        rate=None,
        max_concurrency=2,
        max_cost=1.0,
    )
    first = coordinator.acquire(reserve_cost=0.7)

    with pytest.raises(SpendBudgetExceededError):
        coordinator.acquire(reserve_cost=0.4)

    coordinator.release(first, actual_cost=0.5)
    assert coordinator.spent == pytest.approx(0.5)
    assert coordinator.reserved == 0.0
    assert coordinator.remaining == pytest.approx(0.5)


def test_authoritative_reconciliation_replaces_completed_estimate():
    coordinator = ProviderResourceCoordinator(
        scope="billing",
        rate=None,
        max_concurrency=1,
        max_cost=2.0,
    )
    permit = coordinator.acquire(reserve_cost=1.0)
    coordinator.release(permit, actual_cost=0.4)
    coordinator.reconcile_spend(0.75)

    assert coordinator.spent == pytest.approx(0.75)
    assert coordinator.remaining == pytest.approx(1.25)


def test_sqlite_instances_share_concurrency_state(tmp_path):
    path = tmp_path / "provider.sqlite3"
    first = ProviderResourceCoordinator(
        scope="shared",
        rate=None,
        max_concurrency=1,
        state_path=path,
        poll_interval=0.01,
    )
    second = ProviderResourceCoordinator(
        scope="shared",
        rate=None,
        max_concurrency=1,
        state_path=path,
        poll_interval=0.01,
    )
    permit = first.acquire()
    acquired = threading.Event()

    def worker():
        child = second.acquire()
        acquired.set()
        second.release(child)

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.05)
    assert not acquired.is_set()
    first.release(permit)
    assert acquired.wait(timeout=2)
    thread.join(timeout=2)


def test_sqlite_coordinates_separate_processes(tmp_path):
    path = str(tmp_path / "provider.sqlite3")
    parent = ProviderResourceCoordinator(
        scope="shared",
        rate=None,
        max_concurrency=1,
        state_path=path,
        poll_interval=0.02,
        lease_ttl=10.0,
    )
    permit = parent.acquire()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_child_acquire, args=(path, queue))
    process.start()
    try:
        assert queue.get(timeout=5) == "started"
        with pytest.raises(Exception):
            queue.get(timeout=0.2)
        parent.release(permit)
        assert queue.get(timeout=5) == "acquired"
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_sqlite_scope_configuration_mismatch_fails_closed(tmp_path):
    path = tmp_path / "provider.sqlite3"
    ProviderResourceCoordinator(
        scope="same",
        rate=10,
        per=60,
        max_concurrency=2,
        max_cost=1.0,
        state_path=path,
    )
    with pytest.raises(ConfigurationError, match="different limits"):
        ProviderResourceCoordinator(
            scope="same",
            rate=11,
            per=60,
            max_concurrency=2,
            max_cost=1.0,
            state_path=path,
        )


def test_cancelled_sqlite_waiter_does_not_block_queue(tmp_path):
    path = tmp_path / "provider.sqlite3"
    coordinator = ProviderResourceCoordinator(
        scope="cancel-sqlite",
        rate=None,
        max_concurrency=1,
        state_path=path,
        poll_interval=0.01,
        lease_ttl=10.0,
    )
    first = coordinator.acquire()
    cancel = threading.Event()
    outcome = []

    def cancelled_waiter():
        try:
            coordinator.acquire(cancel_event=cancel)
        except RateLimiterCancelledError:
            outcome.append("cancelled")

    worker = threading.Thread(target=cancelled_waiter)
    worker.start()
    time.sleep(0.05)
    cancel.set()
    worker.join(timeout=2)
    coordinator.release(first)

    assert not worker.is_alive()
    assert outcome == ["cancelled"]

    acquired = []

    def next_waiter():
        permit = coordinator.acquire()
        acquired.append(True)
        coordinator.release(permit)

    next_worker = threading.Thread(target=next_waiter)
    next_worker.start()
    next_worker.join(timeout=2)

    assert not next_worker.is_alive()
    assert acquired == [True]



def test_sqlite_waiter_cleanup_retries_and_surfaces_failure(tmp_path, monkeypatch):
    coordinator = ProviderResourceCoordinator(
        scope="cleanup-failure",
        rate=None,
        max_concurrency=1,
        state_path=tmp_path / "provider.sqlite3",
        poll_interval=0.001,
    )
    attempts = []

    def fail_connect():
        attempts.append(True)
        raise sqlite3.OperationalError("database locked")

    monkeypatch.setattr(coordinator, "_connect", fail_connect)

    with pytest.raises(sqlite3.OperationalError, match="database locked"):
        coordinator._remove_sqlite_waiter("stale-owner")

    assert len(attempts) == 3
