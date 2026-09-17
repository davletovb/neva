"""Data-only scheduler checkpoint adapters, with explicit custom hooks."""
from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from typing import Any, Dict


def _kind(scheduler: Any) -> str:
    return f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"


def capture_scheduler(scheduler: Any) -> Any:
    from neva.schedulers import (
        EventDrivenScheduler,
        LeastRecentlyUsedScheduler,
        PriorityScheduler,
        RandomScheduler,
        RoundRobinScheduler,
        WeightedRandomScheduler,
    )

    if scheduler is None:
        return None
    payload: Dict[str, Any] = {
        "type": _kind(scheduler),
        "order": [a.name for a in scheduler.agents],
        "paused": [a.name for a in scheduler._paused_agents],
    }
    cls = type(scheduler)
    if cls is RoundRobinScheduler:
        payload["index"] = scheduler.current_index
    elif cls is LeastRecentlyUsedScheduler:
        payload["queue"] = [a.name for a in scheduler._queue]
    elif cls is EventDrivenScheduler:
        payload["events"] = [a.name for a in scheduler._event_queue]
    elif cls is PriorityScheduler:
        payload["priorities"] = [(p, a.name) for p, a in scheduler._queue]
    elif cls in (RandomScheduler, WeightedRandomScheduler):
        payload["rng"] = scheduler._rng.getstate()
        if cls is WeightedRandomScheduler:
            payload["weights"] = [(w, a.name) for w, a in scheduler._entries]
    elif callable(getattr(scheduler, "checkpoint_state", None)) and callable(
        getattr(scheduler, "restore_checkpoint_state", None)
    ):
        payload["custom"] = scheduler.checkpoint_state()
    else:
        raise ValueError(f"Scheduler {_kind(scheduler)} requires checkpoint hooks")
    observer = getattr(scheduler, "simulation_observer", None)
    if observer is not None and callable(getattr(observer, "checkpoint_state", None)):
        payload["observer"] = observer.checkpoint_state()
    return payload


def prepare_scheduler(scheduler: Any, payload: Any, agents: Dict[str, Any]) -> Any:
    """Validate and decode before changing live state; return an apply callback."""
    if payload is None:
        if scheduler is not None:
            raise ValueError("Checkpoint scheduler does not match")
        return lambda: None
    if scheduler is None or payload["type"] != _kind(scheduler):
        raise ValueError("Checkpoint scheduler does not match")
    order = payload["order"]
    if len(set(order)) != len(order) or set(order) - set(agents):
        raise ValueError("Checkpoint contains invalid scheduled agents")
    if set(payload["paused"]) - set(order):
        raise ValueError("Checkpoint contains unknown paused agents")
    updates: Dict[str, Any] = {
        "agents": [agents[name] for name in order],
        "_paused_agents": {agents[name] for name in payload["paused"]},
    }
    if "index" in payload:
        index = payload["index"]
        if not isinstance(index, int) or not 0 <= index < max(1, len(order)):
            raise ValueError("Checkpoint scheduler index is invalid")
        updates["current_index"] = index
    for key, attr in (("queue", "_queue"), ("events", "_event_queue")):
        if key in payload:
            if set(payload[key]) - set(order):
                raise ValueError("Checkpoint queue contains unknown agents")
            sequence = [agents[name] for name in payload[key]]
            updates[attr] = deque(sequence) if key == "events" else sequence
    for key, attr in (("priorities", "_queue"), ("weights", "_entries")):
        if key in payload:
            if {name for _, name in payload[key]} - set(order):
                raise ValueError("Checkpoint queue contains unknown agents")
            updates[attr] = [(value, agents[name]) for value, name in payload[key]]
    if "rng" in payload:
        rng = random.Random()
        state = payload["rng"]
        rng.setstate((state[0], tuple(state[1]), state[2]))
        updates["_rng"] = rng  # Do not mutate process-global random state.
    if "custom" in payload and not callable(getattr(scheduler, "restore_checkpoint_state", None)):
        raise ValueError("Scheduler requires restore_checkpoint_state hook")

    def apply() -> None:
        if "custom" in payload:
            scheduler.restore_checkpoint_state(deepcopy(payload["custom"]))
        scheduler.__dict__.update(updates)
        if "observer" in payload:
            scheduler.simulation_observer.restore_checkpoint_state(deepcopy(payload["observer"]))

    return apply
