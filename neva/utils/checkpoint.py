"""Versioned checkpoints without pickle or serialization of executable code.

Restore into an equivalently configured environment. Unsupported memory and
scheduler implementations must provide explicit checkpoint hooks.
"""
from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional
from uuid import UUID

from neva.memory import CompositeMemory, MemoryRecord, ShortTermMemory, SummaryMemory
from neva.utils.scheduler_checkpoint import capture_scheduler, prepare_scheduler


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _records(records: Any) -> Any:
    result = []
    for record in records:
        payload = asdict(record)
        payload["timestamp"] = record.timestamp.isoformat()
        result.append(payload)
    return result


def _load_records(records: Any) -> Any:
    from datetime import datetime

    return [
        MemoryRecord(**dict(r, timestamp=datetime.fromisoformat(r["timestamp"]))) for r in records
    ]


def _capture_memory(memory: Any) -> Any:
    if memory is None:
        return None
    state: Dict[str, Any] = {"type": _type_name(memory), "label": memory.label}
    if type(memory) is ShortTermMemory:
        state.update(capacity=memory.capacity, records=_records(memory._entries))
    elif type(memory) is SummaryMemory:
        state.update(summary=memory._summary, records=_records(memory._history))
    elif type(memory) is CompositeMemory:
        state["modules"] = [_capture_memory(m) for m in memory._modules]
    elif callable(getattr(memory, "checkpoint_state", None)):
        state["custom"] = memory.checkpoint_state()
    else:
        raise ValueError(f"Memory {_type_name(memory)} requires checkpoint hooks")
    return state


def _restore_memory(memory: Any, state: Any) -> None:
    if state is None:
        if memory is not None:
            raise ValueError("Checkpoint memory configuration does not match")
        return
    if memory is None or state["type"] != _type_name(memory):
        raise ValueError("Checkpoint memory type does not match")
    if type(memory) is ShortTermMemory:
        if state["capacity"] != memory.capacity:
            raise ValueError("Checkpoint memory capacity does not match")
        memory._entries = deque(_load_records(state["records"]), maxlen=memory.capacity)
    elif type(memory) is SummaryMemory:
        memory._summary = state["summary"]
        memory._history = _load_records(state["records"])
    elif type(memory) is CompositeMemory:
        if len(memory._modules) != len(state["modules"]):
            raise ValueError("Checkpoint memory modules do not match")
        for child, payload in zip(memory._modules, state["modules"]):
            _restore_memory(child, payload)
    elif callable(getattr(memory, "restore_checkpoint_state", None)):
        memory.restore_checkpoint_state(deepcopy(state["custom"]))
    else:
        raise ValueError("Memory requires restore_checkpoint_state hook")
    memory.label = state["label"]


_ENV_FIELDS = {"state", "scheduler", "agents", "conversation_id"}


def capture_runtime(environment: Any) -> Dict[str, Any]:
    scheduler = environment.scheduler
    names = [agent.name for agent in environment.agents]
    if len(set(names)) != len(names):
        raise ValueError("Checkpoint agent names must be unique")
    extra = {k: v for k, v in vars(environment).items() if k not in _ENV_FIELDS}
    runtime = {
        "environment_type": _type_name(environment),
        "environment_extra": extra,
        "conversation_id": environment.conversation_id,
        "agents": {
            agent.name: {
                "id": str(agent.id),
                "type": _type_name(agent),
                "memory": _capture_memory(agent.memory),
                "attributes": agent.attributes,
            }
            for agent in environment.agents
        },
        "scheduler": capture_scheduler(scheduler),
    }
    try:
        # A JSON roundtrip both isolates the state and rejects unsupported values.
        return json.loads(json.dumps(runtime, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Environment checkpoint fields must be JSON serializable") from exc


def restore_runtime(environment: Any, runtime: Optional[Dict[str, Any]]) -> None:
    if runtime is None:
        raise ValueError("Missing checkpoint runtime state")
    runtime = deepcopy(runtime)
    agents = {agent.name: agent for agent in environment.agents}
    if len(agents) != len(environment.agents) or set(agents) != set(runtime["agents"]):
        raise ValueError("Checkpoint agent population does not match")
    if runtime["environment_type"] != _type_name(environment):
        raise ValueError("Checkpoint environment type does not match")
    apply_scheduler = prepare_scheduler(environment.scheduler, runtime["scheduler"], agents)
    # Stage memory restoration so a mismatch cannot partially mutate live agents.
    memories = {}
    ids = {}
    for name, agent in agents.items():
        payload = runtime["agents"][name]
        if payload["type"] != _type_name(agent):
            raise ValueError("Checkpoint agent type does not match")
        ids[name] = UUID(payload["id"])
        memory = deepcopy(agent.memory)
        _restore_memory(memory, payload["memory"])
        memories[name] = memory
    apply_scheduler()
    for name, agent in agents.items():
        agent.id = ids[name]
        agent.attributes = runtime["agents"][name]["attributes"]
        agent.set_memory(memories[name])
        if agent.cache is not None:
            agent.cache.clear()
    for key in list(vars(environment)):
        if key not in _ENV_FIELDS:
            delattr(environment, key)
    environment.__dict__.update(runtime["environment_extra"])
    environment.conversation_id = runtime["conversation_id"]
