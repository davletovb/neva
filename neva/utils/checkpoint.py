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

from neva.memory import (
    AdaptiveConversationMemory,
    CompositeMemory,
    MemoryRecord,
    ShortTermMemory,
    SummaryMemory,
    VectorStoreMemory,
)
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


def _record(record: MemoryRecord) -> Dict[str, Any]:
    return _records([record])[0]


def _load_record(record: Dict[str, Any]) -> MemoryRecord:
    return _load_records([record])[0]


def _capture_budget(budget: Any) -> Any:
    if budget is None:
        return None
    return {
        "max_records": budget.max_records,
        "max_tokens": budget.max_tokens,
        "max_embeddings": budget.max_embeddings,
        "embedding_calls": budget._embedding_calls,
    }


def _restore_budget(budget: Any, state: Any) -> None:
    if state is None:
        if budget is not None:
            raise ValueError("Checkpoint memory budget configuration does not match")
        return
    if budget is None:
        raise ValueError("Checkpoint memory budget configuration does not match")
    configured = (budget.max_records, budget.max_tokens, budget.max_embeddings)
    saved = (state["max_records"], state["max_tokens"], state["max_embeddings"])
    if configured != saved:
        raise ValueError("Checkpoint memory budget configuration does not match")
    budget._embedding_calls = int(state["embedding_calls"])


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
    elif type(memory) is VectorStoreMemory:
        state.update(
            top_k=memory._top_k,
            counter=memory._counter,
            vectors=[
                {
                    "index": index,
                    "record": _record(record),
                    "vector": list(vector),
                }
                for index, record, vector in memory._vectors
            ],
        )
    elif type(memory) is AdaptiveConversationMemory:
        state.update(
            short_term_capacity=memory._short_term_capacity,
            semantic_top_k=memory._semantic_top_k,
            initial_summary=memory._initial_summary,
            has_embedder=memory._embedder is not None,
            id_counter=memory._id_counter,
            history=[
                {"id": record_id, "record": _record(record)}
                for record_id, record in memory._history
            ],
            token_counts=[
                {"id": record_id, "tokens": tokens}
                for record_id, tokens in sorted(memory._token_counts.items())
            ],
            vector_cache=[
                {"id": record_id, "vector": list(vector)}
                for record_id, vector in sorted(memory._vector_cache.items())
            ],
            budget=_capture_budget(memory._budget),
            short_term=_capture_memory(memory._short_term),
            summary=_capture_memory(memory._summary),
        )
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
    elif type(memory) is VectorStoreMemory:
        if state["top_k"] != memory._top_k:
            raise ValueError("Checkpoint vector memory configuration does not match")
        memory._counter = int(state["counter"])
        memory._vectors = [
            (
                int(item["index"]),
                _load_record(item["record"]),
                tuple(float(value) for value in item["vector"]),
            )
            for item in state["vectors"]
        ]
    elif type(memory) is AdaptiveConversationMemory:
        configured = (
            memory._short_term_capacity,
            memory._semantic_top_k,
            memory._initial_summary,
            memory._embedder is not None,
        )
        saved = (
            state["short_term_capacity"],
            state["semantic_top_k"],
            state["initial_summary"],
            state["has_embedder"],
        )
        if configured != saved:
            raise ValueError("Checkpoint adaptive memory configuration does not match")
        _restore_budget(memory._budget, state["budget"])
        memory._id_counter = int(state["id_counter"])
        memory._history = [
            (int(item["id"]), _load_record(item["record"])) for item in state["history"]
        ]
        memory._token_counts = {
            int(item["id"]): int(item["tokens"]) for item in state["token_counts"]
        }
        memory._vector_cache = {
            int(item["id"]): tuple(float(value) for value in item["vector"])
            for item in state["vector_cache"]
        }
        by_id = {record_id: record for record_id, record in memory._history}
        memory._semantic_entries = [
            (record_id, by_id[record_id], vector)
            for record_id, vector in memory._vector_cache.items()
            if record_id in by_id
        ]
        _restore_memory(memory._short_term, state["short_term"])
        _restore_memory(memory._summary, state["summary"])
    elif callable(getattr(memory, "restore_checkpoint_state", None)):
        memory.restore_checkpoint_state(deepcopy(state["custom"]))
    else:
        raise ValueError("Memory requires restore_checkpoint_state hook")
    memory.label = state["label"]


# Fields handled explicitly by capture/restore or intentionally preserved
# across restore (failure_log is external durable storage, not checkpoint state).
_ENV_FIELDS = {"state", "scheduler", "agents", "conversation_id", "failure_log"}


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
