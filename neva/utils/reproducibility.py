"""Reproducible experiment manifests, seeding, and deterministic offline replay.

Live model providers are not deterministic experiment substrates: provider-side
model revisions, routing, sampling implementations, and service state can change
without a local code change. This module separates two guarantees:

* manifests record the inputs/configuration needed to explain a run;
* offline replay reproduces the exact recorded model boundary, validating prompt
  order/content before returning recorded responses.

Unified seeding covers Python's process RNG, optional NumPy/PyTorch RNGs, Neva
scheduler RNGs (including nested Composite schedulers), and custom agent or
scheduler set_seed hooks. It cannot retroactively change Python hash
randomization in the current process, so PYTHONHASHSEED is set for child
processes and that limitation is recorded.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from neva.utils.exceptions import RecordedReplayError, ReplayMismatchError, ReproducibilityError

if TYPE_CHECKING:  # pragma: no cover
    from neva.agents.base import AIAgent
    from neva.environments.base import Environment
    from neva.schedulers.base import Scheduler

_MANIFEST_VERSION = 1
_REPLAY_VERSION = 1
_DEFAULT_DEPENDENCIES = (
    "neva",
    "requests",
    "openai",
    "anthropic",
    "httpx",
    "google-generativeai",
    "transformers",
    "torch",
    "numpy",
    "faiss-cpu",
)


def _type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return _type_name(value)


def _stable_seed(seed: int, path: str) -> int:
    raw = f"{seed}:{path}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReproducibilityError("seed must be an integer")
    if seed < 0:
        raise ReproducibilityError("seed must be non-negative")
    return seed


def _json_native(value: Any, *, depth: int = 0) -> Any:
    """Convert configuration values to deterministic JSON-native data."""

    if depth > 8:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if isinstance(value, (datetime, date, time)):
        return {"type": _type_name(value), "isoformat": value.isoformat()}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {
                key: _json_native(item, depth=depth + 1)
                for key, item in sorted(value.items())
            }
        entries = [
            [
                _json_native(key, depth=depth + 1),
                _json_native(item, depth=depth + 1),
            ]
            for key, item in value.items()
        ]
        entries.sort(
            key=lambda pair: json.dumps(
                pair[0],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return {"__neva_mapping_entries__": entries}
    if isinstance(value, (list, tuple)):
        return [_json_native(item, depth=depth + 1) for item in value]
    if isinstance(value, set):
        converted = [_json_native(item, depth=depth + 1) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if callable(value):
        return {"callable": _callable_name(value)}
    return {"type": _type_name(value)}


def _dependency_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in sorted(set(names)):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass(frozen=True)
class SeedReport:
    """What the unified seeding operation actually seeded."""

    seed: int
    scheduler_seeds: Mapping[str, int]
    agent_seeds: Mapping[str, int]
    optional_libraries: Mapping[str, str]
    python_hash_seed: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "scheduler_seeds": dict(self.scheduler_seeds),
            "agent_seeds": dict(self.agent_seeds),
            "optional_libraries": dict(self.optional_libraries),
            "python_hash_seed": self.python_hash_seed,
        }


def _seed_scheduler(
    scheduler: "Scheduler",
    *,
    seed: int,
    path: str,
    seeded: Dict[str, int],
    seen: set[int],
) -> None:
    marker = id(scheduler)
    if marker in seen:
        return
    seen.add(marker)

    scheduler_seed = _stable_seed(seed, path)
    hook = getattr(scheduler, "set_seed", None)
    if callable(hook):
        hook(scheduler_seed)
        seeded[path] = scheduler_seed
    else:
        rng = getattr(scheduler, "_rng", None)
        if rng is random or isinstance(rng, random.Random):
            setattr(scheduler, "_rng", random.Random(scheduler_seed))
            seeded[path] = scheduler_seed

    children = getattr(scheduler, "_group_schedulers", None)
    if isinstance(children, Mapping):
        for group, child in sorted(children.items(), key=lambda pair: str(pair[0])):
            _seed_scheduler(
                child,
                seed=seed,
                path=f"{path}.group[{group}]",
                seeded=seeded,
                seen=seen,
            )


def seed_everything(
    seed: int,
    *,
    environment: Optional["Environment"] = None,
    scheduler: Optional["Scheduler"] = None,
    agents: Optional[Iterable["AIAgent"]] = None,
    optional_libraries: bool = True,
    set_child_hash_seed: bool = False,
) -> SeedReport:
    """Seed Python, optional numerical runtimes, and Neva-owned RNG hooks."""

    seed = _validate_seed(seed)
    if not isinstance(set_child_hash_seed, bool):
        raise ReproducibilityError("set_child_hash_seed must be a boolean")
    random.seed(seed)
    hash_seed = seed % (2**32)
    previous_hash_seed = os.environ.get("PYTHONHASHSEED")
    if set_child_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(hash_seed)

    optional_status: Dict[str, str] = {}
    if optional_libraries:
        for name in ("numpy", "torch"):
            try:
                module = importlib.import_module(name)
            except ImportError:
                optional_status[name] = "not-installed"
                continue
            if name == "numpy":
                numpy_seed = seed % (2**32)
                getattr(module, "random").seed(numpy_seed)
                optional_status[name] = f"seeded:{numpy_seed}"
            else:
                torch_seed = seed % (2**63)
                getattr(module, "manual_seed")(torch_seed)
                cuda = getattr(module, "cuda", None)
                if cuda is not None and callable(getattr(cuda, "manual_seed_all", None)):
                    cuda.manual_seed_all(torch_seed)
                optional_status[name] = f"seeded:{torch_seed}"
    else:
        optional_status = {"numpy": "skipped", "torch": "skipped"}

    schedulers: List["Scheduler"] = []
    if environment is not None:
        if environment.scheduler is not None:
            schedulers.append(environment.scheduler)
        if agents is None:
            agents = environment.agents
    if scheduler is not None and all(existing is not scheduler for existing in schedulers):
        schedulers.append(scheduler)

    scheduler_seeds: Dict[str, int] = {}
    seen: set[int] = set()
    for index, item in enumerate(schedulers):
        _seed_scheduler(
            item,
            seed=seed,
            path=f"scheduler[{index}]",
            seeded=scheduler_seeds,
            seen=seen,
        )

    agent_seeds: Dict[str, int] = {}
    for index, agent in enumerate(list(agents or ())):
        hook = getattr(agent, "set_seed", None)
        if callable(hook):
            agent_seed = _stable_seed(seed, f"agent[{index}]:{agent.name}")
            hook(agent_seed)
            agent_seeds[agent.name] = agent_seed

    return SeedReport(
        seed=seed,
        scheduler_seeds=scheduler_seeds,
        agent_seeds=agent_seeds,
        optional_libraries=optional_status,
        python_hash_seed=(
            (
                f"set-to-{hash_seed}-for-child-processes; current interpreter hash "
                "randomization was fixed at interpreter startup"
            )
            if set_child_hash_seed
            else (
                "unchanged-for-child-processes"
                if previous_hash_seed is None
                else f"unchanged-existing-value-{previous_hash_seed}"
            )
        ),
    )


def _scheduler_config(scheduler: Optional["Scheduler"]) -> Optional[Dict[str, Any]]:
    if scheduler is None:
        return None

    config: Dict[str, Any] = {
        "type": _type_name(scheduler),
        "agents": [agent.name for agent in scheduler.agents],
        "paused": sorted(agent.name for agent in scheduler._paused_agents),
    }

    if hasattr(scheduler, "current_index"):
        config["current_index"] = int(getattr(scheduler, "current_index"))
    if hasattr(scheduler, "_current_index"):
        config["current_index"] = int(getattr(scheduler, "_current_index"))
    if hasattr(scheduler, "_group_index"):
        config["group_index"] = int(getattr(scheduler, "_group_index"))

    for attribute, key in (("_queue", "queue"), ("_event_queue", "event_queue")):
        if not hasattr(scheduler, attribute):
            continue
        rendered = []
        for item in list(getattr(scheduler, attribute)):
            if isinstance(item, tuple) and len(item) == 2:
                rendered.append([_json_native(item[0]), getattr(item[1], "name", str(item[1]))])
            else:
                rendered.append(getattr(item, "name", _json_native(item)))
        config[key] = rendered

    entries = getattr(scheduler, "_entries", None)
    if isinstance(entries, list):
        config["weights"] = [[weight, agent.name] for weight, agent in entries]

    conditions = getattr(scheduler, "_conditions", None)
    if isinstance(conditions, Mapping):
        config["conditions"] = {
            agent.name: _callable_name(condition)
            for agent, condition in sorted(conditions.items(), key=lambda pair: pair[0].name)
        }

    children = getattr(scheduler, "_group_schedulers", None)
    if isinstance(children, Mapping):
        config["group_order"] = list(getattr(scheduler, "_group_order", []))
        membership = getattr(scheduler, "_group_membership", {})
        config["group_membership"] = {
            agent.name: group
            for agent, group in sorted(membership.items(), key=lambda pair: pair[0].name)
        }
        config["groups"] = {
            str(group): _scheduler_config(child)
            for group, child in sorted(children.items(), key=lambda pair: str(pair[0]))
        }

    rng = getattr(scheduler, "_rng", None)
    if rng is not None and callable(getattr(rng, "getstate", None)):
        state = repr(rng.getstate()).encode("utf-8")
        config["rng_state_sha256"] = hashlib.sha256(state).hexdigest()

    hook = getattr(scheduler, "reproducibility_config", None)
    if callable(hook):
        config["custom"] = _json_native(hook())

    return config


def _cache_policy(agent: "AIAgent") -> Dict[str, Any]:
    cache = agent.cache
    if cache is None:
        return {"enabled": False}

    policy: Dict[str, Any] = {
        "enabled": True,
        "type": _type_name(cache),
        "max_size": getattr(cache, "_max_size", None),
    }
    store = getattr(cache, "_store", None)
    lock = getattr(cache, "_lock", None)
    if isinstance(store, Mapping) and lock is not None:
        with lock:
            items = list(store.items())
        state_raw = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        policy["initial_entries"] = len(items)
        policy["state_sha256"] = hashlib.sha256(state_raw).hexdigest()
    return policy


def _without_runtime_timestamps(value: Any) -> Any:
    """Remove wall-clock record timestamps from captured initial memory state."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_timestamps(item)
            for key, item in value.items()
            if key != "timestamp"
        }
    if isinstance(value, list):
        return [_without_runtime_timestamps(item) for item in value]
    return value


def _memory_callable_config(memory: Any) -> Dict[str, Any]:
    if memory is None:
        return {}

    config: Dict[str, Any] = {}
    for attribute, key in (
        ("_summarizer", "summarizer"),
        ("_embedder", "embedder"),
    ):
        value = getattr(memory, attribute, None)
        if value is not None:
            config[key] = _callable_name(value)

    budget = getattr(memory, "_budget", None)
    if budget is not None:
        estimator = getattr(budget, "_token_estimator", None)
        if estimator is not None:
            config["budget_token_estimator"] = _callable_name(estimator)

    modules = getattr(memory, "_modules", None)
    if isinstance(modules, list):
        config["modules"] = [_memory_callable_config(module) for module in modules]
    return config


def _memory_record_config(record: Any) -> Dict[str, Any]:
    return {
        "speaker": record.speaker,
        "message": record.message,
        "metadata": _json_native(record.metadata),
    }


def _memory_budget_config(budget: Any) -> Any:
    if budget is None:
        return None
    return {
        "max_records": budget.max_records,
        "max_tokens": budget.max_tokens,
        "max_embeddings": budget.max_embeddings,
        "embedding_calls": getattr(budget, "_embedding_calls", 0),
        "token_estimator": _callable_name(getattr(budget, "_token_estimator")),
    }


def _capture_manifest_memory(memory: Any) -> Any:
    from neva.memory import (
        AdaptiveConversationMemory,
        CompositeMemory,
        FaissVectorStoreMemory,
        ShortTermMemory,
        SummaryMemory,
        VectorStoreMemory,
    )

    base = {"type": _type_name(memory), "label": memory.label}

    if isinstance(memory, ShortTermMemory):
        base.update(
            capacity=memory.capacity,
            records=[_memory_record_config(record) for record in memory._entries],
        )
        return base

    if isinstance(memory, SummaryMemory):
        base.update(
            summary=memory._summary,
            records=[_memory_record_config(record) for record in memory._history],
            summarizer=_callable_name(memory._summarizer),
        )
        return base

    if isinstance(memory, CompositeMemory):
        base["modules"] = [_capture_manifest_memory(module) for module in memory._modules]
        return base

    if isinstance(memory, FaissVectorStoreMemory):
        index_sha256 = None
        if memory._index is not None:
            serialized = memory._faiss.serialize_index(memory._index)
            raw = serialized.tobytes() if hasattr(serialized, "tobytes") else bytes(serialized)
            index_sha256 = hashlib.sha256(raw).hexdigest()
        base.update(
            top_k=memory._top_k,
            index_factory=memory._index_factory,
            normalize_embeddings=memory._normalize_embeddings,
            id_counter=memory._id_counter,
            order=list(memory._order),
            records=[
                {
                    "id": record_id,
                    "record": _memory_record_config(memory._records[record_id]),
                }
                for record_id in memory._order
                if record_id in memory._records
            ],
            index_sha256=index_sha256,
            embedder=_callable_name(memory._embedder),
        )
        return base

    if isinstance(memory, VectorStoreMemory):
        base.update(
            top_k=memory._top_k,
            counter=memory._counter,
            vectors=[
                {
                    "index": index,
                    "record": _memory_record_config(record),
                    "vector": list(vector),
                }
                for index, record, vector in memory._vectors
            ],
            embedder=_callable_name(memory._embedder),
        )
        return base

    if isinstance(memory, AdaptiveConversationMemory):
        base.update(
            short_term_capacity=memory._short_term_capacity,
            semantic_top_k=memory._semantic_top_k,
            initial_summary=memory._initial_summary,
            id_counter=memory._id_counter,
            history=[
                {"id": record_id, "record": _memory_record_config(record)}
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
            budget=_memory_budget_config(memory._budget),
            short_term=_capture_manifest_memory(memory._short_term),
            summary=_capture_manifest_memory(memory._summary),
            summarizer=_callable_name(memory._summarizer),
            embedder=(
                _callable_name(memory._embedder)
                if memory._embedder is not None
                else None
            ),
        )
        return base

    checkpoint_hook = getattr(memory, "checkpoint_state", None)
    if callable(checkpoint_hook):
        base["checkpoint_state"] = _json_native(checkpoint_hook())
        return base

    reproducibility_hook = getattr(memory, "reproducibility_config", None)
    if callable(reproducibility_hook):
        base["custom"] = _json_native(reproducibility_hook())
        return base

    raise ReproducibilityError(
        f"memory {_type_name(memory)} must expose checkpoint_state() or "
        "reproducibility_config() for reproducible manifests"
    )


def _memory_config(memory: Any) -> Dict[str, Any]:
    if memory is None:
        return {"type": None}
    return _json_native(_capture_manifest_memory(memory))


def _argument_schema_config(schema: Any) -> Any:
    if schema is None:
        return None

    fields = getattr(schema, "fields", None)
    allow_extra = getattr(schema, "allow_extra", None)
    if isinstance(fields, Mapping) and isinstance(allow_extra, bool):
        rendered = {}
        for name, spec in sorted(fields.items()):
            expected = getattr(spec, "type", None)
            expected_types = expected if isinstance(expected, tuple) else (expected,)
            rendered[name] = {
                "types": [_callable_name(item) for item in expected_types],
                "required": getattr(spec, "required", None),
                "min_length": getattr(spec, "min_length", None),
                "max_length": getattr(spec, "max_length", None),
                "min_value": getattr(spec, "min_value", None),
                "max_value": getattr(spec, "max_value", None),
                "choices": _json_native(getattr(spec, "choices", None)),
            }
        return {
            "type": _type_name(schema),
            "allow_extra": allow_extra,
            "fields": rendered,
        }

    hook = getattr(schema, "reproducibility_config", None)
    if callable(hook):
        return {"type": _type_name(schema), "custom": _json_native(hook())}
    return {
        "type": _type_name(schema),
        "state": _json_native(vars(schema)) if hasattr(schema, "__dict__") else None,
    }


def _tool_guard_config(guard: Any) -> Any:
    if guard is None:
        return None
    limits = guard.limits
    hook = getattr(guard, "reproducibility_config", None)
    config = {
        "type": _type_name(guard),
        "allowed_tools": (
            sorted(guard.allowed_tools) if guard.allowed_tools is not None else None
        ),
        "approve": _callable_name(guard.approve) if guard.approve is not None else None,
        "limits": {
            "timeout": limits.timeout,
            "max_output_chars": limits.max_output_chars,
            "max_concurrency": limits.max_concurrency,
            "isolate_process": limits.isolate_process,
            "max_memory_bytes": limits.max_memory_bytes,
        },
    }
    if callable(hook):
        config["custom"] = _json_native(hook())
    return config


def _agent_config(agent: "AIAgent") -> Dict[str, Any]:
    generation: Dict[str, Any] = {}
    for name in (
        "_max_output_tokens",
        "_max_context_chars",
        "_max_retries",
        "_retry_backoff",
        "_request_timeout",
    ):
        if hasattr(agent, name):
            generation[name.removeprefix("_")] = _json_native(getattr(agent, name))

    if hasattr(agent, "model_name"):
        generation.setdefault("max_length", 200)

    backend = agent.llm_backend
    state = agent.conversation_state
    conversation = {
        "max_turns": state.max_turns,
        "max_turn_bytes": state.max_turn_bytes,
        "max_history_bytes": state.max_history_bytes,
        "turns": [{"speaker": turn.speaker, "message": turn.message} for turn in state.turns],
    }
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "capabilities": list(tool.capabilities),
            "argument_schema": _argument_schema_config(tool.argument_schema),
            "tool_guard": _tool_guard_config(tool.tool_guard),
        }
        for tool in sorted(agent.tools, key=lambda item: item.name)
    ]
    memory_config = _memory_config(agent.memory)

    return {
        "name": agent.name,
        "type": _type_name(agent),
        "provider": getattr(agent, "provider", None),
        "model": getattr(agent, "model", getattr(agent, "model_name", None)),
        "api_base": getattr(agent, "api_base", None),
        "generation": generation,
        "prompt_validator": {
            "max_length": getattr(agent.prompt_validator, "max_length", None),
            "forbidden_patterns": [
                {"pattern": pattern.pattern, "flags": pattern.flags}
                for pattern in getattr(agent.prompt_validator, "_compiled_patterns", ())
            ],
        },
        "cache": _cache_policy(agent),
        "backend": _callable_name(backend) if backend is not None else None,
        "attributes": _json_native(dict(sorted(agent.attributes.items()))),
        "conversation": conversation,
        "memory": memory_config,
        "tool_guard": _tool_guard_config(agent.tool_guard),
        "tools": tools,
    }


def _environment_config(environment: "Environment") -> Dict[str, Any]:
    excluded = {
        "agents",
        "scheduler",
        "conversation_id",
        "failure_log",
        "recovery_policy",
        "state",
        "error_policy",
        "error_value",
    }
    public_config = {
        name: _json_native(value)
        for name, value in sorted(vars(environment).items())
        if not name.startswith("_") and name not in excluded
    }
    agent_error_policies = []
    configured_policies = getattr(environment, "_agent_error_policies", {})
    for index, agent in enumerate(environment.agents):
        policy = configured_policies.get(str(agent.id))
        if policy is not None:
            agent_error_policies.append(
                {
                    "agent_index": index,
                    "agent_name": agent.name,
                    "policy": _json_native(policy),
                }
            )

    config: Dict[str, Any] = {
        "type": _type_name(environment),
        "error_policy": environment.error_policy,
        "error_value": environment.error_value,
        "recovery_policy": _json_native(vars(environment.recovery_policy)),
        "agent_error_policies": agent_error_policies,
        "state": _json_native(environment.state),
        "public_config": public_config,
    }
    hook = getattr(environment, "reproducibility_config", None)
    if callable(hook):
        config["custom"] = _json_native(hook())
    return config


def _normalize_prompts(
    prompts: Optional[Mapping[str, str] | Sequence[str]],
) -> Dict[str, str]:
    if prompts is None:
        return {}
    if isinstance(prompts, Mapping):
        normalized: Dict[str, str] = {}
        for key, value in prompts.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ReproducibilityError("manifest prompts must map strings to strings")
            normalized[key] = value
        return dict(sorted(normalized.items()))
    if isinstance(prompts, str):
        raise ReproducibilityError("prompts must be a mapping or sequence of strings")
    normalized = {}
    for index, value in enumerate(prompts):
        if not isinstance(value, str):
            raise ReproducibilityError("manifest prompts must contain only strings")
        normalized[f"prompt_{index}"] = value
    return normalized


@dataclass(frozen=True)
class RunManifest:
    """Serializable experiment configuration used for audit and replay."""

    created_at: str
    seed: int
    prompts: Mapping[str, str]
    environment: Mapping[str, Any]
    scheduler: Optional[Mapping[str, Any]]
    agents: Sequence[Mapping[str, Any]]
    dependencies: Mapping[str, Optional[str]]
    runtime: Mapping[str, str]
    seed_report: Mapping[str, Any]
    metadata: Mapping[str, Any]
    reproducibility: Mapping[str, Any]
    version: int = _MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "seed": self.seed,
            "prompts": dict(self.prompts),
            "environment": deepcopy(dict(self.environment)),
            "scheduler": deepcopy(self.scheduler),
            "agents": deepcopy(list(self.agents)),
            "dependencies": dict(self.dependencies),
            "runtime": dict(self.runtime),
            "seed_report": deepcopy(dict(self.seed_report)),
            "metadata": deepcopy(dict(self.metadata)),
            "reproducibility": deepcopy(dict(self.reproducibility)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        version = payload.get("version", _MANIFEST_VERSION)
        if version != _MANIFEST_VERSION:
            raise ReproducibilityError(f"unsupported manifest version: {version}")
        try:
            prompts = _normalize_prompts(dict(payload.get("prompts", {})))
            environment = dict(payload.get("environment", {}))
            agents = list(payload.get("agents", []))
            dependencies = dict(payload.get("dependencies", {}))
            runtime = dict(payload.get("runtime", {}))
            seed_report = dict(payload.get("seed_report", {}))
            metadata = dict(payload.get("metadata", {}))
            reproducibility = dict(payload.get("reproducibility", {}))
            created_at = str(payload["created_at"])
            seed = _validate_seed(payload["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReproducibilityError("manifest has an invalid shape") from exc
        if not all(isinstance(agent, Mapping) for agent in agents):
            raise ReproducibilityError("manifest agents must be JSON objects")
        return cls(
            version=version,
            created_at=created_at,
            seed=seed,
            prompts=deepcopy(prompts),
            environment=deepcopy(environment),
            scheduler=deepcopy(payload.get("scheduler")),
            agents=deepcopy(agents),
            dependencies=deepcopy(dependencies),
            runtime=deepcopy(runtime),
            seed_report=deepcopy(seed_report),
            metadata=deepcopy(metadata),
            reproducibility=deepcopy(reproducibility),
        )

    def compatibility_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        for key in (
            "created_at",
            "dependencies",
            "runtime",
            "seed_report",
            "metadata",
            "reproducibility",
        ):
            payload.pop(key, None)
        return payload

    def audit_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.pop("created_at", None)
        return payload

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.compatibility_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def audit_fingerprint(self) -> str:
        raw = json.dumps(
            self.audit_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def save(self, path: Path | str) -> None:
        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> "RunManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReproducibilityError("manifest root must be a JSON object")
        return cls.from_dict(payload)


def create_run_manifest(
    environment: "Environment",
    *,
    seed: int,
    prompts: Optional[Mapping[str, str] | Sequence[str]] = None,
    seed_report: Optional[SeedReport] = None,
    dependencies: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> RunManifest:
    """Capture the configuration needed to explain or replay a simulation run."""

    seed = _validate_seed(seed)
    agent_names = [agent.name for agent in environment.agents]
    if len(agent_names) != len(set(agent_names)):
        raise ReproducibilityError("run manifests require unique agent names")
    agents = [_agent_config(agent) for agent in environment.agents]
    live_providers = sorted(
        {
            str(agent["provider"])
            for agent in agents
            if agent.get("provider") and agent.get("backend") is None
        }
    )
    notes = [
        "offline replay is deterministic at the recorded model-backend boundary "
        "when prompt order/content match",
        "PYTHONHASHSEED cannot change current-process hash randomization after interpreter startup",
    ]
    if live_providers:
        notes.append(
            "live provider responses are not guaranteed reproducible: provider model revisions, "
            "routing, sampling/runtime implementations, and service state are outside Neva control"
        )

    dependency_names = tuple(_DEFAULT_DEPENDENCIES if dependencies is None else dependencies)
    report = seed_report or SeedReport(
        seed=seed,
        scheduler_seeds={},
        agent_seeds={},
        optional_libraries={"numpy": "unknown", "torch": "unknown"},
        python_hash_seed="not recorded; call seed_everything() or prepare_reproducible_run()",
    )

    return RunManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        prompts=_normalize_prompts(prompts),
        environment=_environment_config(environment),
        scheduler=_scheduler_config(environment.scheduler),
        agents=agents,
        dependencies=_dependency_versions(dependency_names),
        runtime={
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        seed_report=report.to_dict(),
        metadata=_json_native(dict(metadata or {})),
        reproducibility={
            "offline_replay_boundary": "model prompt -> response/error",
            "live_providers": live_providers,
            "live_provider_exact_replay": False if live_providers else None,
            "notes": notes,
        },
    )


def prepare_reproducible_run(
    environment: "Environment",
    *,
    seed: int,
    prompts: Optional[Mapping[str, str] | Sequence[str]] = None,
    dependencies: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    optional_libraries: bool = True,
    set_child_hash_seed: bool = False,
) -> RunManifest:
    """Seed a run and immediately capture its manifest."""

    report = seed_everything(
        seed,
        environment=environment,
        optional_libraries=optional_libraries,
        set_child_hash_seed=set_child_hash_seed,
    )
    return create_run_manifest(
        environment,
        seed=seed,
        prompts=prompts,
        seed_report=report,
        dependencies=dependencies,
        metadata=metadata,
    )


def _record_digest(
    prompt: str,
    prompt_sha256: str,
    response: Optional[str],
    error_type: Optional[str],
    error_message: Optional[str],
) -> str:
    payload = {
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "response": response,
        "error_type": error_type,
        "error_message": error_message,
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _manifest_diff_paths(expected: Any, actual: Any, *, prefix: str = "") -> List[str]:
    differences: List[str] = []
    if type(expected) is not type(actual):
        return [prefix or "<root>"]
    if isinstance(expected, Mapping):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                differences.append(path)
                continue
            differences.extend(
                _manifest_diff_paths(expected[key], actual[key], prefix=path)
            )
            if len(differences) >= 12:
                break
        return differences
    if isinstance(expected, list):
        if len(expected) != len(actual):
            differences.append(f"{prefix}.length" if prefix else "length")
        for index, (left, right) in enumerate(zip(expected, actual)):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            differences.extend(_manifest_diff_paths(left, right, prefix=path))
            if len(differences) >= 12:
                break
        return differences
    if expected != actual:
        return [prefix or "<root>"]
    return []


@dataclass(frozen=True)
class ReplayRecord:
    """One replay-identity call recorded for deterministic offline replay."""

    prompt: str
    prompt_sha256: str
    response: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    record_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        digest = self.record_sha256 or _record_digest(
            self.prompt,
            self.prompt_sha256,
            self.response,
            self.error_type,
            self.error_message,
        )
        object.__setattr__(self, "record_sha256", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "response": self.response,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "record_sha256": self.record_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayRecord":
        if not isinstance(payload, Mapping):
            raise ReproducibilityError("replay record must be a JSON object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ReproducibilityError("replay record prompt must be a string")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if payload.get("prompt_sha256") != digest:
            raise ReproducibilityError("replay record prompt digest does not match prompt")
        response = payload.get("response")
        if response is not None and not isinstance(response, str):
            raise ReproducibilityError("replay record response must be a string or null")
        error_type = payload.get("error_type")
        error_message = payload.get("error_message")
        if error_type is not None and not isinstance(error_type, str):
            raise ReproducibilityError("replay record error_type must be a string or null")
        if error_message is not None and not isinstance(error_message, str):
            raise ReproducibilityError("replay record error_message must be a string or null")
        if response is not None and error_type is not None:
            raise ReproducibilityError("replay record cannot contain both response and error")
        if response is None and error_type is None:
            raise ReproducibilityError("replay record must contain a response or error")
        record_digest = payload.get("record_sha256")
        expected_record_digest = _record_digest(
            prompt,
            digest,
            response,
            error_type,
            error_message,
        )
        if record_digest != expected_record_digest:
            raise ReproducibilityError("replay record digest does not match record contents")
        return cls(
            prompt=prompt,
            prompt_sha256=digest,
            response=response,
            error_type=error_type,
            error_message=error_message,
            record_sha256=record_digest,
        )


class ReplayTape:
    """Thread-safe sequence of exact model-boundary identities and outcomes."""

    def __init__(
        self,
        records: Optional[Iterable[ReplayRecord]] = None,
        *,
        manifest_fingerprint: Optional[str] = None,
        manifest_compatibility: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._records: List[Optional[ReplayRecord]] = list(records or ())
        self.manifest_fingerprint = manifest_fingerprint
        self.manifest_compatibility = (
            deepcopy(dict(manifest_compatibility))
            if manifest_compatibility is not None
            else None
        )
        self._lock = threading.RLock()

    @classmethod
    def for_manifest(cls, manifest: RunManifest) -> "ReplayTape":
        return cls(
            manifest_fingerprint=manifest.fingerprint(),
            manifest_compatibility=manifest.compatibility_payload(),
        )

    @property
    def records(self) -> Sequence[ReplayRecord]:
        with self._lock:
            if any(record is None for record in self._records):
                raise ReproducibilityError("replay tape contains an in-flight recording")
            return tuple(record for record in self._records if record is not None)

    def _reserve(self) -> int:
        with self._lock:
            index = len(self._records)
            self._records.append(None)
            return index

    def _complete(self, index: int, record: ReplayRecord) -> None:
        with self._lock:
            if index >= len(self._records) or self._records[index] is not None:
                raise ReproducibilityError("replay recording slot is invalid or already complete")
            self._records[index] = record

    def recording_backend(
        self,
        backend: Callable[[str], str],
        *,
        identity: Optional[Callable[[str], str]] = None,
    ) -> Callable[[str], str]:
        """Record invocation order without holding the tape lock during model work."""

        if not callable(backend):
            raise ReproducibilityError("recording backend must be callable")
        if identity is not None and not callable(identity):
            raise ReproducibilityError("recording identity must be callable")
        identity_fn = identity or (lambda prompt: prompt)

        def _record(prompt: str) -> str:
            if not isinstance(prompt, str):
                raise ReproducibilityError("recorded model prompt must be a string")
            replay_identity = identity_fn(prompt)
            if not isinstance(replay_identity, str):
                raise ReproducibilityError("replay identity must be a string")
            prompt_digest = hashlib.sha256(replay_identity.encode("utf-8")).hexdigest()
            slot = self._reserve()
            try:
                response = backend(prompt)
            except BaseException as exc:
                self._complete(
                    slot,
                    ReplayRecord(
                        prompt=replay_identity,
                        prompt_sha256=prompt_digest,
                        error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                        error_message=str(exc),
                    ),
                )
                raise
            if not isinstance(response, str):
                error = ReproducibilityError("recorded model response must be a string")
                self._complete(
                    slot,
                    ReplayRecord(
                        prompt=replay_identity,
                        prompt_sha256=prompt_digest,
                        error_type=f"{type(error).__module__}.{type(error).__qualname__}",
                        error_message=str(error),
                    ),
                )
                raise error
            self._complete(
                slot,
                ReplayRecord(
                    prompt=replay_identity,
                    prompt_sha256=prompt_digest,
                    response=response,
                ),
            )
            return response

        return _record

    def attach_recording(self, agents: Iterable["AIAgent"]) -> None:
        """Install recording wrappers after all model boundaries validate."""

        agent_list = list(agents)
        identities = [agent.replay_identity_resolver() for agent in agent_list]
        for agent in agent_list:
            agent.replayable_backend()
        for agent, identity in zip(agent_list, identities):
            agent.set_model_backend_wrapper(
                lambda backend, identity=identity: self.recording_backend(
                    backend,
                    identity=identity,
                )
            )

    def attach_replay(
        self,
        agents: Iterable["AIAgent"],
        *,
        manifest: Optional[RunManifest] = None,
    ) -> "ReplayBackend":
        """Install one shared replay cursor with per-agent request identities."""

        agent_list = list(agents)
        identities = [agent.replay_identity_resolver() for agent in agent_list]
        backend = self.replay_backend(manifest=manifest)
        for agent, identity in zip(agent_list, identities):
            agent.set_model_backend_wrapper(None)
            agent.set_llm_backend(backend.bind_identity(identity))
        return backend

    def replay_backend(
        self,
        *,
        manifest: Optional[RunManifest] = None,
    ) -> "ReplayBackend":
        if (
            manifest is not None
            and self.manifest_fingerprint is not None
            and manifest.fingerprint() != self.manifest_fingerprint
        ):
            suffix = ""
            if self.manifest_compatibility is not None:
                differences = _manifest_diff_paths(
                    self.manifest_compatibility,
                    manifest.compatibility_payload(),
                )
                if differences:
                    suffix = ": differing fields: " + ", ".join(differences[:12])
            raise ReplayMismatchError("run manifest does not match replay tape" + suffix)
        return ReplayBackend(self)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            if any(record is None for record in self._records):
                raise ReproducibilityError("cannot persist replay tape with in-flight recordings")
            return {
                "version": _REPLAY_VERSION,
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_compatibility": deepcopy(self.manifest_compatibility),
                "records": [
                    record.to_dict()
                    for record in self._records
                    if record is not None
                ],
            }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayTape":
        version = payload.get("version", _REPLAY_VERSION)
        if version != _REPLAY_VERSION:
            raise ReproducibilityError(f"unsupported replay tape version: {version}")
        records_payload = payload.get("records", [])
        if not isinstance(records_payload, list):
            raise ReproducibilityError("replay tape records must be a list")
        if not all(isinstance(item, Mapping) for item in records_payload):
            raise ReproducibilityError("replay tape records must contain JSON objects")
        records = [ReplayRecord.from_dict(item) for item in records_payload]
        fingerprint = payload.get("manifest_fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ReproducibilityError("manifest_fingerprint must be a string or null")
        compatibility = payload.get("manifest_compatibility")
        if compatibility is not None and not isinstance(compatibility, Mapping):
            raise ReproducibilityError("manifest_compatibility must be a JSON object or null")
        return cls(
            records,
            manifest_fingerprint=fingerprint,
            manifest_compatibility=compatibility,
        )

    def save(self, path: Path | str) -> None:
        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> "ReplayTape":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReproducibilityError("replay tape root must be a JSON object")
        return cls.from_dict(payload)


class ReplayBackend:
    """Sequential backend that validates exact replay-identity order and content."""

    def __init__(self, tape: ReplayTape) -> None:
        self._tape = tape
        self._index = 0
        self._lock = threading.Lock()

    @property
    def position(self) -> int:
        with self._lock:
            return self._index

    def _consume(self, replay_identity: str) -> str:
        if not isinstance(replay_identity, str):
            raise ReplayMismatchError("replay identity must be a string")
        with self._lock:
            records = self._tape.records
            if self._index >= len(records):
                raise ReplayMismatchError(
                    f"replay exhausted at call {self._index}; no recorded response remains"
                )
            record = records[self._index]
            digest = hashlib.sha256(replay_identity.encode("utf-8")).hexdigest()
            if digest != record.prompt_sha256 or replay_identity != record.prompt:
                raise ReplayMismatchError(
                    f"replay prompt mismatch at call {self._index}: "
                    f"expected sha256={record.prompt_sha256}, got sha256={digest}"
                )
            self._index += 1
            if record.error_type is not None:
                raise RecordedReplayError(
                    f"recorded model failure {record.error_type}: {record.error_message or ''}"
                )
            if record.response is None:
                raise ReproducibilityError("replay record has neither response nor error")
            return record.response

    def __call__(self, prompt: str) -> str:
        return self._consume(prompt)

    def bind_identity(
        self,
        identity: Callable[[str], str],
    ) -> Callable[[str], str]:
        if not callable(identity):
            raise ReproducibilityError("replay identity must be callable")

        def _bound(prompt: str) -> str:
            replay_identity = identity(prompt)
            return self._consume(replay_identity)

        return _bound

    def assert_consumed(self) -> None:
        with self._lock:
            total = len(self._tape.records)
            if self._index != total:
                raise ReplayMismatchError(
                    f"replay consumed {self._index} of {total} recorded model calls"
                )

    def reset(self) -> None:
        with self._lock:
            self._index = 0


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temp_name: Optional[str] = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "ReplayBackend",
    "ReplayRecord",
    "ReplayTape",
    "RunManifest",
    "SeedReport",
    "create_run_manifest",
    "prepare_reproducible_run",
    "seed_everything",
]
