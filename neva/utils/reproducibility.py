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
from datetime import datetime, timezone
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
    if isinstance(value, Mapping):
        return {
            str(key): _json_native(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
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
    elif hasattr(scheduler, "_rng"):
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
) -> SeedReport:
    """Seed Python, optional numerical runtimes, and Neva-owned RNG hooks."""

    seed = _validate_seed(seed)
    random.seed(seed)
    hash_seed = seed % (2**32)
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
            f"set-to-{hash_seed}-for-child-processes; current interpreter hash "
            "randomization was fixed at interpreter startup"
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


def _memory_config(memory: Any) -> Dict[str, Any]:
    if memory is None:
        return {"type": None}

    config: Dict[str, Any] = {"type": _type_name(memory)}
    try:
        from neva.utils.checkpoint import _capture_memory

        captured = _capture_memory(memory)
    except ValueError:
        captured = None
    if captured is not None:
        config["state"] = _without_runtime_timestamps(captured)

    callables = _memory_callable_config(memory)
    if callables:
        config["callables"] = callables

    hook = getattr(memory, "reproducibility_config", None)
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
) -> RunManifest:
    """Seed a run and immediately capture its manifest."""

    report = seed_everything(
        seed,
        environment=environment,
        optional_libraries=optional_libraries,
    )
    return create_run_manifest(
        environment,
        seed=seed,
        prompts=prompts,
        seed_report=report,
        dependencies=dependencies,
        metadata=metadata,
    )


@dataclass(frozen=True)
class ReplayRecord:
    """One model-boundary call recorded for deterministic offline replay."""

    prompt: str
    prompt_sha256: str
    response: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "response": self.response,
            "error_type": self.error_type,
            "error_message": self.error_message,
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
        return cls(
            prompt=prompt,
            prompt_sha256=digest,
            response=response,
            error_type=error_type,
            error_message=error_message,
        )


class ReplayTape:
    """Thread-safe sequence of exact model prompts and recorded outcomes."""

    def __init__(
        self,
        records: Optional[Iterable[ReplayRecord]] = None,
        *,
        manifest_fingerprint: Optional[str] = None,
    ) -> None:
        self._records: List[ReplayRecord] = list(records or ())
        self.manifest_fingerprint = manifest_fingerprint
        self._lock = threading.RLock()

    @classmethod
    def for_manifest(cls, manifest: RunManifest) -> "ReplayTape":
        return cls(manifest_fingerprint=manifest.fingerprint())

    @property
    def records(self) -> Sequence[ReplayRecord]:
        with self._lock:
            return tuple(self._records)

    def _append_success(self, prompt: str, response: str) -> None:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self._records.append(ReplayRecord(prompt=prompt, prompt_sha256=digest, response=response))

    def _append_error(self, prompt: str, exc: Exception) -> None:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self._records.append(
            ReplayRecord(
                prompt=prompt,
                prompt_sha256=digest,
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                error_message=str(exc),
            )
        )

    def recording_backend(self, backend: Callable[[str], str]) -> Callable[[str], str]:
        """Wrap a synchronous backend and record calls in deterministic order."""

        if not callable(backend):
            raise ReproducibilityError("recording backend must be callable")

        def _record(prompt: str) -> str:
            if not isinstance(prompt, str):
                raise ReproducibilityError("recorded model prompt must be a string")
            with self._lock:
                try:
                    response = backend(prompt)
                except Exception as exc:
                    self._append_error(prompt, exc)
                    raise
                if not isinstance(response, str):
                    error = ReproducibilityError("recorded model response must be a string")
                    self._append_error(prompt, error)
                    raise error
                self._append_success(prompt, response)
                return response

        return _record

    def attach_recording(self, agents: Iterable["AIAgent"]) -> None:
        """Install recording wrapper factories after every agent validates."""

        agent_list = list(agents)
        for agent in agent_list:
            agent.replayable_backend()
        for agent in agent_list:
            agent.set_model_backend_wrapper(self.recording_backend)

    def attach_replay(
        self,
        agents: Iterable["AIAgent"],
        *,
        manifest: Optional[RunManifest] = None,
    ) -> "ReplayBackend":
        """Install one shared replay sequence across all supplied agents."""

        backend = self.replay_backend(manifest=manifest)
        for agent in agents:
            agent.set_model_backend_wrapper(None)
            agent.set_llm_backend(backend)
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
            raise ReplayMismatchError("run manifest does not match replay tape")
        return ReplayBackend(self)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": _REPLAY_VERSION,
                "manifest_fingerprint": self.manifest_fingerprint,
                "records": [record.to_dict() for record in self._records],
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
        return cls(records, manifest_fingerprint=fingerprint)

    def save(self, path: Path | str) -> None:
        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> "ReplayTape":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReproducibilityError("replay tape root must be a JSON object")
        return cls.from_dict(payload)


class ReplayBackend:
    """Sequential backend that validates exact prompt order and content."""

    def __init__(self, tape: ReplayTape) -> None:
        self._tape = tape
        self._index = 0
        self._lock = threading.Lock()

    @property
    def position(self) -> int:
        with self._lock:
            return self._index

    def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ReplayMismatchError("replay prompt must be a string")
        with self._lock:
            records = self._tape.records
            if self._index >= len(records):
                raise ReplayMismatchError(
                    f"replay exhausted at call {self._index}; no recorded response remains"
                )
            record = records[self._index]
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if digest != record.prompt_sha256 or prompt != record.prompt:
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
