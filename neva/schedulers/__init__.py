"""Schedulers controlling agent execution order."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Type, TypeVar, cast

from neva.utils.exceptions import ConfigurationError

from .base import Scheduler
from .composite import CompositeScheduler
from .conditional import ConditionalScheduler
from .event_driven import EventDrivenScheduler
from .least_recently_used import LeastRecentlyUsedScheduler
from .priority import PriorityScheduler
from .random import RandomScheduler
from .round_robin import RoundRobinScheduler
from .weighted_random import WeightedRandomScheduler

SchedulerType = TypeVar("SchedulerType", bound=Scheduler)

_REGISTRY: Dict[str, Type[Scheduler]] = {}
_ALIASES: Dict[str, str] = {}


def _canonical_name(name: str) -> str:
    if not name or not name.strip():
        raise ConfigurationError("Scheduler name cannot be empty.")
    return name.strip().lower()


def _resolve_to_canonical(name: str) -> str:
    key = name.strip()
    if not key:
        raise ConfigurationError("Scheduler name cannot be empty.")

    canonical = key.lower()
    if canonical in _REGISTRY:
        return canonical

    alias_target = _ALIASES.get(key) or _ALIASES.get(canonical)
    if alias_target is not None and alias_target in _REGISTRY:
        return alias_target

    available = ", ".join(sorted(_REGISTRY)) or "<none>"
    raise ConfigurationError(f"Unknown scheduler '{name}'. Available schedulers: {available}.")


def register_scheduler(
    name: str,
    scheduler_cls: Type[SchedulerType],
    *,
    overwrite: bool = False,
    aliases: Optional[Iterable[str]] = None,
) -> None:
    """Register ``scheduler_cls`` under ``name`` and optional aliases."""

    if not issubclass(scheduler_cls, Scheduler):
        raise ConfigurationError("scheduler_cls must inherit from Scheduler")

    canonical = _canonical_name(name)
    existing = _REGISTRY.get(canonical)
    if existing is not None and existing is not scheduler_cls and not overwrite:
        raise ConfigurationError(
            f"Scheduler '{canonical}' already registered with {existing.__name__}."
        )

    _REGISTRY[canonical] = scheduler_cls

    for alias in {name, scheduler_cls.__name__} | set(aliases or []):
        alias_key = alias.strip()
        if not alias_key:
            continue
        _ALIASES[alias_key] = canonical
        _ALIASES[alias_key.lower()] = canonical


def unregister_scheduler(name: str) -> None:
    """Remove ``name`` and its aliases from the registry."""

    canonical = _resolve_to_canonical(name)
    _REGISTRY.pop(canonical, None)
    for alias, target in list(_ALIASES.items()):
        if target == canonical:
            _ALIASES.pop(alias)


def get_scheduler_class(name: str) -> Type[SchedulerType]:
    """Return the registered scheduler class for ``name``."""

    canonical = _resolve_to_canonical(name)
    return cast(Type[SchedulerType], _REGISTRY[canonical])


def create_scheduler(name: str, **kwargs: object) -> Scheduler:
    """Instantiate the scheduler associated with ``name``."""

    scheduler_cls = cast(Type[Scheduler], get_scheduler_class(name))
    scheduler = scheduler_cls(**kwargs)
    return cast(Scheduler, scheduler)


def available_schedulers() -> List[str]:
    """Return the sorted list of registered scheduler names."""

    return sorted(_REGISTRY)


# Register built-in schedulers.
register_scheduler("round_robin", RoundRobinScheduler)
register_scheduler("random", RandomScheduler)
register_scheduler("priority", PriorityScheduler)
register_scheduler("least_recently_used", LeastRecentlyUsedScheduler)
register_scheduler("weighted_random", WeightedRandomScheduler)
register_scheduler("composite", CompositeScheduler)
register_scheduler("event_driven", EventDrivenScheduler)
register_scheduler("conditional", ConditionalScheduler)


__all__ = [
    "CompositeScheduler",
    "ConditionalScheduler",
    "EventDrivenScheduler",
    "LeastRecentlyUsedScheduler",
    "PriorityScheduler",
    "RandomScheduler",
    "RoundRobinScheduler",
    "Scheduler",
    "WeightedRandomScheduler",
    "available_schedulers",
    "create_scheduler",
    "get_scheduler_class",
    "register_scheduler",
    "unregister_scheduler",
]
