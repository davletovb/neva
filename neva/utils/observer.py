"""Utilities for collecting, aggregating, and exporting simulation metrics."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from functools import wraps
from time import perf_counter
from types import MethodType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    cast,
)

from neva.utils.exceptions import MissingDependencyError
from neva.utils.telemetry import get_telemetry


class ToolLike(Protocol):
    """Protocol describing the ``use`` method exposed by tools."""

    def use(self, *args: Any, **kwargs: Any) -> Any:
        ...


ContextDict = Dict[str, Any]


logger = logging.getLogger(__name__)


def _safe_len(sequence: Optional[Iterable[Any]]) -> int:
    """Return ``len(sequence)`` while tolerating ``None`` or iterables."""

    if sequence is None:
        return 0
    try:
        return len(sequence)  # type: ignore[arg-type]
    except TypeError:
        return sum(1 for _ in sequence)


class SimulationObserver:
    """Collect and store metrics over a simulation run.

    The observer exposes ready-to-use metrics for common conversational
    analysis tasks while still allowing callers to register their own metrics.
    Each call to :meth:`collect_data` stores a snapshot that can be exported to
    CSV/JSON or logged directly to experiment-tracking dashboards (for example
    MLflow).
    """

    def __init__(
        self,
        *,
        enable_builtin_metrics: bool = True,
        sentiment_positive_words: Optional[Iterable[str]] = None,
        sentiment_negative_words: Optional[Iterable[str]] = None,
    ) -> None:
        # ``metrics`` holds callables used to compute each metric while ``data``
        # stores the collected values.
        self.metrics: Dict[str, Callable[..., Any]] = {}
        self.data: Dict[str, List[Any]] = {}

        # Internal state to support built-in metrics.
        self._turn_count = 0
        self._last_turn_timestamp: Optional[datetime] = None
        self._latest_latency: Optional[float] = None
        self._latencies: List[float] = []
        self._participation: Counter[str] = Counter()
        self._tool_usage: MutableMapping[str, Counter[str]] = defaultdict(Counter)
        self._tool_usage_events: List[Dict[str, Any]] = []
        self._latest_agent_name: Optional[str] = None
        self._sentiment_positive_words = set(
            sentiment_positive_words
            or {
                "achieve",
                "awesome",
                "collaborate",
                "creative",
                "excellent",
                "great",
                "progress",
                "success",
                "wonderful",
            }
        )
        self._sentiment_negative_words = set(
            sentiment_negative_words
            or {
                "angry",
                "bad",
                "conflict",
                "difficult",
                "fail",
                "frustrated",
                "issue",
                "problem",
                "struggle",
            }
        )
        self._sentiment_warning_emitted = False

        if enable_builtin_metrics:
            self._register_builtin_metrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_metric(self, metric_name: str, metric_function: Callable[..., Any]) -> None:
        """Register ``metric_function`` under ``metric_name``.

        The callable can optionally accept a ``context`` keyword argument which
        provides additional information (active agent, latency, tool usage,
        etc.). Callables that do not declare ``context`` remain supported for
        backwards compatibility.
        """

        self.metrics[metric_name] = metric_function
        self.data.setdefault(metric_name, [])

    def collect_data(
        self,
        agents: Iterable[Any],
        environment: Optional[Any] = None,
        *,
        active_agent: Optional[Any] = None,
    ) -> None:
        """Collect the current value for each registered metric."""

        agent_list: List[Any] = list(agents)
        now = datetime.utcnow()
        latency: Optional[float] = None
        if self._last_turn_timestamp is not None:
            latency_delta = now - self._last_turn_timestamp
            latency = latency_delta.total_seconds()
            self._latest_latency = latency
            self._latencies.append(latency)
        self._last_turn_timestamp = now

        if active_agent is not None:
            self._turn_count += 1
            agent_name = getattr(active_agent, "name", str(active_agent))
            self._latest_agent_name = agent_name
            self._participation[agent_name] += 1
        else:
            self._latest_agent_name = None

        context: ContextDict = {
            "active_agent": active_agent,
            "active_agent_name": self._latest_agent_name,
            "agents": list(agent_list),
            "environment": environment,
            "latency": latency,
            "latencies": list(self._latencies),
            "turn_count": self._turn_count,
            "participation": dict(self._participation),
            "tool_usage": {agent: dict(counter) for agent, counter in self._tool_usage.items()},
            "tool_usage_events": list(self._tool_usage_events),
            "timestamp": now,
        }

        for metric_name, metric_function in self.metrics.items():
            try:
                value = metric_function(agent_list, environment, context=context)
            except TypeError:
                value = metric_function(agent_list, environment)
            self.data[metric_name].append(value)

    def export_to_csv(self, filename: str) -> None:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for key, value in self.data.items():
                writer.writerow([key] + value)

    def export_to_json(self, filename: str) -> None:
        with open(filename, "w") as jsonfile:
            json.dump(self.data, jsonfile, default=self._json_default)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Return the collected metric history as a serialisable dictionary."""

        return json.loads(json.dumps(self.data, default=self._json_default))

    def latest_snapshot(self) -> Dict[str, Any]:
        """Return the most recent value for each tracked metric."""

        return {name: values[-1] if values else None for name, values in self.data.items()}

    def log_to_mlflow(
        self,
        *,
        step: Optional[int] = None,
        prefix: str = "neva",
    ) -> None:
        """Log the most recent metrics to an active MLflow run.

        Parameters
        ----------
        step:
            Optional step value to associate with the logged metrics.
        prefix:
            Namespace prefix applied to metric names to avoid collisions.
        """

        try:
            import mlflow  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise MissingDependencyError(
                "MLflow is not installed. Install `mlflow` or disable dashboard "
                "logging to continue."
            ) from exc

        snapshot = self.latest_snapshot()
        for metric_name, value in snapshot.items():
            key = f"{prefix}/{metric_name}"
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=step)
            else:
                mlflow.log_param(key, json.dumps(value, default=self._json_default))

    # ------------------------------------------------------------------
    # Observation helpers used by the schedulers/agents
    # ------------------------------------------------------------------
    def watch_agent(self, agent: Any) -> None:
        """Initialise counters and tool instrumentation for ``agent``."""

        agent_name = getattr(agent, "name", str(agent))
        self._participation.setdefault(agent_name, 0)
        self._tool_usage.setdefault(agent_name, Counter())
        for tool in getattr(agent, "tools", []):
            self.watch_tool(agent, tool)

    def watch_tool(self, agent: Any, tool: ToolLike) -> None:
        """Wrap ``tool.use`` to record usage statistics when available."""

        if tool is None:
            return

        sentinel = f"__neva_observer_wrapped_{id(self)}__"
        if getattr(tool, sentinel, False):
            return

        original_use = getattr(tool, "use", None)
        if not callable(original_use):
            return
        wrapped_use: Callable[..., Any] = cast(Callable[..., Any], original_use)

        @wraps(wrapped_use)
        def instrumented_use(self_tool: ToolLike, *args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            try:
                return wrapped_use(*args, **kwargs)
            finally:
                duration = perf_counter() - start
                self.record_tool_usage(agent, self_tool, duration=duration)

        tool.use = MethodType(instrumented_use, tool)  # type: ignore[assignment]
        setattr(tool, sentinel, True)

    def record_tool_usage(
        self, agent: Any, tool: ToolLike, *, duration: Optional[float] = None
    ) -> None:
        """Record a tool invocation emitted by the instrumentation wrapper."""

        agent_name = getattr(agent, "name", str(agent))
        tool_name = getattr(tool, "name", getattr(tool, "__class__", type(tool)).__name__)
        self._tool_usage[agent_name][tool_name] += 1
        event: Dict[str, Any] = {
            "agent": agent_name,
            "tool": tool_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if duration is not None:
            event["duration_seconds"] = duration
        self._tool_usage_events.append(event)
        telemetry = get_telemetry()
        if telemetry is not None:
            try:
                conversation_id = getattr(
                    getattr(agent, "environment", None), "conversation_id", agent_name
                )
                telemetry.record_tool_call(
                    conversation_id=conversation_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    duration=duration,
                    metadata={"sequence": len(self._tool_usage_events)},
                )
            except Exception:  # pragma: no cover - telemetry must remain optional.
                logger.debug("Failed to emit telemetry for tool call", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _json_default(self, value: Any) -> Any:
        if isinstance(value, (datetime,)):
            return value.isoformat()
        if isinstance(value, Counter):
            return dict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return value

    def _register_builtin_metrics(self) -> None:
        def turn_count_metric(
            _agents: Iterable[Any],
            _env: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> int:
            if context is None:
                return self._turn_count
            value = context.get("turn_count")
            return int(value) if isinstance(value, int) else self._turn_count

        self.add_metric("turn_count", turn_count_metric)

        def participation_metric(
            _agents: Iterable[Any],
            _env: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> Dict[str, int]:
            if context is None:
                return dict(self._participation)
            participation = context.get("participation", {})
            return {key: int(value) for key, value in participation.items()}

        self.add_metric("per_agent_participation", participation_metric)

        def active_agent_metric(
            _agents: Iterable[Any],
            _env: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> Optional[str]:
            if context is None:
                return self._latest_agent_name
            value = context.get("active_agent_name")
            return str(value) if value is not None else None

        self.add_metric("active_agent", active_agent_metric)

        def dialogue_length_metric(
            _agents: Iterable[Any],
            environment: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> int:
            transcript = self._extract_transcript(environment)
            return _safe_len(transcript)

        self.add_metric("dialogue_length", dialogue_length_metric)

        def latency_metric(
            _agents: Iterable[Any],
            _env: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> Optional[float]:
            if context is None:
                return self._latest_latency
            value = context.get("latency")
            return float(value) if isinstance(value, (int, float)) else None

        self.add_metric("latest_response_latency_seconds", latency_metric)

        def average_latency_metric(
            _agents: Iterable[Any],
            _env: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> Optional[float]:
            latencies: Iterable[float]
            if context is None:
                latencies = self._latencies
            else:
                latencies = context.get("latencies", [])
            latencies_list = list(latencies)
            if not latencies_list:
                return None
            return sum(latencies_list) / len(latencies_list)

        self.add_metric("average_response_latency_seconds", average_latency_metric)

        def tool_usage_metric(
            _agents: Iterable[Any],
            _env: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> Dict[str, Dict[str, int]]:
            if context is None:
                return {agent: dict(counter) for agent, counter in self._tool_usage.items()}
            usage = context.get("tool_usage", {})
            return {
                agent: {tool: int(count) for tool, count in counters.items()}
                for agent, counters in usage.items()
            }

        self.add_metric("tool_usage_counts", tool_usage_metric)

        def sentiment_metric(
            _agents: Iterable[Any],
            environment: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> Dict[str, Any]:
            message = self._extract_latest_message(environment)
            if not message:
                return {"score": 0.0, "label": "neutral"}
            score = self._compute_sentiment(message)
            label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            return {"score": score, "label": label}

        self.add_metric("recent_sentiment", sentiment_metric)

        def intent_metric(
            _agents: Iterable[Any],
            environment: Optional[Any],
            *,
            context: Optional[ContextDict] = None,
        ) -> str:
            message = self._extract_latest_message(environment)
            if not message:
                return "unknown"
            stripped = message.strip()
            if stripped.endswith("?"):
                return "question"
            lowered = stripped.lower()
            if lowered.startswith("let's") or "let us" in lowered or lowered.startswith("please"):
                return "collaboration"
            if any(
                lowered.startswith(prefix) for prefix in ("do ", "please", "consider", "review")
            ):
                return "request"
            if any(lowered.startswith(prefix) for prefix in ("plan", "we should", "let's")):
                return "planning"
            return "statement"

        self.add_metric("recent_intent", intent_metric)

    def _extract_transcript(self, environment: Optional[Any]) -> List[str]:
        if environment is None:
            return []
        transcript = getattr(environment, "transcript", None)
        if transcript is None and hasattr(environment, "state"):
            transcript = environment.state.get("transcript")  # type: ignore[assignment]
        if transcript is None:
            return []
        if isinstance(transcript, list):
            return transcript
        try:
            return list(transcript)
        except TypeError:
            logger.debug("Transcript attribute %r is not iterable", transcript)
            return []

    def _extract_latest_message(self, environment: Optional[Any]) -> Optional[str]:
        transcript = self._extract_transcript(environment)
        if not transcript:
            return None
        latest = transcript[-1]
        return str(latest) if latest is not None else None

    def _compute_sentiment(self, message: str) -> float:
        words = [token.strip(".,!?") for token in message.lower().split()]
        positive = sum(word in self._sentiment_positive_words for word in words)
        negative = sum(word in self._sentiment_negative_words for word in words)
        total = positive + negative
        if total == 0:
            if not self._sentiment_warning_emitted:
                logger.debug(
                    "Sentiment word lists produced a neutral score for message: %s", message
                )
                self._sentiment_warning_emitted = True
            return 0.0
        return (positive - negative) / total
