"""OpenTelemetry-based observability primitives for Neva."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from typing_extensions import Literal

from neva.utils.exceptions import MissingDependencyError

try:  # pragma: no cover - imported for typing only when available.
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Span, Tracer
except Exception:  # pragma: no cover - optional dependency.
    Meter = Any  # type: ignore[misc, assignment]
    Span = Any  # type: ignore[misc, assignment]
    Tracer = Any  # type: ignore[misc, assignment]

if TYPE_CHECKING:  # pragma: no cover - import for type checking only.
    from neva.utils.state_management import ConversationState


logger = logging.getLogger(__name__)


def _require_opentelemetry() -> "_OpenTelemetryModules":
    """Import OpenTelemetry modules lazily to keep the dependency optional."""

    try:
        otel_logs = importlib.import_module("opentelemetry.logs")
        otel_metrics = importlib.import_module("opentelemetry.metrics")
        otel_trace = importlib.import_module("opentelemetry.trace")
        context_module = importlib.import_module("opentelemetry.context")
        Context = getattr(context_module, "Context")
        set_span_in_context = getattr(context_module, "set_span_in_context")
        logs_module = importlib.import_module("opentelemetry.sdk._logs")
        LoggerProvider = getattr(logs_module, "LoggerProvider")
        logs_export_module = importlib.import_module("opentelemetry.sdk._logs.export")
        BatchLogRecordProcessor = getattr(logs_export_module, "BatchLogRecordProcessor")
        LogExporter = getattr(logs_export_module, "LogExporter")
        logging_module = importlib.import_module("opentelemetry.sdk._logs.logging")
        LoggingHandler = getattr(logging_module, "LoggingHandler")
        metrics_module = importlib.import_module("opentelemetry.sdk.metrics")
        MeterProvider = getattr(metrics_module, "MeterProvider")
        metrics_export_module = importlib.import_module("opentelemetry.sdk.metrics.export")
        MetricReader = getattr(metrics_export_module, "MetricReader")
        resources_module = importlib.import_module("opentelemetry.sdk.resources")
        OTEL_SERVICE_NAME = getattr(resources_module, "SERVICE_NAME")
        Resource = getattr(resources_module, "Resource")
        trace_module = importlib.import_module("opentelemetry.sdk.trace")
        TracerProvider = getattr(trace_module, "TracerProvider")
        trace_export_module = importlib.import_module("opentelemetry.sdk.trace.export")
        BatchSpanProcessor = getattr(trace_export_module, "BatchSpanProcessor")
        SpanExporter = getattr(trace_export_module, "SpanExporter")
        SpanKind = getattr(otel_trace, "SpanKind")
    except Exception as exc:  # pragma: no cover - dependency missing at runtime.
        raise MissingDependencyError(
            "OpenTelemetry is required for telemetry support. Install the 'observability' extra "
            'via `pip install neva[observability]` or `poetry install --extras "observability"`.'
        ) from exc

    return _OpenTelemetryModules(
        trace=otel_trace,
        metrics=otel_metrics,
        logs=otel_logs,
        context_cls=Context,
        set_span_in_context=set_span_in_context,
        tracer_provider_cls=TracerProvider,
        span_exporter_cls=SpanExporter,
        batch_span_processor_cls=BatchSpanProcessor,
        meter_provider_cls=MeterProvider,
        metric_reader_cls=MetricReader,
        resource_cls=Resource,
        service_name_const=OTEL_SERVICE_NAME,
        logger_provider_cls=LoggerProvider,
        log_exporter_cls=LogExporter,
        batch_log_processor_cls=BatchLogRecordProcessor,
        logging_handler_cls=LoggingHandler,
        span_kind=SpanKind,
    )


@dataclass
class _OpenTelemetryModules:
    trace: Any
    metrics: Any
    logs: Any
    context_cls: Any
    set_span_in_context: Any
    tracer_provider_cls: Any
    span_exporter_cls: Any
    batch_span_processor_cls: Any
    meter_provider_cls: Any
    metric_reader_cls: Any
    resource_cls: Any
    service_name_const: str
    logger_provider_cls: Any
    log_exporter_cls: Any
    batch_log_processor_cls: Any
    logging_handler_cls: Any
    span_kind: Any


@dataclass
class _NoOpHistogram:
    """Fallback histogram used when metrics are disabled."""

    def record(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - trivial.
        return


@dataclass
class _NoOpCounter:
    """Fallback counter used when metrics are disabled."""

    def add(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - trivial.
        return


@dataclass
class _FallbackSpan:
    """Span implementation used when OpenTelemetry is unavailable."""

    attributes: Dict[str, Any]
    events: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    ended: bool = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def end(self) -> None:
        self.ended = True


class _FallbackSpanContext:
    def __init__(self, span: _FallbackSpan):
        self._span = span

    def __enter__(self) -> _FallbackSpan:
        return self._span

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> Literal[False]:
        self._span.end()
        return False


class _FallbackTracer:
    def __init__(self) -> None:
        self.started: List[Tuple[str, _FallbackSpan]] = []

    def start_span(
        self, name: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> _FallbackSpan:
        span = _FallbackSpan(dict(attributes or {}))
        self.started.append((name, span))
        return span

    def start_as_current_span(
        self,
        name: str,
        context: Optional[Any] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        kind: Optional[Any] = None,
    ) -> _FallbackSpanContext:
        span = self.start_span(name, attributes=attributes)
        span.attributes["span.kind"] = kind
        span.attributes["span.context"] = context
        return _FallbackSpanContext(span)


class _FallbackHistogram:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records: List[Tuple[Any, Dict[str, Any]]] = []

    def record(self, value: Any, attributes: Optional[Mapping[str, Any]] = None) -> None:
        self.records.append((value, dict(attributes or {})))


class _FallbackCounter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records: List[Tuple[Any, Dict[str, Any]]] = []

    def add(self, value: Any, attributes: Optional[Mapping[str, Any]] = None) -> None:
        self.records.append((value, dict(attributes or {})))


class _FallbackMeter:
    def __init__(self) -> None:
        self.histograms: Dict[str, _FallbackHistogram] = {}
        self.counters: Dict[str, _FallbackCounter] = {}

    def create_histogram(self, name: str, **_kwargs: Any) -> _FallbackHistogram:
        instrument = _FallbackHistogram(name)
        self.histograms[name] = instrument
        return instrument

    def create_counter(self, name: str, **_kwargs: Any) -> _FallbackCounter:
        instrument = _FallbackCounter(name)
        self.counters[name] = instrument
        return instrument


class _FallbackStructuredLogger:
    def __init__(self) -> None:
        self.records: List[Tuple[str, Dict[str, Any]]] = []

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        rendered = msg % args if args else msg
        payload = dict(kwargs.get("extra", {}))
        self.records.append((rendered, payload))

    def debug(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - compatibility shim.
        return


def _estimate_tokens(text: Optional[str]) -> int:
    """Rough token estimation that works without backend specific tooling."""

    if not text:
        return 0
    tokens = len(text.split())
    return tokens if tokens > 0 else 0


def _normalise_attributes(attributes: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if not attributes:
        return payload
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[key] = value
        else:
            try:
                payload[key] = json.dumps(value)
            except TypeError:
                payload[key] = str(value)
    return payload


def _extract_reasoning_steps(response: Optional[str]) -> List[str]:
    if not response:
        return []
    steps: List[str] = []
    for line in response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith(("thought", "reasoning", "step", "analysis")):
            steps.append(stripped)
    return steps


@dataclass
class _TelemetryInstruments:
    agent_latency: Any = field(default_factory=_NoOpHistogram)
    llm_latency: Any = field(default_factory=_NoOpHistogram)
    tool_latency: Any = field(default_factory=_NoOpHistogram)
    prompt_tokens: Any = field(default_factory=_NoOpCounter)
    completion_tokens: Any = field(default_factory=_NoOpCounter)
    total_tokens: Any = field(default_factory=_NoOpCounter)
    context_tokens: Any = field(default_factory=_NoOpHistogram)
    tool_invocations: Any = field(default_factory=_NoOpCounter)


class TelemetryManager:
    """Coordinate OpenTelemetry traces, metrics, and logs for Neva."""

    def __init__(
        self,
        *,
        service_name: str = "neva",
        resource_attributes: Optional[Mapping[str, Any]] = None,
        tracer_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
        logger_provider: Optional[Any] = None,
        span_exporter: Optional[Any] = None,
        metric_readers: Optional[Iterable[Any]] = None,
        log_exporter: Optional[Any] = None,
        tracer: Optional[Tracer] = None,
        meter: Optional[Meter] = None,
        structured_logger: Optional[logging.Logger] = None,
    ) -> None:
        manual_instrumentation = (
            tracer is not None
            and meter is not None
            and structured_logger is not None
            and tracer_provider is None
            and meter_provider is None
            and logger_provider is None
            and span_exporter is None
            and log_exporter is None
            and not metric_readers
        )

        modules: Optional[_OpenTelemetryModules]
        if manual_instrumentation:
            modules = None
        else:
            try:
                modules = _require_opentelemetry()
            except MissingDependencyError:
                modules = None

        self._conversation_spans: Dict[str, Union[Span, _FallbackSpan]] = {}
        self._owns_tracer_provider = False
        self._owns_meter_provider = False
        self._owns_logger_provider = False
        self._trace_api: Any = None
        self._metrics_api: Any = None
        self._set_span_in_context: Callable[[Any], Any]
        self._span_kind: Any
        self._tracer_provider: Optional[Any] = tracer_provider
        self._meter_provider: Optional[Any] = meter_provider
        self._logger_provider: Optional[Any] = logger_provider
        self._tracer: Any
        self._meter: Any
        self._structured_logger: Optional[Union[logging.Logger, _FallbackStructuredLogger]]
        self._logging_handler: Optional[logging.Handler]

        if modules is None:

            class _SpanKindShim:
                INTERNAL = "internal"
                CLIENT = "client"

            self._trace_api = None
            self._metrics_api = None
            self._set_span_in_context = lambda span: span
            self._span_kind = _SpanKindShim()
            self._tracer_provider = None
            self._meter_provider = None
            self._logger_provider = None
            self._tracer = tracer or _FallbackTracer()
            self._meter = meter or _FallbackMeter()
            self._structured_logger = structured_logger or _FallbackStructuredLogger()
            self._logging_handler = None
        else:
            resource_attributes = dict(resource_attributes or {})
            if modules.service_name_const not in resource_attributes:
                resource_attributes[modules.service_name_const] = service_name
            resource = modules.resource_cls.create(resource_attributes)

            self._trace_api = modules.trace
            self._metrics_api = modules.metrics
            self._set_span_in_context = modules.set_span_in_context
            self._span_kind = modules.span_kind

            if tracer_provider is None and tracer is None:
                tracer_provider = modules.tracer_provider_cls(resource=resource)
                if span_exporter is not None:
                    processor = modules.batch_span_processor_cls(span_exporter)
                    tracer_provider.add_span_processor(processor)
                modules.trace.set_tracer_provider(tracer_provider)
                self._owns_tracer_provider = True
            self._tracer_provider = tracer_provider
            self._tracer = tracer or modules.trace.get_tracer(__name__)

            readers = list(metric_readers or [])
            if meter_provider is None and meter is None:
                meter_provider = modules.meter_provider_cls(
                    resource=resource, metric_readers=readers
                )
                modules.metrics.set_meter_provider(meter_provider)
                self._owns_meter_provider = True
            self._meter_provider = meter_provider
            self._meter = meter or modules.metrics.get_meter(__name__)

            if logger_provider is None and structured_logger is None:
                logger_provider = modules.logger_provider_cls(resource=resource)
                if log_exporter is not None:
                    processor = modules.batch_log_processor_cls(log_exporter)
                    logger_provider.add_log_record_processor(processor)
                modules.logs.set_logger_provider(logger_provider)
                self._owns_logger_provider = True
            self._logger_provider = logger_provider

            if structured_logger is None and logger_provider is not None:
                handler = modules.logging_handler_cls(
                    level=logging.NOTSET, logger_provider=logger_provider
                )
                telemetry_logger = logging.getLogger("neva.telemetry")
                telemetry_logger.setLevel(logging.INFO)
                telemetry_logger.propagate = False
                # Remove any stale telemetry handlers to avoid duplicate emission.
                telemetry_logger.handlers = [
                    existing
                    for existing in telemetry_logger.handlers
                    if not isinstance(existing, modules.logging_handler_cls)
                ]
                telemetry_logger.addHandler(handler)
                self._logging_handler = handler
                self._structured_logger = telemetry_logger
            else:
                self._logging_handler = None
                self._structured_logger = structured_logger

        self._log_payload_direct = modules is None or structured_logger is not None
        self._instruments = self._create_instruments()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _create_instruments(self) -> _TelemetryInstruments:
        instruments = _TelemetryInstruments()
        try:
            instruments.agent_latency = self._meter.create_histogram(
                name="neva.agent.response.latency",
                unit="s",
                description="Latency for each agent turn.",
            )
            instruments.llm_latency = self._meter.create_histogram(
                name="neva.llm.api.latency",
                unit="s",
                description="Latency of direct LLM API calls.",
            )
            instruments.tool_latency = self._meter.create_histogram(
                name="neva.tool.invocation.latency",
                unit="s",
                description="Runtime of tool invocations triggered by agents.",
            )
            instruments.prompt_tokens = self._meter.create_counter(
                name="neva.llm.prompt.tokens",
                unit="{token}",
                description="Estimated prompt tokens consumed per turn.",
            )
            instruments.completion_tokens = self._meter.create_counter(
                name="neva.llm.completion.tokens",
                unit="{token}",
                description="Estimated completion tokens produced per turn.",
            )
            instruments.total_tokens = self._meter.create_counter(
                name="neva.llm.total.tokens",
                unit="{token}",
                description="Total estimated tokens exchanged with language models.",
            )
            instruments.context_tokens = self._meter.create_histogram(
                name="neva.context.window.tokens",
                unit="{token}",
                description="Tokens tracked in the active conversation context.",
            )
            instruments.tool_invocations = self._meter.create_counter(
                name="neva.tool.invocations",
                unit="{count}",
                description="Number of tool invocations triggered by agents.",
            )
        except Exception:  # pragma: no cover - metric provider may be a noop.
            logger.debug("Failed to create one or more telemetry instruments", exc_info=True)
        return instruments

    def _ensure_conversation_span(
        self, conversation_id: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> Union[Span, _FallbackSpan]:
        span = self._conversation_spans.get(conversation_id)
        if span is not None:
            if attributes:
                for key, value in _normalise_attributes(attributes).items():
                    span.set_attribute(key, value)
            return span
        payload = {"conversation.id": conversation_id}
        payload.update(_normalise_attributes(attributes))
        span = self._tracer.start_span("neva.conversation", attributes=payload)
        self._conversation_spans[conversation_id] = span
        return span

    def _conversation_context(
        self, conversation_id: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> Any:
        span = self._ensure_conversation_span(conversation_id, attributes=attributes)
        return self._set_span_in_context(span)

    def _log_event(self, event: str, attributes: Mapping[str, Any]) -> None:
        if self._structured_logger is None:
            return
        payload = {key: value for key, value in attributes.items() if value is not None}
        try:
            if self._log_payload_direct:
                self._structured_logger.info("%s", event, extra=payload)
            else:
                self._structured_logger.info("%s", event, extra={"otel_attributes": payload})
        except Exception:  # pragma: no cover - defensive logging guard.
            logger.debug("Failed to emit telemetry log", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_agent_registration(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = {"agent.name": agent_name}
        payload.update(_normalise_attributes(attributes))
        span = self._ensure_conversation_span(conversation_id, attributes=payload)
        span.add_event("agent.registered", payload)
        self._log_event("agent_registered", {"conversation.id": conversation_id, **payload})

    def record_scheduler_decision(
        self,
        *,
        conversation_id: str,
        scheduler_name: str,
        agent_name: str,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = {
            "conversation.id": conversation_id,
            "scheduler.name": scheduler_name,
            "agent.name": agent_name,
        }
        payload.update(_normalise_attributes(attributes))
        span = self._ensure_conversation_span(conversation_id)
        span.add_event("scheduler.decision", payload)
        self._log_event("scheduler_decision", payload)

    def record_agent_turn(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        prompt: str,
        response: str,
        latency: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        context_tokens: Optional[int] = None,
        model: Optional[str] = None,
        reasoning_steps: Optional[Sequence[str]] = None,
        tool_calls: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        conversation_state: Optional["ConversationState"] = None,
    ) -> None:
        metric_attrs: Dict[str, str] = {
            "conversation.id": conversation_id,
            "agent.name": agent_name,
        }
        if model:
            metric_attrs["llm.model"] = model
        span_attrs: Dict[str, Any] = dict(metric_attrs)
        span_attrs.update(_normalise_attributes(metadata))
        context = self._conversation_context(conversation_id)
        prompt_tokens = prompt_tokens if prompt_tokens is not None else _estimate_tokens(prompt)
        completion_tokens = (
            completion_tokens if completion_tokens is not None else _estimate_tokens(response)
        )
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        if context_tokens is None and conversation_state is not None:
            try:
                context_tokens = sum(
                    _estimate_tokens(turn.message) for turn in conversation_state.turns
                )
            except Exception:  # pragma: no cover - defensive guard.
                context_tokens = None
        if reasoning_steps is None:
            reasoning_steps = _extract_reasoning_steps(response)

        span_kind_internal = getattr(self._span_kind, "INTERNAL", None)
        with self._tracer.start_as_current_span(
            "neva.agent.turn",
            context=context,
            attributes=span_attrs,
            kind=span_kind_internal,
        ) as span:
            if prompt:
                span.add_event("llm.prompt", {"llm.prompt": prompt})
            if response:
                span.add_event("llm.completion", {"llm.completion": response})
            if reasoning_steps:
                for idx, step in enumerate(reasoning_steps, start=1):
                    span.add_event(
                        "agent.reasoning",
                        {"reasoning.index": idx, "reasoning.content": step},
                    )
            if tool_calls:
                for index, call in enumerate(tool_calls, start=1):
                    event_attrs = {
                        "tool.sequence": index,
                        "tool.name": call.get("name"),
                        "tool.response": call.get("response"),
                        "tool.error": call.get("error"),
                    }
                    arguments = call.get("arguments")
                    if arguments is not None:
                        try:
                            event_attrs["tool.arguments"] = json.dumps(arguments)
                        except TypeError:
                            event_attrs["tool.arguments"] = str(arguments)
                    duration = call.get("duration")
                    if duration is not None:
                        event_attrs["tool.duration"] = float(duration)
                    span.add_event("tool.invocation", _normalise_attributes(event_attrs))

        try:
            if latency is not None:
                self._instruments.agent_latency.record(float(latency), attributes=metric_attrs)
            if prompt_tokens:
                self._instruments.prompt_tokens.add(int(prompt_tokens), attributes=metric_attrs)
            if completion_tokens:
                self._instruments.completion_tokens.add(
                    int(completion_tokens), attributes=metric_attrs
                )
            if total_tokens:
                self._instruments.total_tokens.add(int(total_tokens), attributes=metric_attrs)
            if context_tokens:
                self._instruments.context_tokens.record(
                    int(context_tokens), attributes=metric_attrs
                )
        except Exception:  # pragma: no cover - defensive guard around metric emission.
            logger.debug("Failed to record telemetry metrics for agent turn", exc_info=True)

        log_payload: Dict[str, Any] = {
            **metric_attrs,
            "llm.model": model,
            "agent.prompt": prompt,
            "agent.response": response,
            "agent.latency_seconds": latency,
            "agent.prompt_tokens": prompt_tokens,
            "agent.completion_tokens": completion_tokens,
            "agent.total_tokens": total_tokens,
            "agent.context_tokens": context_tokens,
        }
        log_payload.update(_normalise_attributes(metadata))
        self._log_event("agent_turn", log_payload)

    def record_llm_api_call(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        prompt: str,
        completion: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        latency: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        conversation_state: Optional["ConversationState"] = None,
    ) -> None:
        metric_attrs: Dict[str, str] = {
            "conversation.id": conversation_id,
            "agent.name": agent_name,
        }
        if provider:
            metric_attrs["llm.provider"] = provider
        if model:
            metric_attrs["llm.model"] = model
        prompt_tokens = prompt_tokens if prompt_tokens is not None else _estimate_tokens(prompt)
        completion_tokens = (
            completion_tokens if completion_tokens is not None else _estimate_tokens(completion)
        )
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        if conversation_state is not None:
            try:
                context_tokens = sum(
                    _estimate_tokens(turn.message) for turn in conversation_state.turns
                )
            except Exception:  # pragma: no cover - defensive guard.
                context_tokens = None
        else:
            context_tokens = None

        span_attrs: Dict[str, Any] = dict(metric_attrs)
        span_attrs.update(_normalise_attributes(metadata))
        context = self._conversation_context(conversation_id)
        span_kind_client = getattr(self._span_kind, "CLIENT", None)
        with self._tracer.start_as_current_span(
            "neva.llm.call",
            context=context,
            attributes=span_attrs,
            kind=span_kind_client,
        ) as span:
            span.add_event("llm.prompt", {"llm.prompt": prompt})
            span.add_event("llm.completion", {"llm.completion": completion})

        try:
            if latency is not None:
                self._instruments.llm_latency.record(float(latency), attributes=metric_attrs)
            if prompt_tokens:
                self._instruments.prompt_tokens.add(int(prompt_tokens), attributes=metric_attrs)
            if completion_tokens:
                self._instruments.completion_tokens.add(
                    int(completion_tokens), attributes=metric_attrs
                )
            if total_tokens:
                self._instruments.total_tokens.add(int(total_tokens), attributes=metric_attrs)
            if context_tokens:
                self._instruments.context_tokens.record(
                    int(context_tokens), attributes=metric_attrs
                )
        except Exception:  # pragma: no cover - defensive guard.
            logger.debug("Failed to record telemetry metrics for LLM call", exc_info=True)

        log_payload: Dict[str, Any] = {
            **metric_attrs,
            "llm.latency_seconds": latency,
            "llm.prompt_tokens": prompt_tokens,
            "llm.completion_tokens": completion_tokens,
            "llm.total_tokens": total_tokens,
            "llm.context_tokens": context_tokens,
        }
        log_payload.update(_normalise_attributes(metadata))
        self._log_event("llm_call", log_payload)

    def record_tool_call(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        tool_name: str,
        duration: Optional[float] = None,
        arguments: Optional[Any] = None,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        metric_attrs: Dict[str, str] = {
            "conversation.id": conversation_id,
            "agent.name": agent_name,
            "tool.name": tool_name,
        }
        span_attrs: Dict[str, Any] = dict(metric_attrs)
        span_attrs.update(_normalise_attributes(metadata))
        context = self._conversation_context(conversation_id)
        span_kind_client = getattr(self._span_kind, "CLIENT", None)
        with self._tracer.start_as_current_span(
            "neva.tool.call",
            context=context,
            attributes=span_attrs,
            kind=span_kind_client,
        ) as span:
            event_attrs: Dict[str, Any] = {
                "tool.name": tool_name,
                "tool.error": error,
            }
            if arguments is not None:
                try:
                    event_attrs["tool.arguments"] = json.dumps(arguments)
                except TypeError:
                    event_attrs["tool.arguments"] = str(arguments)
            if output is not None:
                event_attrs["tool.response"] = str(output)
            if duration is not None:
                event_attrs["tool.duration"] = float(duration)
            span.add_event("tool.invocation", _normalise_attributes(event_attrs))

        try:
            self._instruments.tool_invocations.add(1, attributes=metric_attrs)
            if duration is not None:
                self._instruments.tool_latency.record(float(duration), attributes=metric_attrs)
        except Exception:  # pragma: no cover - defensive guard.
            logger.debug("Failed to record telemetry metrics for tool call", exc_info=True)

        log_payload: Dict[str, Any] = dict(metric_attrs)
        log_payload.update(
            {
                "tool.duration_seconds": duration,
                "tool.error": error,
                "tool.arguments": json.dumps(arguments) if arguments is not None else None,
            }
        )
        if output is not None:
            log_payload["tool.response"] = str(output)
        log_payload.update(_normalise_attributes(metadata))
        self._log_event("tool_call", log_payload)

    def record_reasoning_step(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        content: str,
        index: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "conversation.id": conversation_id,
            "agent.name": agent_name,
            "reasoning.content": content,
        }
        if index is not None:
            payload["reasoning.index"] = index
        payload.update(_normalise_attributes(metadata))
        span = self._ensure_conversation_span(conversation_id)
        span.add_event("agent.reasoning", payload)
        self._log_event("reasoning_step", payload)

    def end_conversation(self, conversation_id: str) -> None:
        span = self._conversation_spans.pop(conversation_id, None)
        if span is not None:
            try:
                span.end()
            except Exception:  # pragma: no cover - defensive guard.
                logger.debug("Failed to end conversation span", exc_info=True)

    def shutdown(self) -> None:
        for conversation_id in list(self._conversation_spans.keys()):
            self.end_conversation(conversation_id)
        if self._logging_handler is not None:
            if isinstance(self._structured_logger, logging.Logger):
                try:
                    self._structured_logger.removeHandler(self._logging_handler)
                except Exception:  # pragma: no cover - defensive guard.
                    logger.debug("Failed to detach telemetry logging handler", exc_info=True)
            self._logging_handler = None
        if self._owns_tracer_provider and self._tracer_provider is not None:
            try:
                self._tracer_provider.shutdown()
            except Exception:  # pragma: no cover - provider may not expose shutdown.
                logger.debug("Failed to shutdown tracer provider", exc_info=True)
        if self._owns_meter_provider and self._meter_provider is not None:
            try:
                self._meter_provider.shutdown()
            except Exception:  # pragma: no cover - optional API.
                logger.debug("Failed to shutdown meter provider", exc_info=True)
        if self._owns_logger_provider and self._logger_provider is not None:
            try:
                self._logger_provider.shutdown()
            except Exception:  # pragma: no cover - optional API.
                logger.debug("Failed to shutdown logger provider", exc_info=True)


_GLOBAL_TELEMETRY: Optional[TelemetryManager] = None
_GLOBAL_LOCK = Lock()


def configure_telemetry(**kwargs: Any) -> TelemetryManager:
    """Initialise and store a global :class:`TelemetryManager` instance."""

    telemetry = TelemetryManager(**kwargs)
    with _GLOBAL_LOCK:
        global _GLOBAL_TELEMETRY
        if _GLOBAL_TELEMETRY is not None:
            _GLOBAL_TELEMETRY.shutdown()
        _GLOBAL_TELEMETRY = telemetry
    return telemetry


def get_telemetry() -> Optional[TelemetryManager]:
    """Return the globally configured :class:`TelemetryManager`, if available."""

    return _GLOBAL_TELEMETRY


def reset_telemetry() -> None:
    """Dispose of the globally configured telemetry manager."""

    with _GLOBAL_LOCK:
        global _GLOBAL_TELEMETRY
        if _GLOBAL_TELEMETRY is not None:
            _GLOBAL_TELEMETRY.shutdown()
        _GLOBAL_TELEMETRY = None


__all__ = [
    "TelemetryManager",
    "configure_telemetry",
    "get_telemetry",
    "reset_telemetry",
]
