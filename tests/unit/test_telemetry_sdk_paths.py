"""Real OpenTelemetry wiring and the no-dependency fallback for telemetry.

``test_telemetry.py`` injects stub tracer/meter/logger objects, which bypasses
both the real SDK construction path and the fallback objects used when
OpenTelemetry is unavailable. These tests exercise the declared
``observability`` extra end to end: loading the pinned SDK release, provider
construction with in-memory exporters, handler attach/detach, shutdown, the
fallback path, and the helper edge cases the stub-based tests never reach.
"""

from __future__ import annotations

import logging

import pytest

from neva.utils.exceptions import MissingDependencyError
from neva.utils.state_management import ConversationState, ConversationTurn
from neva.utils.telemetry import (
    TelemetryManager,
    _estimate_tokens,
    _extract_reasoning_steps,
    _normalise_attributes,
    _require_opentelemetry,
    configure_telemetry,
    get_telemetry,
    reset_telemetry,
)

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")


@pytest.fixture(autouse=True)
def cleanup_global_telemetry():
    reset_telemetry()
    yield
    reset_telemetry()


def _conversation_state() -> ConversationState:
    return ConversationState(
        agent_name="agent-alpha",
        turns=[
            ConversationTurn(speaker="user", message="hello there"),
            ConversationTurn(speaker="agent", message="general kenobi"),
        ],
    )


def test_declared_observability_versions_load_the_real_sdk():
    """The pinned opentelemetry-api/sdk release must load its real modules.

    Older releases expose the logs API as ``opentelemetry._logs`` and
    ``set_span_in_context`` from ``opentelemetry.trace``; loading must not
    silently fall back to the no-dependency instrumentation.
    """

    modules = _require_opentelemetry()

    assert modules.logger_provider_cls.__name__ == "LoggerProvider"
    assert callable(modules.set_span_in_context)
    assert callable(modules.logs.set_logger_provider)
    assert modules.logging_handler_cls is not None


def test_owned_providers_attach_a_logging_handler_and_shut_down():
    telemetry = TelemetryManager(
        service_name="neva-tests",
        resource_attributes={"deployment.environment": "test"},
    )

    assert telemetry._owns_tracer_provider and telemetry._owns_meter_provider
    assert telemetry._owns_logger_provider
    assert telemetry._logging_handler is not None
    assert isinstance(telemetry._structured_logger, logging.Logger)

    telemetry.record_agent_registration(
        conversation_id="conversation-sdk",
        agent_name="agent-alpha",
        attributes={"agent.role": "observer"},
    )
    # A second call reuses the conversation span and applies attributes to it.
    telemetry.record_agent_registration(
        conversation_id="conversation-sdk",
        agent_name="agent-alpha",
        attributes={"agent.role": "analyst"},
    )
    telemetry.record_agent_turn(
        conversation_id="conversation-sdk",
        agent_name="agent-alpha",
        prompt="Hello there",
        response="General Kenobi",
        latency=0.25,
        model="gpt-test",
        metadata={"attempt": 1},
        reasoning_steps=["Thought: check the tracker"],
        tool_calls=[
            {
                "name": "search",
                "response": "neva project",
                "error": None,
                "arguments": {"query": "neva"},
                "duration": 0.05,
            }
        ],
        conversation_state=_conversation_state(),
    )
    telemetry.record_llm_api_call(
        conversation_id="conversation-sdk",
        agent_name="agent-alpha",
        prompt="Hello there",
        completion="General Kenobi",
        provider="openai",
        model="gpt-test",
        latency=0.5,
        first_token_seconds=0.1,
        conversation_state=_conversation_state(),
    )
    telemetry.record_tool_call(
        conversation_id="conversation-sdk",
        agent_name="agent-alpha",
        tool_name="search",
        duration=0.05,
        arguments={"query": "neva"},
        output="neva project",
    )
    telemetry.record_reasoning_step(
        conversation_id="conversation-sdk",
        agent_name="agent-alpha",
        content="Thought: check the tracker",
        index=1,
    )
    telemetry.end_conversation("conversation-sdk")
    assert "conversation-sdk" not in telemetry._conversation_spans

    telemetry.shutdown()
    assert telemetry._logging_handler is None
    telemetry.shutdown()


def test_injected_sdk_components_receive_exported_telemetry():
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    log_exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    telemetry = TelemetryManager(
        tracer=tracer_provider.get_tracer("neva-tests"),
        meter=meter_provider.get_meter("neva-tests"),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )

    telemetry.record_agent_registration(
        conversation_id="conversation-export",
        agent_name="agent-beta",
        attributes={"agent.role": "analyst"},
    )
    telemetry.record_agent_turn(
        conversation_id="conversation-export",
        agent_name="agent-beta",
        prompt="Explain the result",
        response="Done",
        latency=0.3,
        prompt_tokens=4,
        completion_tokens=2,
    )
    telemetry.end_conversation("conversation-export")
    telemetry.shutdown()

    span_names = {span.name for span in span_exporter.get_finished_spans()}
    assert {"neva.conversation", "neva.agent.turn"} <= span_names

    metrics_data = metric_reader.get_metrics_data()
    metric_names = {
        metric.name
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }
    assert {"neva.agent.response.latency", "neva.llm.total.tokens"} <= metric_names

    log_bodies = [
        getattr(record, "log_record", record).body for record in log_exporter.get_finished_logs()
    ]
    assert "agent_registered" in log_bodies
    assert "agent_turn" in log_bodies


def test_fallback_instrumentation_when_opentelemetry_is_unavailable(monkeypatch):
    def missing() -> None:
        raise MissingDependencyError("opentelemetry unavailable")

    monkeypatch.setattr("neva.utils.telemetry._require_opentelemetry", missing)
    telemetry = TelemetryManager(service_name="neva-fallback")

    assert telemetry._trace_api is None and telemetry._metrics_api is None
    assert telemetry._tracer_provider is None and telemetry._logging_handler is None

    telemetry.record_agent_registration(
        conversation_id="conversation-fallback",
        agent_name="agent-gamma",
        attributes={"agent.role": "observer"},
    )
    telemetry.record_agent_registration(
        conversation_id="conversation-fallback",
        agent_name="agent-gamma",
        attributes={"agent.role": "analyst"},
    )
    telemetry.record_scheduler_decision(
        conversation_id="conversation-fallback",
        scheduler_name="round_robin",
        agent_name="agent-gamma",
        attributes={"decision.reason": "initial"},
    )
    telemetry.record_agent_turn(
        conversation_id="conversation-fallback",
        agent_name="agent-gamma",
        prompt="Summarise",
        response="Thought: done",
        latency=0.2,
        conversation_state=_conversation_state(),
    )
    telemetry.record_llm_api_call(
        conversation_id="conversation-fallback",
        agent_name="agent-gamma",
        prompt="Summarise",
        completion="Done",
        provider="openai",
        model="gpt-test",
        latency=0.1,
        first_token_seconds=0.05,
        conversation_state=_conversation_state(),
    )
    telemetry.record_tool_call(
        conversation_id="conversation-fallback",
        agent_name="agent-gamma",
        tool_name="search",
        duration=0.01,
        output="neva project",
    )
    telemetry.record_reasoning_step(
        conversation_id="conversation-fallback",
        agent_name="agent-gamma",
        content="Thought: summarise the transcript",
        index=2,
    )
    telemetry.end_conversation("conversation-fallback")
    telemetry.shutdown()

    spans = dict(telemetry._tracer.started)
    assert set(spans) == {"neva.conversation", "neva.agent.turn", "neva.llm.call", "neva.tool.call"}
    conversation_span = spans["neva.conversation"]
    assert conversation_span.attributes["agent.role"] == "analyst"
    assert conversation_span.ended
    agent_span = spans["neva.agent.turn"]
    assert [name for name, _ in agent_span.events][:2] == ["llm.prompt", "llm.completion"]
    assert any(name == "agent.reasoning" for name, _ in agent_span.events)
    assert telemetry._meter.histograms["neva.agent.response.latency"].records
    assert telemetry._meter.counters["neva.tool.invocations"].records
    assert telemetry._structured_logger.records[-1][0] == "reasoning_step"


def test_normalise_attributes_skips_none_and_serialises_composites():
    assert _normalise_attributes(
        {"text": "value", "count": 2, "ratio": 0.5, "flag": True, "missing": None}
    ) == {"text": "value", "count": 2, "ratio": 0.5, "flag": True}
    assert _normalise_attributes({"mapping": {"key": "value"}}) == {"mapping": '{"key": "value"}'}
    assert _normalise_attributes({"opaque": {1, 2}}) == {"opaque": str({1, 2})}


def test_extract_reasoning_steps_ignores_blanks_and_unrelated_lines():
    assert _extract_reasoning_steps(None) == []
    assert _extract_reasoning_steps("") == []
    assert _extract_reasoning_steps("plain answer only") == []

    text = (
        "Thought: first\n\nReasoning: second\nunrelated line\n"
        "Step: third\nAnalysis: fourth\n  \n"
    )
    assert _extract_reasoning_steps(text) == [
        "Thought: first",
        "Reasoning: second",
        "Step: third",
        "Analysis: fourth",
    ]


def test_estimate_tokens_returns_zero_for_empty_text():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("two words") == 2


def test_configure_telemetry_shuts_down_the_previous_global():
    first = configure_telemetry()
    second = configure_telemetry()

    assert get_telemetry() is second
    assert first is not second
    assert first._logging_handler is None
