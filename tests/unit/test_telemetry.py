from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")

from neva.utils.telemetry import (
    TelemetryManager,
    configure_telemetry,
    get_telemetry,
    reset_telemetry,
)


class DummySpan:
    def __init__(self, attributes=None):
        self.attributes = dict(attributes or {})
        self.events = []
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append((name, dict(attributes or {})))

    def end(self):
        self.ended = True


class DummySpanContext:
    def __init__(self, span):
        self._span = span

    def __enter__(self):
        return self._span

    def __exit__(self, exc_type, exc, tb):
        self._span.end()
        return False


class DummyTracer:
    def __init__(self):
        self.started = []

    def start_span(self, name, attributes=None):
        span = DummySpan(attributes)
        self.started.append((name, span))
        return span

    def start_as_current_span(self, name, context=None, attributes=None, kind=None):
        span = DummySpan(attributes)
        span.context = context
        span.kind = kind
        self.started.append((name, span))
        return DummySpanContext(span)


class DummyHistogram:
    def __init__(self, name):
        self.name = name
        self.records = []

    def record(self, value, attributes=None):
        self.records.append((value, dict(attributes or {})))


class DummyCounter:
    def __init__(self, name):
        self.name = name
        self.records = []

    def add(self, value, attributes=None):
        self.records.append((value, dict(attributes or {})))


class DummyMeter:
    def __init__(self):
        self.histograms = {}
        self.counters = {}

    def create_histogram(self, name, **_kwargs):
        instrument = DummyHistogram(name)
        self.histograms[name] = instrument
        return instrument

    def create_counter(self, name, **_kwargs):
        instrument = DummyCounter(name)
        self.counters[name] = instrument
        return instrument


class DummyLogger:
    def __init__(self):
        self.records = []

    def info(self, msg, *args, **kwargs):
        rendered = msg % args if args else msg
        self.records.append((rendered, dict(kwargs.get("extra", {}))))

    def debug(self, *args, **kwargs):  # pragma: no cover - compatibility shim.
        return


@pytest.fixture(autouse=True)
def cleanup_global_telemetry():
    reset_telemetry()
    yield
    reset_telemetry()


def test_record_agent_turn_tracks_metrics_and_traces():
    tracer = DummyTracer()
    meter = DummyMeter()
    logger = DummyLogger()

    telemetry = TelemetryManager(tracer=tracer, meter=meter, structured_logger=logger)
    telemetry.record_agent_turn(
        conversation_id="conversation-1",
        agent_name="agent-alpha",
        prompt="Hello there",
        response="General Kenobi",
        latency=0.42,
    )

    names = [name for name, _ in tracer.started]
    assert "neva.conversation" in names
    assert "neva.agent.turn" in names
    agent_span = next(span for name, span in tracer.started if name == "neva.agent.turn")
    assert ("llm.prompt", {"llm.prompt": "Hello there"}) in agent_span.events
    latency_records = meter.histograms["neva.agent.response.latency"].records
    assert pytest.approx(latency_records[0][0]) == 0.42
    log_event, payload = logger.records[-1]
    assert log_event == "agent_turn"
    assert payload["agent.name"] == "agent-alpha"


def test_record_llm_api_call_emits_metrics():
    tracer = DummyTracer()
    meter = DummyMeter()
    logger = DummyLogger()
    telemetry = TelemetryManager(tracer=tracer, meter=meter, structured_logger=logger)

    telemetry.record_llm_api_call(
        conversation_id="conversation-2",
        agent_name="beta",
        prompt="Compute integral",
        completion="Result is 42",
        provider="openai",
        model="gpt-test",
        latency=1.25,
        prompt_tokens=15,
        completion_tokens=10,
    )

    names = [name for name, _ in tracer.started]
    assert "neva.llm.call" in names
    llm_latency = meter.histograms["neva.llm.api.latency"].records
    assert pytest.approx(llm_latency[0][0]) == 1.25
    token_counts = meter.counters["neva.llm.total.tokens"].records[0][0]
    assert token_counts == 25
    log_event, payload = logger.records[-1]
    assert log_event == "llm_call"
    assert payload["llm.model"] == "gpt-test"


def test_record_tool_call_tracks_usage():
    tracer = DummyTracer()
    meter = DummyMeter()
    logger = DummyLogger()
    telemetry = TelemetryManager(tracer=tracer, meter=meter, structured_logger=logger)

    telemetry.record_tool_call(
        conversation_id="conversation-3",
        agent_name="gamma",
        tool_name="search",
        duration=0.1,
        arguments={"query": "neva"},
        output="neva project",
    )

    names = [name for name, _ in tracer.started]
    assert "neva.tool.call" in names
    invocation_count = meter.counters["neva.tool.invocations"].records[0][0]
    assert invocation_count == 1
    log_event, payload = logger.records[-1]
    assert log_event == "tool_call"
    assert payload["tool.name"] == "search"


def test_configure_telemetry_controls_global_state():
    telemetry = configure_telemetry()
    try:
        assert get_telemetry() is telemetry
    finally:
        reset_telemetry()
        assert get_telemetry() is None


def test_record_agent_registration_and_scheduler_decision():
    tracer = DummyTracer()
    meter = DummyMeter()
    logger = DummyLogger()
    telemetry = TelemetryManager(tracer=tracer, meter=meter, structured_logger=logger)

    telemetry.record_agent_registration(
        conversation_id="conversation-42",
        agent_name="sentinel",
        attributes={"agent.role": "observer"},
    )

    assert tracer.started[0][0] == "neva.conversation"
    conversation_span = tracer.started[0][1]
    assert (
        "agent.registered",
        {"agent.name": "sentinel", "agent.role": "observer"},
    ) in conversation_span.events

    log_event, payload = logger.records[-1]
    assert log_event == "agent_registered"
    assert payload["agent.name"] == "sentinel"
    assert payload["agent.role"] == "observer"

    telemetry.record_scheduler_decision(
        conversation_id="conversation-42",
        scheduler_name="round_robin",
        agent_name="sentinel",
        attributes={"decision.reason": "initial"},
    )

    _, scheduler_span = tracer.started[0]
    assert (
        "scheduler.decision",
        {
            "conversation.id": "conversation-42",
            "scheduler.name": "round_robin",
            "agent.name": "sentinel",
            "decision.reason": "initial",
        },
    ) in scheduler_span.events

    log_event, payload = logger.records[-1]
    assert log_event == "scheduler_decision"
    assert payload["scheduler.name"] == "round_robin"
    assert payload["decision.reason"] == "initial"


def test_record_reasoning_step_adds_event_and_log():
    tracer = DummyTracer()
    meter = DummyMeter()
    logger = DummyLogger()
    telemetry = TelemetryManager(tracer=tracer, meter=meter, structured_logger=logger)

    telemetry.record_reasoning_step(
        conversation_id="conversation-99",
        agent_name="analyst",
        content="Thought: evaluate options",
        index=3,
        metadata={"confidence": 0.75},
    )

    span = tracer.started[0][1]
    assert span.events[-1][0] == "agent.reasoning"
    assert span.events[-1][1]["reasoning.index"] == 3
    assert span.events[-1][1]["reasoning.content"] == "Thought: evaluate options"

    log_event, payload = logger.records[-1]
    assert log_event == "reasoning_step"
    assert payload["agent.name"] == "analyst"
    assert payload["confidence"] == 0.75
