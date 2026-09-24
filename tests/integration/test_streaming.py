"""Stream transport and delivery contracts without external credentials."""

import asyncio
import importlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from neva.agents import GPTAgent
from neva.agents.streaming import StreamInterruptedError
from neva.utils.exceptions import BackendError, RateLimiterCancelledError
from neva.utils.metrics import SpendBudget


@pytest.fixture
def stream_server():
    calls = []
    mode = {"value": "success"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            if mode["value"] == "early" and len(calls) == 1:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(payload):
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            try:
                event({"choices": [{"delta": {"content": "Hello"}}]})
                if mode["value"] == "partial":
                    return  # EOF without [DONE] is a partial failure.
                if mode["value"] == "slow":
                    time.sleep(0.08)
                event({"choices": [{"delta": {"content": " world"}}]})
                event({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", calls, mode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _agent(url, **kwargs):
    return GPTAgent(
        api_key="test",
        provider="grok",
        api_base=url,
        max_retries=1,
        retry_backoff=0,
        request_timeout=1,
        **kwargs,
    )


def test_stream_success_usage_history_cache_and_timings(stream_server):
    url, calls, _ = stream_server
    agent = _agent(url)
    events = list(agent.stream_response("ping", max_queue_size=1))
    assert [event.kind for event in events] == ["delta", "delta", "complete"]
    assert "".join(event.text for event in events[:-1]) == events[-1].text == "Hello world"
    assert 0 <= events[-1].first_token_seconds <= events[-1].completion_seconds
    assert agent._token_tracker.records == [(9, 2)]
    assert [turn.message for turn in agent.conversation_state.turns][-1] == "Hello world"
    assert calls[0]["stream"] is True
    agent.conversation_state.turns.clear()  # restore the identical request identity
    assert list(agent.stream_response("ping"))[-1].text == "Hello world"
    assert len(calls) == 1


def test_partial_stream_fails_without_retry_or_commit(stream_server):
    url, calls, mode = stream_server
    mode["value"] = "partial"
    agent = _agent(url)
    session = agent.stream_response("ping")
    iterator = iter(session)
    assert next(iterator).text == "Hello"
    with pytest.raises(StreamInterruptedError) as caught:
        list(iterator)
    assert caught.value.partial_text == "Hello"
    assert len(calls) == 1
    assert agent.conversation_state.turns == []
    assert agent._token_tracker.records  # consumed tokens still counted
    mode["value"] = "success"
    assert list(agent.stream_response("ping"))[-1].text == "Hello world"
    assert len(calls) == 2  # partial response was not cached


def test_pre_output_retry_and_first_token_latency(stream_server):
    url, calls, mode = stream_server
    mode["value"] = "early"
    agent = _agent(url)
    assert list(agent.stream_response("retry"))[-1].text == "Hello world"
    assert len(calls) == 2
    mode["value"] = "slow"
    event = list(agent.stream_response("slow"))[-1]
    assert event.completion_seconds - event.first_token_seconds >= 0.06


def test_partial_spend_is_settled_and_cache_is_empty(stream_server):
    url, _, mode = stream_server
    mode["value"] = "partial"
    budget = SpendBudget(max_cost=1.0)
    agent = _agent(url, spend_budget=budget)
    with pytest.raises(StreamInterruptedError):
        list(agent.stream_response("partial"))
    assert budget.reserved == 0
    assert budget.spent > 0
    assert agent.conversation_state.turns == []


def test_concurrent_streams_settle_their_own_usage(stream_server, monkeypatch):
    url, _, _ = stream_server
    budget = SpendBudget(max_cost=1.0)
    agent = _agent(url, spend_budget=budget)
    both = threading.Barrier(2)

    def stream(prompt, usage_state):
        if "alpha" in prompt:
            usage_state["value"] = {"prompt_tokens": 3, "completion_tokens": 1}
            yield "alpha"
        else:
            usage_state["value"] = {"prompt_tokens": 7, "completion_tokens": 2}
            yield "beta"
        both.wait(timeout=2)

    monkeypatch.setattr(agent, "_stream_provider", stream)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lambda: list(agent.stream_response("alpha"))[-1])
        second = pool.submit(lambda: list(agent.stream_response("beta"))[-1])
        assert {first.result(timeout=3).text, second.result(timeout=3).text} == {
            "alpha",
            "beta",
        }
    assert sorted(agent._token_tracker.records) == [(3, 1), (7, 2)]
    assert budget.reserved == 0
    assert budget.spent == pytest.approx(agent._call_cost(prompt_tokens=10, response_tokens=3))


def test_closing_after_last_delta_does_not_commit(stream_server, monkeypatch):
    url, _, _ = stream_server
    budget = SpendBudget(max_cost=1.0)
    agent = _agent(url, spend_budget=budget)

    def stream(_prompt, _usage_state):
        yield "hello "
        yield "world"

    monkeypatch.setattr(agent, "_stream_provider", stream)
    session = agent.stream_response("unfinished")
    iterator = iter(session)
    assert next(iterator).text == "hello "
    assert next(iterator).text == "world"
    session.close()
    with pytest.raises(RateLimiterCancelledError):
        next(iterator)
    assert agent.conversation_state.turns == []
    assert (
        agent._cache_lookup(agent.prompt_validator.validate(agent.prepare_prompt("unfinished")))
        is None
    )
    # Provider completion may be billed before the terminal event is accepted.
    assert budget.reserved == 0


def test_cached_stream_respects_the_requested_cap(stream_server):
    url, calls, _ = stream_server
    agent = _agent(url)
    list(agent.stream_response("ping"))
    agent.conversation_state.turns.clear()
    with pytest.raises(BackendError, match="Cached response exceeds max_response_chars"):
        list(agent.stream_response("ping", max_response_chars=5))
    assert agent.conversation_state.turns == []
    assert len(calls) == 1


def test_stream_saves_validated_message(stream_server):
    url, _, _ = stream_server
    agent = _agent(url)
    list(agent.stream_response("hi\x00there"))
    assert agent.conversation_state.turns[0].message == "hithere"


def test_stream_worker_preserves_caller_context(stream_server, monkeypatch):
    url, _, _ = stream_server
    agent = _agent(url)
    marker: ContextVar[str] = ContextVar("stream_marker", default="missing")
    token = marker.set("caller")

    def stream(_prompt, _usage_state):
        yield marker.get()

    monkeypatch.setattr(agent, "_stream_provider", stream)
    try:
        assert list(agent.stream_response("context"))[-1].text == "caller"
    finally:
        marker.reset(token)


def test_async_delivery_does_not_use_executor(stream_server, monkeypatch):
    url, _, _ = stream_server
    agent = _agent(url)

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("async stream must not poll through the default executor")

    monkeypatch.setattr(asyncio, "to_thread", forbidden)

    async def consume():
        return [event async for event in agent.stream_response("async")]

    assert asyncio.run(consume())[-1].text == "Hello world"


def test_stream_telemetry_records_both_latencies(stream_server, monkeypatch):
    url, _, _ = stream_server
    calls = []

    class Telemetry:
        def record_llm_api_call(self, **kwargs):
            calls.append(kwargs)

        def record_agent_turn(self, **kwargs):
            pass

    monkeypatch.setattr("neva.agents.gpt.get_telemetry", lambda: Telemetry())
    event = list(_agent(url).stream_response("timed"))[-1]
    assert calls[0]["first_token_seconds"] == event.first_token_seconds
    assert calls[0]["latency"] == event.completion_seconds


def test_bounded_queue_close_releases_provider_slot(stream_server):
    url, _, _ = stream_server
    agent = _agent(url, max_provider_concurrency=1)
    session = agent.stream_response("one", max_queue_size=1)
    iterator = iter(session)
    assert next(iterator).text == "Hello"
    session.close()
    with pytest.raises(RateLimiterCancelledError):
        next(iterator)
    # The worker may be blocked delivering the second delta/terminal event.
    deadline = time.monotonic() + 2
    while agent._provider_resources._active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not agent._provider_resources._active
    iterator.close()


def test_async_stream_can_be_cancelled(stream_server):
    url, _, _ = stream_server
    agent = _agent(url)

    async def consume():
        session = agent.stream_response("ping")
        async for event in session:
            assert event.kind == "delta"
            await session.aclose()
            break
        assert session._cancel.is_set()

    asyncio.run(consume())


def test_stream_validation_and_nonstream_backend(stream_server):
    url, _, _ = stream_server
    agent = _agent(url)
    with pytest.raises(ValueError):
        agent.stream_response("ping", max_queue_size=0)
    with pytest.raises(Exception, match="max_response_chars"):
        agent.stream_response("ping", max_response_chars=0)
    with pytest.raises(StreamInterruptedError, match="after emitting text"):
        list(agent.stream_response("ping", max_response_chars=5))
    custom = GPTAgent(llm_backend=lambda prompt: prompt)
    with pytest.raises(Exception, match="built-in provider"):
        custom.stream_response("ping")


def test_real_anthropic_sdk_stream_over_local_http(monkeypatch):
    pytest.importorskip("anthropic")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-test",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                        },
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Claude"},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": 1},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, data in events:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = GPTAgent(
            api_key="test",
            provider="anthropic",
            model="claude-test",
            api_base=f"http://127.0.0.1:{server.server_address[1]}",
            max_retries=0,
            request_timeout=2,
        )
        events = list(agent.stream_response("hi"))
        assert events[-1].text == "Claude"
        assert agent._token_tracker.records == [(3, 1)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_gemini_stream_adapter_and_usage(monkeypatch):
    class Response:
        closed = False

        def __iter__(self):
            for word in ("Gem", "ini"):
                yield type("Chunk", (), {"text": word, "usage_metadata": None})()
            yield type(
                "Chunk",
                (),
                {
                    "text": "",
                    "usage_metadata": {
                        "prompt_token_count": 4,
                        "candidates_token_count": 2,
                    },
                },
            )()

        def close(self):
            self.closed = True

    response = Response()

    class Model:
        def __init__(self, model):
            assert model == "gemini-test"

        def generate_content(self, prompt, **kwargs):
            assert "ping" in prompt
            assert kwargs["stream"] is True
            return response

    module = type(
        "GenerativeAI", (), {"configure": staticmethod(lambda **_: None), "GenerativeModel": Model}
    )
    original = importlib.import_module
    monkeypatch.setattr(
        "neva.agents.gpt._import_module",
        lambda name: module if name == "google.generativeai" else original(name),
    )
    agent = GPTAgent(api_key="test", provider="gemini", model="gemini-test", max_retries=0)
    assert list(agent.stream_response("ping"))[-1].text == "Gemini"
    assert agent._token_tracker.records == [(4, 2)]
    assert response.closed
