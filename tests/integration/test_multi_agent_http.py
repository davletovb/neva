"""Exercise real HTTP transport, history, and recovery without live providers."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from neva.agents import GPTAgent
from neva.environments.base import Environment
from neva.schedulers.round_robin import RoundRobinScheduler
from neva.utils.exceptions import BackendError


@pytest.fixture
def provider_server():
    calls = []
    lock = threading.Lock()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                number = len(calls) + 1
                calls.append(
                    {"path": self.path, "authorization": self.headers["Authorization"], **body}
                )
            if self.path.startswith("/slow/"):
                # Safety bound prevents hangs if the client's timeout wiring regresses.
                release.wait(timeout=3)
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": f"reply-{number}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }
            ).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                # Expected when a read timeout closes the client socket.
                pass

        def log_message(self, *args, **kwargs):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # Join request workers during close; none may outlive fixture cleanup.
    server.daemon_threads = False
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


class DialogueEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__(RoundRobinScheduler(), **kwargs)
        self.transcript = []

    def context(self):
        return "Start a discussion" if not self.transcript else self.transcript[-1]

    def on_turn_complete(self, response):
        self.transcript.append(response)


def make_agent(base, name, **kwargs):
    return GPTAgent(
        name=name,
        api_key="offline-test-key",
        provider="openai",
        api_base=base,
        max_retries=0,
        **kwargs,
    )


def test_two_agent_simulation_sends_history_and_collects_completed_metrics(provider_server):
    base, calls, _ = provider_server
    environment = DialogueEnvironment()
    agents = [make_agent(base + "/ok", name) for name in ("Alice", "Bob")]
    for agent in agents:
        environment.register_agent(agent)

    assert environment.run(4) == ["reply-1", "reply-2", "reply-3", "reply-4"]
    assert len(calls) == 4
    assert all(call["path"] == "/ok/chat/completions" for call in calls)
    assert all(call["authorization"] == "Bearer offline-test-key" for call in calls)
    assert all(call["model"] == "gpt-4o-mini" for call in calls)
    assert [len(call["messages"]) for call in calls] == [1, 1, 3, 3]
    # Alice's second turn contains her previous reply plus Bob's new observation.
    assert calls[2]["messages"][1] == {"role": "assistant", "content": "reply-1"}
    assert "reply-2" in calls[2]["messages"][-1]["content"]
    assert calls[3]["messages"][1] == {"role": "assistant", "content": "reply-2"}
    assert "reply-3" in calls[3]["messages"][-1]["content"]
    assert all(len(agent.conversation_state.turns) == 4 for agent in agents)
    metrics = environment.scheduler.simulation_observer.latest_snapshot()
    assert metrics["completed_turn_count"] == 4
    assert metrics["failed_turn_count"] == 0
    assert metrics["dialogue_length"] == 4


def test_real_read_timeout_is_wrapped_without_retry(provider_server):
    base, calls, release = provider_server
    agent = make_agent(base + "/slow", "Slow", request_timeout=0.1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.respond, "hello")
        try:
            with pytest.raises(BackendError) as error:
                future.result(timeout=5)
            assert isinstance(error.value.__cause__, requests.exceptions.ReadTimeout)
            assert len(calls) == 1
        finally:
            release.set()


def test_environment_continues_with_healthy_agent_after_real_timeout(provider_server):
    base, calls, release = provider_server
    environment = DialogueEnvironment(error_policy="return", error_value="unavailable")
    environment.register_agent(make_agent(base + "/slow", "Slow", request_timeout=0.1))
    environment.register_agent(make_agent(base + "/ok", "Healthy", request_timeout=2))
    try:
        assert environment.run(2) == ["unavailable", "reply-2"]
        assert len(calls) == 2
        # Fallback values aren't completed turns or transcript messages.
        assert environment.transcript == ["reply-2"]
        metrics = environment.scheduler.simulation_observer.latest_snapshot()
        assert metrics["scheduled_turn_count"] == 2
        assert metrics["failed_turn_count"] == 1
        assert metrics["completed_turn_count"] == 1
        assert metrics["dialogue_length"] == 1
    finally:
        release.set()
