"""Transport edge cases for the real requests-based provider path."""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
import urllib3

from neva.agents import GPTAgent
from neva.utils.exceptions import BackendError


def _agent(base: str, *, timeout: float = 0.1, max_context_chars: int = 24_000) -> GPTAgent:
    return GPTAgent(
        api_key="offline-test-key",
        provider="openai",
        api_base=base,
        max_retries=0,
        request_timeout=timeout,
        max_context_chars=max_context_chars,
    )


@pytest.fixture
def malformed_server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            if self.path.startswith("/invalid-json/"):
                payload = b"{not-json"
            elif self.path.startswith("/array/"):
                payload = b"[]"
            else:
                payload = json.dumps({"choices": "not-a-list"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args, **kwargs):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("invalid-json", "invalid JSON"),
        ("array", "malformed JSON response"),
        ("wrong-shape", "Empty response"),
    ],
)
def test_real_http_malformed_responses_fail_once(malformed_server, path, message):
    base, calls = malformed_server
    agent = _agent(f"{base}/{path}")

    with pytest.raises(BackendError) as caught:
        agent.respond("hello")

    assert len(calls) == 1
    assert isinstance(caught.value.__cause__, BackendError)
    assert message in str(caught.value.__cause__)


@pytest.fixture
def saturated_connect_listener():
    if sys.platform != "linux":
        pytest.skip("deterministic listen-backlog connect timeout is Linux-specific")

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    blockers = []
    saturated = False

    try:
        for _ in range(8):
            client = socket.socket()
            client.settimeout(0.05)
            try:
                client.connect(listener.getsockname())
            except TimeoutError:
                client.close()
                saturated = True
                break
            blockers.append(client)

        if not saturated:
            pytest.skip("could not deterministically saturate the local accept queue")

        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        for client in blockers:
            client.close()
        listener.close()


def test_real_connect_timeout_is_wrapped_without_retry(saturated_connect_listener):
    agent = _agent(saturated_connect_listener, timeout=0.05)

    with pytest.raises(BackendError) as caught:
        agent.respond("hello")

    assert isinstance(caught.value.__cause__, requests.exceptions.ConnectTimeout)


@pytest.fixture
def write_stall_server():
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.2)
    release = threading.Event()
    accepted = threading.Event()

    def worker():
        connection = None
        try:
            while not release.is_set():
                try:
                    connection, _ = listener.accept()
                    break
                except socket.timeout:
                    continue
                except OSError:
                    return
            if connection is None:
                return
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            accepted.set()
            # Deliberately never read the request body.
            release.wait(timeout=3)
        finally:
            if connection is not None:
                connection.close()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", accepted
    finally:
        release.set()
        listener.close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_real_request_body_write_timeout_is_wrapped(write_stall_server, monkeypatch):
    base, accepted = write_stall_server

    original_connect = urllib3.connection.HTTPConnection.connect

    def connect_with_small_send_buffer(connection):
        original_connect(connection)
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

    monkeypatch.setattr(
        urllib3.connection.HTTPConnection,
        "connect",
        connect_with_small_send_buffer,
    )

    agent = _agent(base, timeout=0.05, max_context_chars=2_000_000)
    agent.prompt_validator.max_length = 2_000_000

    with pytest.raises(BackendError) as caught:
        agent.respond("x" * 1_000_000)

    assert accepted.wait(timeout=1)
    # requests/urllib3 surfaces a socket send timeout as a ConnectionError
    # (rather than ReadTimeout); Neva classifies it as retryable transport I/O.
    assert isinstance(caught.value.__cause__, requests.exceptions.ConnectionError)
    assert "timed out" in str(caught.value.__cause__).lower()
