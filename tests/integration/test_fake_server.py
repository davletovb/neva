import json
import threading

import pytest

from neva.agents import GPTAgent
from neva.utils.exceptions import BackendError


def test_real_http_server_roundtrip_and_retry_classification():
    server, handler_class, calls, lock = _start_server()
    try:
        agent = GPTAgent(
            api_key="test-key",
            provider="grok",
            max_retries=2,
            retry_backoff=0.0,
            api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        )
        # First call fails 503 twice, then succeeds: exercises retry through a
        # real TCP socket and verifies headers/timeouts reach the wire.
        assert agent.respond("ping") == "recovered"
        assert len(calls) == 3
        assert calls[0]["Authorization"] == "Bearer test-key"
        assert calls[0]["model"] == "grok-4.5"
        # Cached second call never hits the server.
        assert agent.respond("ping") == "recovered"
        assert len(calls) == 3
        # 401 through a real socket fails fast, no retries.
        handler_class.mode = "auth"
        with pytest.raises(BackendError):
            agent.set_cache(None)
            agent.model = "other"
            agent.respond("ping")
        auth_failures = [c for c in calls if c.get("status") == 401]
        assert len(auth_failures) == 1
    finally:
        server.shutdown()
        server.server_close()


def _start_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    calls = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        mode = "flaky"

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            record = {
                "Authorization": self.headers.get("Authorization"),
                "model": body.get("model"),
            }
            with lock:
                if Handler.mode == "flaky" and len(calls) < 2:
                    record["status"] = 503
                    calls.append(record)
                    self.send_response(503)
                    self.end_headers()
                    return
                if Handler.mode == "auth":
                    record["status"] = 401
                    calls.append(record)
                    self.send_response(401)
                    self.end_headers()
                    return
                record["status"] = 200
                calls.append(record)
            payload = json.dumps({"choices": [{"message": {"content": "recovered"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args, **kwargs):  # silence test output
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, Handler, calls, lock
