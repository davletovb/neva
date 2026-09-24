"""Exercise the real ``google-genai`` SDK against a loopback endpoint.

Skipped unless the optional ``google-genai`` package is installed. CI's
optional-integrations job runs this module with the manifest's pinned version;
no network access or live credentials are involved — the client's ``base_url``
points at a local server.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("google.genai")

from neva.agents import GPTAgent
from neva.utils.exceptions import BackendError
from neva.utils.metrics import TokenUsageTracker

REQUESTS: list = []


class _Handler(BaseHTTPRequestHandler):
    """Serve canned Gemini responses; behaviour is keyed by the model name."""

    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        REQUESTS.append(
            {
                "path": self.path,
                "api_key": self.headers.get("x-goog-api-key"),
                "body": json.loads(raw.decode("utf-8")),
            }
        )
        model = self.path.split("/models/")[1].split(":")[0]
        if model == "slow-test":
            time.sleep(1.0)
        if ":streamGenerateContent" in self.path:
            payload, ctype = _stream_payload(), "text/event-stream"
        elif model == "blocked-test":
            payload = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode("utf-8")
            ctype = "application/json"
        else:
            payload = json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "loopback-gemini-ok"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 7,
                    },
                }
            ).encode("utf-8")
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def _stream_payload() -> bytes:
    chunks = [
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "loop-1"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "loop-2"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 6,
                "totalTokenCount": 11,
            },
        },
    ]
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode("utf-8") for chunk in chunks)


@pytest.fixture()
def gemini_endpoint():
    REQUESTS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _agent(endpoint: str, **kwargs) -> GPTAgent:
    return GPTAgent(
        api_key="loopback-key",
        provider="gemini",
        api_base=endpoint,
        max_retries=0,
        **kwargs,
    )


def test_real_sdk_posts_to_the_configured_base_url_and_parses_content(gemini_endpoint):
    tracker = TokenUsageTracker()
    agent = _agent(gemini_endpoint, model="gemini-test", token_tracker=tracker)
    assert "loopback-gemini-ok" in agent.respond("hello")
    assert tracker.records[-1] == (3, 4)

    request = REQUESTS[-1]
    assert request["path"].endswith("/models/gemini-test:generateContent")
    assert request["api_key"] == "loopback-key"
    sent = request["body"]["contents"][0]["parts"][0]["text"]
    assert sent.endswith("hello")  # the agent's system preamble travels with the prompt
    assert request["body"]["generationConfig"]["maxOutputTokens"] == 1024


def test_real_sdk_streams_chunks_with_final_usage(gemini_endpoint):
    agent = _agent(gemini_endpoint, model="gemini-test")
    usage: dict = {}
    chunks = list(agent._stream_provider("hello", usage))
    assert "".join(chunks) == "loop-1loop-2"
    assert usage["value"] == {"prompt_tokens": 5, "completion_tokens": 6}
    assert ":streamGenerateContent" in REQUESTS[-1]["path"]


def test_real_sdk_blocked_response_fails_closed(gemini_endpoint):
    agent = _agent(gemini_endpoint, model="blocked-test")
    with pytest.raises(BackendError, match="empty content"):
        agent._invoke_gemini("hello")


def test_real_sdk_honors_request_timeout_in_milliseconds(gemini_endpoint):
    agent = _agent(gemini_endpoint, model="slow-test", request_timeout=0.25)
    started = time.monotonic()
    with pytest.raises(BackendError):
        agent.respond("hello")
    assert time.monotonic() - started < 0.9
    assert len(REQUESTS) == 1
