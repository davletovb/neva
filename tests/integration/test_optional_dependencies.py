"""Exercise optional heavyweight integrations with real local dependencies.

This module is skipped by the normal lightweight matrix and run explicitly by
CI's optional-integration job, which installs torch/transformers/FAISS/Anthropic.
No model or provider network download is required.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("faiss")
pytest.importorskip("numpy")
pytest.importorskip("anthropic")

from neva.agents import GPTAgent, TransformerAgent
from neva.utils.exceptions import BackendError
from neva.memory import FaissVectorStoreMemory


class _TinyTokenizer:
    """Minimal tensor tokenizer so the test exercises real model weights only."""

    def __call__(self, text, *, return_tensors, truncation, padding):
        assert text
        assert return_tensors == "pt"
        assert truncation is True
        assert padding is True
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        assert len(token_ids) >= 1
        return "tiny-transformer-ok"


def test_transformer_agent_runs_real_locally_saved_tiny_weights(tmp_path):
    config = transformers.T5Config(
        vocab_size=16,
        d_model=8,
        d_kv=8,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=1,
        decoder_start_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    model = transformers.T5ForConditionalGeneration(config)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    model.save_pretrained(tmp_path)

    agent = TransformerAgent(
        model_name=str(tmp_path),
        model_loader=transformers.AutoModelForSeq2SeqLM.from_pretrained,
        tokenizer_loader=lambda _: _TinyTokenizer(),
    )

    assert agent.respond("hello tiny model") == "tiny-transformer-ok"
    assert isinstance(agent._model, transformers.T5ForConditionalGeneration)
    # The second call proves the loaded model/tokenizer remain reusable.
    assert agent.respond("hello again") == "tiny-transformer-ok"


def _axis_embedder(text):
    import numpy as np

    if "alpha" in text.lower():
        return np.asarray([1.0, 0.0], dtype="float32")
    return np.asarray([0.0, 1.0], dtype="float32")


def test_faiss_real_dependency_returns_nearest_record_first():
    memory = FaissVectorStoreMemory(_axis_embedder, top_k=2)
    memory.remember("A", "alpha nearest")
    memory.remember("B", "beta farther")

    assert memory.recall(query="alpha query").splitlines() == [
        "A: alpha nearest",
        "B: beta farther",
    ]


@pytest.fixture
def anthropic_server():
    calls = []
    mode = {"empty": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(
                {
                    "path": self.path,
                    "body": json.loads(raw),
                    "api_key": self.headers.get("x-api-key"),
                }
            )
            text = "" if mode["empty"] else "anthropic-sdk-ok"
            payload = json.dumps(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                }
            ).encode("utf-8")
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
        yield f"http://127.0.0.1:{server.server_port}", calls, mode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_real_anthropic_sdk_uses_local_transport_and_parses_content(anthropic_server):
    base, calls, _ = anthropic_server
    agent = GPTAgent(
        api_key="anthropic-test-key",
        provider="anthropic",
        model="claude-test",
        api_base=base,
        max_retries=0,
        request_timeout=2.0,
    )

    assert agent._invoke_anthropic("hello") == "anthropic-sdk-ok"
    assert len(calls) == 1
    assert calls[0]["path"].endswith("/v1/messages")
    assert calls[0]["api_key"] == "anthropic-test-key"
    assert calls[0]["body"]["model"] == "claude-test"
    assert calls[0]["body"]["messages"][-1] == {"role": "user", "content": "hello"}


def test_real_anthropic_sdk_empty_text_fails_closed(anthropic_server):
    base, calls, mode = anthropic_server
    mode["empty"] = True
    agent = GPTAgent(
        api_key="anthropic-test-key",
        provider="anthropic",
        model="claude-test",
        api_base=base,
        max_retries=0,
        request_timeout=2.0,
    )

    with pytest.raises(BackendError, match="empty content"):
        agent._invoke_anthropic("hello")
    assert len(calls) == 1
