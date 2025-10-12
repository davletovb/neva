import os
import sys
import types
import pytest

# Stub external dependencies before importing models
openai_stub = types.ModuleType("openai")
sys.modules["openai"] = openai_stub

transformers_stub = types.ModuleType("transformers")
class DummyAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None
class DummyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None
transformers_stub.AutoModelForSeq2SeqLM = DummyAutoModel
transformers_stub.AutoTokenizer = DummyAutoTokenizer
sys.modules["transformers"] = transformers_stub

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from exceptions import AgentCreationError
from models import AgentManager, TransformerAgent, GPTAgent


def test_create_transformer_agent_returns_transformer_agent():
    manager = AgentManager()
    agent = manager.create_agent("transformer")
    assert isinstance(agent, TransformerAgent)


def test_create_gpt_agent_returns_gpt_agent():
    manager = AgentManager()
    agent = manager.create_agent("gpt")
    assert isinstance(agent, GPTAgent)


def test_invalid_agent_type_raises_agent_creation_error():
    manager = AgentManager()
    with pytest.raises(AgentCreationError):
        manager.create_agent("invalid")


def test_manager_communicate_routes_messages_between_agents():
    manager = AgentManager()
    sender = manager.create_agent(
        "transformer",
        name="Sender",
        llm_backend=lambda prompt: f"Sender processed {prompt}",
    )

    received_prompts = []

    def receiver_backend(prompt: str) -> str:
        received_prompts.append(prompt)
        return f"Receiver heard {prompt.split(':')[-1].strip()}"

    receiver = manager.create_agent(
        "transformer",
        name="Receiver",
        llm_backend=receiver_backend,
    )

    reply = manager.communicate(str(sender.id), str(receiver.id), "Hello there")

    assert reply == "Receiver heard Hello there"
    assert received_prompts and received_prompts[0].endswith("Sender says: Hello there")
