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
from models import AgentManager, TransformerAgent, GPTAgent


def test_create_transformer_agent_returns_transformer_agent():
    manager = AgentManager()
    agent = manager.create_agent("transformer")
    assert isinstance(agent, TransformerAgent)


def test_create_gpt_agent_returns_gpt_agent():
    manager = AgentManager()
    agent = manager.create_agent("gpt")
    assert isinstance(agent, GPTAgent)


def test_invalid_agent_type_raises_value_error():
    manager = AgentManager()
    with pytest.raises(ValueError):
        manager.create_agent("invalid")
