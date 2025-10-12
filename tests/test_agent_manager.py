import asyncio
import os
import sys
import types
from typing import Callable, Dict, List

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
from models import AgentManager, ParallelExecutionConfig, TransformerAgent, GPTAgent


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


def test_batch_communicate_supports_parallel_execution():
    config = ParallelExecutionConfig(enabled=True, max_concurrency=2)
    manager = AgentManager(parallel_config=config)

    sender = manager.create_agent(
        "transformer",
        name="Sender",
        llm_backend=lambda prompt: prompt,
    )

    call_counts: Dict[str, int] = {}

    def backend_for(name: str) -> Callable[[str], str]:
        def _inner(prompt: str) -> str:
            call_counts[name] = call_counts.get(name, 0) + 1
            return f"{name} heard {prompt}"

        return _inner

    receiver_ids: List[str] = []
    for index in range(3):
        receiver = manager.create_agent(
            "transformer",
            name=f"Receiver-{index}",
            llm_backend=backend_for(f"Receiver-{index}"),
        )
        receiver_ids.append(str(receiver.id))

    responses = manager.batch_communicate(
        str(sender.id), receiver_ids, "Ping", concurrent=True
    )

    assert set(responses) == set(receiver_ids)
    assert all(count == 1 for count in call_counts.values())


def test_batch_communicate_async_returns_all_responses() -> None:
    config = ParallelExecutionConfig(enabled=True, max_concurrency=1, batch_size=2)
    manager = AgentManager(parallel_config=config)

    sender = manager.create_agent(
        "transformer",
        name="Sender",
        llm_backend=lambda prompt: prompt,
    )

    call_counts: Dict[str, int] = {}

    def backend_for(name: str) -> Callable[[str], str]:
        def _inner(prompt: str) -> str:
            call_counts[name] = call_counts.get(name, 0) + 1
            return f"{name} processed {prompt}"

        return _inner

    receiver_ids: List[str] = []
    for index in range(4):
        receiver = manager.create_agent(
            "transformer",
            name=f"Receiver-{index}",
            llm_backend=backend_for(f"Receiver-{index}"),
        )
        receiver_ids.append(str(receiver.id))

    responses = asyncio.run(
        manager.batch_communicate_async(str(sender.id), receiver_ids, "Broadcast")
    )

    assert set(responses) == set(receiver_ids)
    assert all(count == 1 for count in call_counts.values())


def test_profile_population_memory_reports_usage() -> None:
    def factory() -> TransformerAgent:
        return TransformerAgent(llm_backend=lambda prompt: prompt)

    current, peak = AgentManager.profile_population_memory(factory, population_size=5)

    assert isinstance(current, int) and isinstance(peak, int)
    assert peak >= current >= 0
