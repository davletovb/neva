import pytest

from neva.agents import AIAgent, ToolCall, TransformerAgent
from neva.environments import Environment as BaseEnvironment
from neva.tools import Tool
from neva.utils.exceptions import ToolExecutionError, ToolNotFoundError


class DummyTool(Tool):
    def __init__(self):
        super().__init__(
            "dummy",
            "does dummy things",
            capabilities=["testing"],
        )

    def use(self, task: str) -> str:  # pragma: no cover - behaviour not exercised.
        return task


class EchoAgent(AIAgent):
    def respond(self, message: str) -> str:
        return f"echo:{message}"


class DummyEnvironment(BaseEnvironment):
    def context(self) -> str:
        return "environment context"


class FailingTool(Tool):
    def __init__(self):
        super().__init__("failing", "always errors")

    def use(self, task: str) -> str:
        raise ToolExecutionError("boom")


def test_transformer_agent_uses_backend_and_includes_context():
    prompts = []

    def backend(prompt: str) -> str:
        prompts.append(prompt)
        return f"response-{len(prompts)}"

    agent = TransformerAgent(name="Alpha", llm_backend=backend)
    agent.set_attribute("role", "tester")
    agent.register_tool(DummyTool())

    assert agent.respond("Hello world") == "response-1"
    assert "Agent Alpha attributes -> role: tester." in prompts[-1]
    assert "Available tools: dummy (does dummy things)." in prompts[-1]
    assert prompts[-1].endswith("Hello world")

    agent.process_input("Follow up")
    assert len(prompts) == 2


def test_agent_communicate_uses_receiver_response():
    sender = EchoAgent(name="Sender")
    receiver = EchoAgent(name="Receiver")
    reply = sender.communicate(receiver, "hi")
    assert reply == "echo:Sender says: hi"


def test_agent_step_uses_environment_context():
    class StepAgent(AIAgent):
        def __init__(self):
            super().__init__()
            self.history = []

        def respond(self, message: str) -> str:
            self.history.append(message)
            return message

    env = DummyEnvironment(scheduler=None)
    agent = StepAgent()
    env.register_agent(agent)
    result = agent.step()
    assert result == "environment context"
    assert agent.history == ["environment context"]


def test_agent_step_with_custom_observation():
    agent = EchoAgent(name="Echo")
    assert agent.step("custom") == "echo:custom"


def test_agent_call_tool_success():
    agent = EchoAgent(name="Echo")
    tool = DummyTool()
    agent.register_tool(tool)

    response = agent.call_tool(ToolCall.from_text("dummy", "payload"))

    assert response.name == "dummy"
    assert response.succeeded()
    assert response.output == "payload"


def test_agent_call_tool_missing_registration():
    agent = EchoAgent(name="Echo")

    with pytest.raises(ToolNotFoundError):
        agent.call_tool(ToolCall.from_text("dummy", "payload"))


def test_agent_call_tool_failure_populates_error():
    agent = EchoAgent(name="Echo")
    agent.register_tool(FailingTool())

    response = agent.call_tool(ToolCall(name="failing", arguments={"input": "payload"}))

    assert not response.succeeded()
    assert response.error == "boom"
    assert response.output == ""
