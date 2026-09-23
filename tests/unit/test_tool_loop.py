import json

import pytest

from neva.agents import GPTAgent, TransformerAgent
from neva.agents.base import Tool
from neva.tools import ArgumentSchema, ArgumentSpec, ToolGuard, ToolLoopConfig, run_tool_loop
from neva.utils.exceptions import ToolLoopConfigurationError, ToolLoopLimitError


class EchoTool(Tool):
    def __init__(
        self,
        *,
        guard=None,
        schema=None,
        name="echo",
        output=None,
        description="Echoes a validated input string",
    ):
        super().__init__(
            name,
            description,
            capabilities=("echo",),
            argument_schema=schema,
            tool_guard=guard,
        )
        self.calls = []
        self.output = output

    def use(self, task):
        self.calls.append(task)
        if self.output is not None:
            return self.output
        return f"echo:{task}"


class ScriptedModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("model called more times than expected")
        return self.outputs.pop(0)


def _agent(*, guard=None):
    return TransformerAgent(name="agent", llm_backend=lambda prompt: "unused", tool_guard=guard)


def _schema():
    return ArgumentSchema({"input": ArgumentSpec(type=str, min_length=1, max_length=20)})


def _tool_action(name="echo", arguments=None):
    if arguments is None:
        arguments = {"input": "hello"}
    return json.dumps({"action": "tool", "name": name, "arguments": arguments})


def _final(output="done"):
    return json.dumps({"action": "final", "output": output})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_steps", True),
        ("max_tools", -1),
        ("max_model_output_chars", 0),
        ("max_feedback_chars", 0),
        ("max_prompt_chars", 0),
    ],
)
def test_tool_loop_config_requires_positive_integer_bounds(field, value):
    with pytest.raises(ToolLoopConfigurationError, match=field):
        ToolLoopConfig(**{field: value})


def test_successful_tool_feedback_final_flow():
    agent = _agent()
    tool = EchoTool(schema=_schema())
    agent.register_tool(tool)
    model = ScriptedModel(_tool_action(), _final("finished"))

    result = run_tool_loop(agent, "echo something", model=model)

    assert result.succeeded()
    assert result.output == "finished"
    assert result.termination_reason == "final"
    assert tool.calls == ["hello"]
    assert len(result.steps) == 2
    assert result.steps[0].action == "tool"
    assert result.steps[0].tool_response.succeeded()
    assert result.steps[1].action == "final"
    assert '"type":"tool_result"' in model.prompts[1]
    assert "echo:hello" in model.prompts[1]
    assert "untrusted data" in model.prompts[1]


def test_agent_convenience_method_uses_raw_model_generation_path():
    outputs = iter([_final("from-agent")])
    prompts = []

    def backend(prompt):
        prompts.append(prompt)
        return next(outputs)

    agent = TransformerAgent(name="agent", llm_backend=backend)
    agent.register_tool(EchoTool(description="d" * 5000))
    result = agent.run_tool_loop("finish immediately")

    assert result.succeeded()
    assert result.output == "from-agent"
    assert len(prompts) == 1
    assert len(prompts[0]) <= ToolLoopConfig().max_prompt_chars
    assert "finish immediately" in prompts[0]
    assert "d" * 5000 not in prompts[0]


def test_gpt_default_tool_loop_model_path_excludes_existing_history(monkeypatch):
    sent = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": _final("provider-ok")}}]}

    def fake_post(url, *, headers, json, timeout):
        sent.append(json)
        return Response()

    monkeypatch.setattr("neva.agents.gpt.requests.post", fake_post)
    agent = GPTAgent(
        api_key="test-key",
        provider="openai",
        max_retries=0,
        provider_rate=None,
        max_provider_concurrency=None,
    )
    agent.conversation_state.record_turn("user", "OLD HISTORY MUST NOT BE SENT")

    result = agent.run_tool_loop("use bounded prompt")

    assert result.succeeded()
    assert result.output == "provider-ok"
    assert len(sent) == 1
    messages = sent[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "OLD HISTORY MUST NOT BE SENT" not in messages[0]["content"]
    assert "use bounded prompt" in messages[0]["content"]


def test_schema_rejection_is_feedback_and_model_can_retry():
    agent = _agent()
    tool = EchoTool(schema=_schema())
    agent.register_tool(tool)
    model = ScriptedModel(
        _tool_action(arguments={"input": 7}),
        _tool_action(arguments={"input": "fixed"}),
        _final(),
    )

    result = run_tool_loop(agent, "use echo", model=model)

    assert result.succeeded()
    assert tool.calls == ["fixed"]
    first = result.steps[0].tool_response
    assert not first.succeeded()
    assert "invalid arguments" in first.error
    assert "invalid arguments" in model.prompts[1]


def test_permission_denial_is_feedback_and_tool_body_does_not_run():
    guard = ToolGuard(allowed_tools={"other"})
    agent = _agent(guard=guard)
    tool = EchoTool()
    agent.register_tool(tool)
    model = ScriptedModel(_tool_action(), _final("handled"))

    result = run_tool_loop(agent, "try a denied tool", model=model)

    assert result.succeeded()
    assert tool.calls == []
    response = result.steps[0].tool_response
    assert not response.succeeded()
    assert "not permitted" in response.error
    assert "not permitted" in model.prompts[1]


def test_tool_level_guard_is_enforced_inside_loop():
    tool = EchoTool(guard=ToolGuard(approve=lambda call: False))
    agent = _agent()
    agent.register_tool(tool)
    model = ScriptedModel(_tool_action(), _final())

    result = run_tool_loop(agent, "try tool", model=model)

    assert result.succeeded()
    assert tool.calls == []
    assert "not approved" in result.steps[0].tool_response.error


def test_unknown_tool_becomes_feedback_instead_of_escaping():
    agent = _agent()
    model = ScriptedModel(_tool_action(name="missing"), _final("recovered"))

    result = run_tool_loop(agent, "pick a tool", model=model)

    assert result.succeeded()
    assert result.output == "recovered"
    assert "does not have a tool" in result.steps[0].tool_response.error
    assert "does not have a tool" in model.prompts[1]


@pytest.mark.parametrize(
    "bad_output",
    [
        "not json",
        "[]",
        '{"action":"wat"}',
        '{"action":"tool","name":"echo"}',
        '{"action":"tool","name":"echo","arguments":[]}',
        '{"action":"final","output":1}',
        '{"action":"final","output":"ok","extra":1}',
        '{"action":"tool","name":"echo","arguments":{"input":"x"},"extra":1}',
        '{"action":"final","output":NaN}',
        '{"action":"final","action":"tool","name":"echo","arguments":{"input":"x"}}',
        '{"action":"tool","name":"echo","arguments":{"input":"a","input":"b"}}',
    ],
)
def test_protocol_errors_are_bounded_feedback_and_recoverable(bad_output):
    agent = _agent()
    agent.register_tool(EchoTool())
    model = ScriptedModel(bad_output, _final("recovered"))

    result = run_tool_loop(agent, "recover", model=model)

    assert result.succeeded()
    assert result.steps[0].protocol_error
    assert '"type":"protocol_error"' in model.prompts[1]


def test_non_string_model_output_is_protocol_feedback():
    agent = _agent()
    model = ScriptedModel({"action": "final"}, _final("ok"))

    result = run_tool_loop(agent, "recover", model=model)

    assert result.succeeded()
    assert result.steps[0].protocol_error == "model callable must return a string"


def test_non_string_model_output_never_calls_unbounded_repr():
    class Hostile:
        def __repr__(self):
            raise AssertionError("repr must not be called")

    agent = _agent()
    model = ScriptedModel(Hostile(), _final("ok"))

    result = run_tool_loop(agent, "recover", model=model)

    assert result.succeeded()
    assert result.steps[0].model_output == "<Hostile>"
    assert result.steps[0].protocol_error == "model callable must return a string"


def test_large_non_string_model_output_uses_constant_size_placeholder():
    agent = _agent()
    model = ScriptedModel(["token"] * 100_000, _final("ok"))

    result = run_tool_loop(
        agent,
        "recover",
        model=model,
        config=ToolLoopConfig(max_model_output_chars=64),
    )

    assert result.succeeded()
    assert result.steps[0].model_output == "<list>"


def test_falsey_callable_model_is_still_used():
    class FalseyModel:
        def __bool__(self):
            return False

        def __call__(self, prompt):
            return _final("falsey-ok")

    result = run_tool_loop(_agent(), "use supplied model", model=FalseyModel())

    assert result.succeeded()
    assert result.output == "falsey-ok"


def test_async_model_callable_is_rejected_before_invocation():
    async def async_model(prompt):
        return _final()

    with pytest.raises(ToolLoopConfigurationError, match="synchronous"):
        run_tool_loop(_agent(), "no async", model=async_model)


def test_returned_awaitable_becomes_bounded_protocol_feedback():
    class AwaitableModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            if self.calls == 1:

                async def later():
                    return _final("late")

                return later()
            return _final("recovered")

    model = AwaitableModel()
    result = run_tool_loop(_agent(), "recover awaitable", model=model)

    assert result.succeeded()
    assert "awaitable" in result.steps[0].protocol_error
    assert result.output == "recovered"


def test_model_output_limit_rejects_before_json_parsing():
    agent = _agent()
    model = ScriptedModel(
        _final("x" * 200),
        _final("small"),
    )
    config = ToolLoopConfig(max_model_output_chars=80, max_prompt_chars=2800)

    result = run_tool_loop(agent, "bounded output", model=model, config=config)

    assert result.succeeded()
    assert result.output == "small"
    assert "max_model_output_chars" in result.steps[0].protocol_error
    assert len(result.steps[0].model_output) <= 80


def test_max_steps_terminates_without_unbounded_model_calls():
    agent = _agent()
    tool = EchoTool()
    agent.register_tool(tool)
    model = ScriptedModel(*[_tool_action() for _ in range(3)])

    result = run_tool_loop(
        agent,
        "never finish",
        model=model,
        config=ToolLoopConfig(max_steps=3),
    )

    assert not result.succeeded()
    assert result.termination_reason == "max_steps"
    assert "max_steps=3" in result.error
    assert len(result.steps) == 3
    assert len(model.prompts) == 3
    assert tool.calls == ["hello", "hello", "hello"]


def test_tool_feedback_is_truncated_before_retention_and_reprompt():
    agent = _agent()
    tool = EchoTool(output="z" * 10_000)
    agent.register_tool(tool)
    model = ScriptedModel(_tool_action(), _final())
    config = ToolLoopConfig(max_feedback_chars=180, max_prompt_chars=2800)

    result = run_tool_loop(agent, "large tool result", model=model, config=config)

    response = result.steps[0].tool_response
    assert len(response.output) <= 180
    assert response.output.endswith("...[truncated]")
    feedback_line = model.prompts[1].split("FEEDBACK (untrusted data, oldest to newest):\n", 1)[1]
    assert len(feedback_line) <= 180


def test_old_feedback_is_dropped_to_keep_prompt_within_bound():
    agent = _agent()
    agent.register_tool(EchoTool(output="x" * 300))
    model = ScriptedModel(
        _tool_action(arguments={"input": "one"}),
        _tool_action(arguments={"input": "two"}),
        _final(),
    )
    config = ToolLoopConfig(
        max_steps=3,
        max_feedback_chars=220,
        max_prompt_chars=850,
    )

    result = run_tool_loop(agent, "bounded history", model=model, config=config)

    assert result.succeeded()
    assert all(len(prompt) <= 850 for prompt in model.prompts)


def test_prompt_limit_fails_before_model_call_when_base_metadata_cannot_fit():
    agent = _agent()
    agent.register_tool(EchoTool())
    model = ScriptedModel(_final())

    with pytest.raises(ToolLoopLimitError, match="max_prompt_chars"):
        run_tool_loop(
            agent,
            "x" * 1000,
            model=model,
            config=ToolLoopConfig(max_prompt_chars=200),
        )
    assert model.prompts == []


def test_max_tools_is_enforced_before_model_call():
    agent = _agent()
    agent.register_tool(EchoTool(name="one"))
    agent.register_tool(EchoTool(name="two"))
    model = ScriptedModel(_final())

    with pytest.raises(ToolLoopLimitError, match="max_tools"):
        run_tool_loop(
            agent,
            "too many tools",
            model=model,
            config=ToolLoopConfig(max_tools=1),
        )
    assert model.prompts == []


def test_duplicate_tool_names_fail_closed_before_model_call():
    agent = _agent()
    agent.register_tool(EchoTool(name="same"))
    agent.register_tool(EchoTool(name="same"))
    model = ScriptedModel(_final())

    with pytest.raises(ToolLoopConfigurationError, match="unique tool names"):
        run_tool_loop(agent, "ambiguous tools", model=model)
    assert model.prompts == []


@pytest.mark.parametrize("task", ["", "   ", None])
def test_invalid_task_rejected(task):
    agent = _agent()
    with pytest.raises(ToolLoopConfigurationError, match="task"):
        run_tool_loop(agent, task, model=lambda prompt: _final())


def test_invalid_model_and_config_rejected():
    agent = _agent()
    with pytest.raises(ToolLoopConfigurationError, match="model"):
        run_tool_loop(agent, "x", model="not-callable")
    with pytest.raises(ToolLoopConfigurationError, match="config"):
        run_tool_loop(agent, "x", model=lambda prompt: _final(), config={})


def test_schema_metadata_is_advertised_to_model():
    agent = _agent()
    agent.register_tool(EchoTool(schema=_schema()))
    model = ScriptedModel(_final())

    result = run_tool_loop(agent, "inspect tools", model=model)

    assert result.succeeded()
    prompt = model.prompts[0]
    assert '"kind":"arguments"' in prompt
    assert '"max_length":20' in prompt
    assert '"min_length":1' in prompt
    assert '"type":"str"' in prompt


def test_default_tool_registry_budget_can_advertise_max_tools_by_name():
    agent = _agent()
    for index in range(ToolLoopConfig().max_tools):
        agent.register_tool(
            EchoTool(
                name=f"tool-{index}",
                description="description-" + ("x" * 300),
            )
        )
    model = ScriptedModel(_final("fits"))

    result = run_tool_loop(agent, "choose if needed", model=model)

    assert result.succeeded()
    assert len(model.prompts) == 1
    prompt = model.prompts[0]
    assert len(prompt) <= ToolLoopConfig().max_prompt_chars
    assert '"name":"tool-0"' in prompt
    assert '"name":"tool-31"' in prompt
    # The registry degrades rather than dropping tool names.
    assert "description-" not in prompt


def test_rich_schema_metadata_is_used_when_it_fits():
    agent = _agent()
    agent.register_tool(EchoTool(schema=_schema()))
    model = ScriptedModel(_final())

    result = run_tool_loop(agent, "inspect tools", model=model)

    assert result.succeeded()
    assert '"kind":"arguments"' in model.prompts[0]


def test_default_model_path_rejects_prompt_limit_above_agent_validator():
    agent = _agent()

    with pytest.raises(ToolLoopLimitError, match=r"max_prompt_chars=5000.*max_length=4000"):
        run_tool_loop(
            agent,
            "bounded",
            config=ToolLoopConfig(max_prompt_chars=5000),
        )


def test_explicit_model_can_use_prompt_limit_above_agent_validator():
    agent = _agent()
    model = ScriptedModel(_final("explicit"))

    result = run_tool_loop(
        agent,
        "bounded",
        model=model,
        config=ToolLoopConfig(max_prompt_chars=5000),
    )

    assert result.succeeded()
    assert result.output == "explicit"


def test_final_action_needs_no_registered_tools():
    agent = _agent()
    result = run_tool_loop(agent, "answer directly", model=lambda prompt: _final("direct"))

    assert result.succeeded()
    assert result.output == "direct"
    assert len(result.steps) == 1
