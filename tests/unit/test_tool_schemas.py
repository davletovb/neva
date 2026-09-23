import pytest

from neva.agents import TransformerAgent
from neva.agents.base import Tool, ToolCall
from neva.tools import ArgumentSchema, ArgumentSpec
from neva.utils.exceptions import ToolSchemaConfigurationError


class SchemaTool(Tool):
    def __init__(self, *, schema=None, name="echo"):
        super().__init__(name, "Echoes input", argument_schema=schema)
        self.calls = []

    def use(self, task):
        self.calls.append(task)
        return f"echo:{task}"


def make_agent(**kwargs):
    return TransformerAgent(name="agent", llm_backend=lambda prompt: "ok", **kwargs)


# --- ArgumentSpec construction -------------------------------------------------


@pytest.mark.parametrize("bad_type", [5, "str", (str, 5), [str]])
def test_invalid_spec_type_rejected(bad_type):
    with pytest.raises(ToolSchemaConfigurationError, match="type"):
        ArgumentSpec(type=bad_type)


@pytest.mark.parametrize("bound", [0, -1, True, 2.5])
def test_invalid_length_bounds_rejected(bound):
    with pytest.raises(ToolSchemaConfigurationError, match="min_length"):
        ArgumentSpec(min_length=bound)
    with pytest.raises(ToolSchemaConfigurationError, match="max_length"):
        ArgumentSpec(max_length=bound)


def test_contradictory_length_bounds_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="min_length"):
        ArgumentSpec(min_length=10, max_length=5)


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), True, "10"])
def test_invalid_value_bounds_rejected(bound):
    with pytest.raises(ToolSchemaConfigurationError, match="min_value"):
        ArgumentSpec(type=int, min_value=bound)
    with pytest.raises(ToolSchemaConfigurationError, match="max_value"):
        ArgumentSpec(type=int, max_value=bound)


def test_contradictory_value_bounds_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="min_value"):
        ArgumentSpec(type=int, min_value=10, max_value=5)


def test_empty_choices_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="choices"):
        ArgumentSpec(choices=())


def test_invalid_required_flag_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="required"):
        ArgumentSpec(required="yes")


# --- ArgumentSchema construction -----------------------------------------------


def test_invalid_schema_fields_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="fields"):
        ArgumentSchema(["input"])
    with pytest.raises(ToolSchemaConfigurationError, match="fields"):
        ArgumentSchema({1: ArgumentSpec()})
    with pytest.raises(ToolSchemaConfigurationError, match="fields"):
        ArgumentSchema({"input": {"type": str}})
    with pytest.raises(ToolSchemaConfigurationError, match="allow_extra"):
        ArgumentSchema({"input": ArgumentSpec()}, allow_extra="yes")


def test_tool_rejects_non_schema_argument_schema():
    with pytest.raises(ToolSchemaConfigurationError, match="argument_schema"):
        SchemaTool(schema="not-a-schema")


# --- validate() -----------------------------------------------------------------


def test_valid_arguments_pass():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str, max_length=10)})
    assert schema.validate({"input": "hello"}) is None


def test_missing_required_argument_reported():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)})
    reason = schema.validate({})
    assert reason is not None
    assert "missing" in reason
    assert "input" in reason


def test_optional_missing_argument_allowed():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str, required=False)})
    assert schema.validate({}) is None


def test_unexpected_argument_reported():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)})
    reason = schema.validate({"input": "hi", "extra": 1})
    assert reason is not None
    assert "unexpected" in reason
    assert "extra" in reason


def test_unexpected_arguments_allowed_when_configured():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)}, allow_extra=True)
    assert schema.validate({"input": "hi", "extra": 1}) is None


def test_wrong_type_reported():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)})
    reason = schema.validate({"input": 5})
    assert reason is not None
    assert "input" in reason
    assert "int" in reason


def test_bool_is_not_accepted_as_number():
    schema = ArgumentSchema({"count": ArgumentSpec(type=int)})
    reason = schema.validate({"count": True})
    assert reason is not None
    assert "count" in reason


def test_union_types_accepted():
    schema = ArgumentSchema({"value": ArgumentSpec(type=(int, str))})
    assert schema.validate({"value": 5}) is None
    assert schema.validate({"value": "five"}) is None
    reason = schema.validate({"value": 5.5})
    assert reason is not None
    assert "float" in reason


def test_length_bounds_enforced():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str, min_length=2, max_length=4)})
    assert schema.validate({"input": "ab"}) is None
    assert schema.validate({"input": "abcd"}) is None
    short = schema.validate({"input": "a"})
    assert short is not None
    assert "at least" in short
    long = schema.validate({"input": "abcde"})
    assert long is not None
    assert "at most" in long


def test_value_bounds_enforced():
    schema = ArgumentSchema({"count": ArgumentSpec(type=int, min_value=1, max_value=5)})
    assert schema.validate({"count": 1}) is None
    assert schema.validate({"count": 5}) is None
    low = schema.validate({"count": 0})
    assert low is not None
    assert ">=" in low
    high = schema.validate({"count": 6})
    assert high is not None
    assert "<=" in high


def test_choices_enforced():
    schema = ArgumentSchema({"mode": ArgumentSpec(type=str, choices=("fast", "slow"))})
    assert schema.validate({"mode": "fast"}) is None
    reason = schema.validate({"mode": "medium"})
    assert reason is not None
    assert "one of" in reason


def test_unsized_type_with_length_bounds_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="length"):
        ArgumentSpec(type=int, min_length=1)
    with pytest.raises(ToolSchemaConfigurationError, match="length"):
        ArgumentSpec(type=(int, float), max_length=5)
    # mixed tuples stay allowed; mismatches fail closed at validation time
    ArgumentSpec(type=(str, int), max_length=5)


def test_mismatched_choices_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="choices"):
        ArgumentSpec(type=int, choices=("a",))
    with pytest.raises(ToolSchemaConfigurationError, match="choices"):
        ArgumentSpec(type=int, choices=(True,))
    ArgumentSpec(type=(int, str), choices=(1, "a"))


def test_duplicate_choices_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="choices"):
        ArgumentSpec(choices=("a", "a"))


def test_nan_choice_rejected():
    with pytest.raises(ToolSchemaConfigurationError, match="choices"):
        ArgumentSpec(type=float, choices=(float("nan"),))


def test_schema_fields_are_immutable():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)})
    with pytest.raises(TypeError):
        schema.fields["other"] = ArgumentSpec(type=str)


def test_tool_rejects_wrong_shape_validator():
    class WrongShape:
        def validate(self, name, value):
            return None

    with pytest.raises(ToolSchemaConfigurationError, match="argument_schema"):
        SchemaTool(schema=WrongShape())


def test_tool_rejects_zero_arg_validator():
    class ZeroArg:
        def validate(self):
            return None

    with pytest.raises(ToolSchemaConfigurationError, match="argument_schema"):
        SchemaTool(schema=ZeroArg())


def test_tool_rejects_async_validator():
    class AsyncSchema:
        async def validate(self, arguments):
            return None

    with pytest.raises(ToolSchemaConfigurationError, match="argument_schema"):
        SchemaTool(schema=AsyncSchema())


def test_tool_accepts_duck_typed_validator():
    class CustomSchema:
        def validate(self, arguments):
            return None if arguments.get("input") == "ok" else "custom violation"

    tool = SchemaTool(schema=CustomSchema())
    agent = make_agent()
    agent.register_tool(tool)
    assert agent.call_tool(ToolCall(name="echo", arguments={"input": "ok"})).succeeded()
    denied = agent.call_tool(ToolCall(name="echo", arguments={"input": "nope"}))
    assert not denied.succeeded()
    assert "custom violation" in denied.error


def test_nan_value_rejected_by_bounds():
    schema = ArgumentSchema({"v": ArgumentSpec(type=float, min_value=0, max_value=1)})
    reason = schema.validate({"v": float("nan")})
    assert reason is not None
    assert "finite" in reason
    assert schema.validate({"v": float("inf")}) is not None
    assert schema.validate({"v": float("-inf")}) is not None
    assert schema.validate({"v": 0.5}) is None


def test_nan_value_rejected_through_call_tool():
    tool = SchemaTool(
        schema=ArgumentSchema({"input": ArgumentSpec(type=float, min_value=0, max_value=1)})
    )
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": float("nan")}))
    assert not response.succeeded()
    assert "finite" in response.error
    assert tool.calls == []


def test_huge_int_value_reported_not_raised():
    schema = ArgumentSchema({"v": ArgumentSpec(type=int, max_value=100)})
    reason = schema.validate({"v": 10**400})
    assert reason is not None
    assert "<=" in reason
    reason_low = ArgumentSchema({"v": ArgumentSpec(type=int, min_value=-100)}).validate(
        {"v": -(10**400)}
    )
    assert reason_low is not None
    assert ">=" in reason_low


def test_len_raising_value_reported():
    class LyingLen:
        def __len__(self):
            raise ValueError("no length here")

    schema = ArgumentSchema({"v": ArgumentSpec(type=LyingLen, max_length=5)})
    reason = schema.validate({"v": LyingLen()})
    assert reason is not None
    assert "length" in reason


def test_mixed_unexpected_keys_reported():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)})
    reason = schema.validate({"input": "x", 1: "a", "z": "b"})
    assert reason is not None
    assert "unexpected" in reason


def test_non_string_schema_reason_normalized():
    class FalseSchema:
        def validate(self, arguments):
            return False

    tool = SchemaTool(schema=FalseSchema())
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "hi"}))
    assert not response.succeeded()
    assert "violation" in response.error
    assert tool.calls == []


def test_integer_bounds_preserved_exactly():
    spec = ArgumentSpec(type=int, min_value=2**53 + 1)
    assert spec.min_value == 2**53 + 1  # not coerced to float
    reason = spec.validate("v", 2**53)
    assert reason is not None
    assert ">=" in reason
    assert spec.validate("v", 2**53 + 1) is None
    # float bounds stay floats
    assert ArgumentSpec(type=float, min_value=0.5).min_value == 0.5


def test_huge_integer_bound_allowed_and_rendered_safely():
    spec = ArgumentSpec(type=int, min_value=10**400)
    reason = spec.validate("v", 0)
    assert reason is not None
    assert ">=" in reason
    assert len(reason) < 400  # bound rendering is truncated


def test_boolean_accepted_for_nonnumeric_supertypes():
    assert ArgumentSchema({"v": ArgumentSpec(type=object)}).validate({"v": True}) is None
    assert ArgumentSchema({"v": ArgumentSpec(type=(int, object))}).validate({"v": True}) is None
    assert ArgumentSchema({"v": ArgumentSpec(type=bool)}).validate({"v": True}) is None
    # numeric declarations still require an explicit bool
    reason = ArgumentSchema({"v": ArgumentSpec(type=int)}).validate({"v": True})
    assert reason is not None
    reason_tuple = ArgumentSchema({"v": ArgumentSpec(type=(int, str))}).validate({"v": True})
    assert reason_tuple is not None


def test_non_serializable_arguments_fail_closed():
    tool = SchemaTool(schema=None)
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"blob": {1, 2}, "raw": b"x"}))
    assert not response.succeeded()
    assert "normalis" in response.error
    assert tool.calls == []


def test_schema_validation_runs_before_payload_normalization():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str)}))
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"blob": {1, 2}}))
    assert not response.succeeded()
    assert "invalid arguments" in response.error  # schema rejects, no TypeError escape
    assert tool.calls == []


def test_non_mapping_arguments_rejected():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str)})
    reason = schema.validate(["input"])
    assert reason is not None
    assert "mapping" in reason


def test_from_text_metadata_needs_allow_extra():
    strict = ArgumentSchema({"input": ArgumentSpec(type=str)})
    call = ToolCall.from_text("echo", "hi", metadata={"source": "unit-test"})
    reason = strict.validate(dict(call.arguments))
    assert reason is not None
    assert "source" in reason
    lenient = ArgumentSchema({"input": ArgumentSpec(type=str)}, allow_extra=True)
    assert lenient.validate(dict(call.arguments)) is None


def test_first_failure_returned():
    schema = ArgumentSchema({"input": ArgumentSpec(type=str), "count": ArgumentSpec(type=int)})
    reason = schema.validate({"input": 5, "count": "five", "extra": 1})
    assert reason is not None
    assert "unexpected" in reason


# --- call_tool integration ------------------------------------------------------


def test_call_tool_rejects_invalid_arguments():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str, max_length=4)}))
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "too long"}))
    assert not response.succeeded()
    assert "invalid arguments" in response.error
    assert tool.calls == []


def test_call_tool_passes_valid_arguments():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str, max_length=10)}))
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "hi"}))
    assert response.succeeded()
    assert response.output == "echo:hi"
    assert tool.calls == ["hi"]


def test_call_tool_validates_string_arguments_as_input():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str, max_length=3)}))
    agent = make_agent()
    agent.register_tool(tool)
    rejected = agent.call_tool(ToolCall(name="echo", arguments="abcd"))
    assert not rejected.succeeded()
    assert "invalid arguments" in rejected.error
    accepted = agent.call_tool(ToolCall(name="echo", arguments="abc"))
    assert accepted.succeeded()


def test_call_tool_tool_without_schema_unchanged():
    tool = SchemaTool(schema=None)
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "anything", "extra": 1}))
    assert response.succeeded()


def test_guard_denial_takes_precedence_over_schema():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str)}))
    agent = make_agent(tool_guard=make_guard())
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": 5}))
    assert not response.succeeded()
    assert "not permitted" in response.error
    assert tool.calls == []


def make_guard():
    from neva.tools import ToolGuard

    return ToolGuard(allowed_tools={"other"})


def test_schema_failure_returns_failed_response_when_validate_raises():
    class ExplodingSchema:
        def validate(self, arguments):
            raise RuntimeError("internal schema bug")

    tool = SchemaTool(schema=None)
    tool.argument_schema = ExplodingSchema()
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "hi"}))
    assert not response.succeeded()
    assert "schema" in response.error
    assert tool.calls == []


def test_direct_tool_use_validates_schema_before_body():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str, max_length=3)}))
    with pytest.raises(Exception, match="invalid arguments"):
        tool.use("too long")
    assert tool.calls == []


def test_direct_tool_use_accepts_schema_valid_input():
    tool = SchemaTool(schema=ArgumentSchema({"input": ArgumentSpec(type=str, max_length=10)}))
    assert tool.use("hello") == "echo:hello"
    assert tool.calls == ["hello"]
