import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from neva.agents import TransformerAgent
from neva.agents.base import Tool, ToolCall
from neva.tools import ToolGuard, ToolLimits
from neva.utils.exceptions import (
    ToolExecutionError,
    ToolGuardConfigurationError,
    ToolTimeoutError,
)


class RecordingTool(Tool):
    def __init__(self, name="echo", *, delay=0.0, output=None, error=None):
        super().__init__(name, "Records calls")
        self.delay = delay
        self.output = output
        self.error = error
        self.calls = []

    def use(self, task):
        self.calls.append(task)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.output if self.output is not None else f"echo:{task}"


class ExplodingGuard:
    """Duck-typed guard whose evaluation fails unexpectedly."""

    def evaluate(self, call):
        raise RuntimeError("internal guard bug")

    def invoke(self, tool, payload):  # pragma: no cover - must never run
        raise AssertionError("denied calls must not execute")


def make_agent(**kwargs):
    return TransformerAgent(name="agent", llm_backend=lambda prompt: "ok", **kwargs)


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ToolGuardConfigurationError, match="timeout"):
        ToolLimits(timeout=timeout)


def test_overflowing_timeout_rejected():
    with pytest.raises(ToolGuardConfigurationError, match="timeout"):
        ToolLimits(timeout=10**400)


def test_timeout_above_platform_maximum_rejected():
    with pytest.raises(ToolGuardConfigurationError, match="timeout"):
        ToolLimits(timeout=threading.TIMEOUT_MAX + 1)
    with pytest.raises(ToolGuardConfigurationError, match="timeout"):
        ToolLimits(timeout=1e100)


def test_timeout_at_platform_maximum_accepted():
    assert ToolLimits(timeout=threading.TIMEOUT_MAX).timeout == threading.TIMEOUT_MAX


@pytest.mark.parametrize("limit", [0, -5, True, 1.5, "100"])
def test_invalid_output_limit_rejected(limit):
    with pytest.raises(ToolGuardConfigurationError, match="max_output_chars"):
        ToolLimits(max_output_chars=limit)


def test_invalid_guard_configuration_rejected():
    with pytest.raises(ToolGuardConfigurationError, match="allowed_tools"):
        ToolGuard(allowed_tools=["echo", 1])
    with pytest.raises(ToolGuardConfigurationError, match="allowed_tools"):
        ToolGuard(allowed_tools=5)
    with pytest.raises(ToolGuardConfigurationError, match="approve"):
        ToolGuard(approve="nope")
    with pytest.raises(ToolGuardConfigurationError, match="limits"):
        ToolGuard(limits={"timeout": 1})


def test_allowed_tools_string_rejected():
    with pytest.raises(ToolGuardConfigurationError, match="allowed_tools"):
        ToolGuard(allowed_tools="echo")


def test_evaluate_allowlist():
    guard = ToolGuard(allowed_tools={"echo"})
    assert guard.evaluate(ToolCall(name="echo", arguments={"input": "hi"})) is None
    reason = guard.evaluate(ToolCall(name="shell", arguments={"input": "rm"}))
    assert reason is not None
    assert "not permitted" in reason


def test_approval_hook_denies_and_receives_call():
    seen = []

    def approve(call):
        seen.append(call)
        return False

    guard = ToolGuard(approve=approve)
    call = ToolCall(name="echo", arguments={"input": "hi"})
    reason = guard.evaluate(call)
    assert reason is not None
    assert "not approved" in reason
    assert seen == [call]


def test_approval_errors_deny_without_leaking_details(caplog):
    def broken(call):
        raise RuntimeError("auth service down")

    guard = ToolGuard(approve=broken)
    with caplog.at_level(logging.DEBUG, logger="neva.tools.guard"):
        reason = guard.evaluate(ToolCall(name="echo", arguments={"input": "hi"}))
    assert reason is not None
    assert "failed" in reason
    assert "auth service down" not in reason
    assert "auth service down" in caplog.text


def test_approve_non_bool_return_denied():
    guard = ToolGuard(approve=lambda call: 1)
    reason = guard.evaluate(ToolCall(name="echo", arguments={"input": "hi"}))
    assert reason is not None
    assert "not a bool" in reason


def test_approve_awaitable_denied():
    async def approve(call):
        return True

    guard = ToolGuard(approve=approve)
    reason = guard.evaluate(ToolCall(name="echo", arguments={"input": "hi"}))
    assert reason is not None
    assert "synchronous" in reason


def test_invoke_without_limits_passes_through():
    tool = RecordingTool()
    guard = ToolGuard()
    assert guard.invoke(tool, "hello") == "echo:hello"
    assert tool.calls == ["hello"]


def test_invoke_timeout_raises():
    tool = RecordingTool(delay=0.5)
    guard = ToolGuard(limits=ToolLimits(timeout=0.05))
    with pytest.raises(ToolTimeoutError, match="execution limit"):
        guard.invoke(tool, "slow")


def test_timed_out_worker_thread_is_daemon():
    tool = RecordingTool(name="hanger", delay=0.4)
    guard = ToolGuard(limits=ToolLimits(timeout=0.05))
    with pytest.raises(ToolTimeoutError):
        guard.invoke(tool, "slow")
    workers = [t for t in threading.enumerate() if t.name == "neva-tool-hanger"]
    assert workers
    assert all(t.daemon for t in workers)


def test_tool_raised_timeout_error_surfaces_unchanged():
    tool = RecordingTool(name="sock", error=TimeoutError("socket timed out"))
    guard = ToolGuard(limits=ToolLimits(timeout=5.0))
    with pytest.raises(TimeoutError, match="socket timed out") as excinfo:
        guard.invoke(tool, "go")
    assert not isinstance(excinfo.value, ToolTimeoutError)


def test_tool_exception_surfaces_through_timeout_path():
    tool = RecordingTool(name="boom", error=ValueError("tool blew up"))
    guard = ToolGuard(limits=ToolLimits(timeout=5.0))
    with pytest.raises(ValueError, match="tool blew up"):
        guard.invoke(tool, "go")


def test_tool_execution_error_surfaces_through_timeout_path():
    tool = RecordingTool(name="boom", error=ToolExecutionError("inner failure"))
    guard = ToolGuard(limits=ToolLimits(timeout=5.0))
    with pytest.raises(ToolExecutionError, match="inner failure"):
        guard.invoke(tool, "go")


def test_invoke_truncates_output():
    tool = RecordingTool(output="x" * 50)
    guard = ToolGuard(limits=ToolLimits(max_output_chars=10))
    output = guard.invoke(tool, "anything")
    assert output.startswith("x" * 10)
    assert "truncated" in output


def test_invoke_within_output_limit_unchanged():
    tool = RecordingTool(output="short")
    guard = ToolGuard(limits=ToolLimits(max_output_chars=100))
    assert guard.invoke(tool, "anything") == "short"


def test_call_tool_denied_by_allowlist():
    tool = RecordingTool(name="shell", output="ran")
    agent = make_agent(tool_guard=ToolGuard(allowed_tools={"echo"}))
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="shell", arguments={"input": "ls"}))
    assert not response.succeeded()
    assert "not permitted" in response.error
    assert tool.calls == []  # denied calls never execute


def test_call_tool_allowed_executes():
    tool = RecordingTool(name="echo")
    agent = make_agent(tool_guard=ToolGuard(allowed_tools={"echo"}))
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "hi"}))
    assert response.succeeded()
    assert response.output == "echo:hi"


def test_call_tool_timeout_reported_as_error():
    tool = RecordingTool(name="slow", delay=0.5)
    agent = make_agent(tool_guard=ToolGuard(limits=ToolLimits(timeout=0.05)))
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="slow", arguments={"input": "go"}))
    assert not response.succeeded()
    assert "execution limit" in response.error


def test_call_tool_surfaces_tool_timeout_message():
    tool = RecordingTool(name="sock", error=TimeoutError("socket timed out"))
    agent = make_agent(tool_guard=ToolGuard(limits=ToolLimits(timeout=5.0)))
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="sock", arguments={"input": "go"}))
    assert not response.succeeded()
    assert "socket timed out" in response.error


def test_call_tool_guard_failure_returns_failed_response():
    tool = RecordingTool(name="echo")
    agent = make_agent(tool_guard=ExplodingGuard())
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "hi"}))
    assert not response.succeeded()
    assert "guard" in response.error
    assert tool.calls == []


def test_call_tool_output_truncated():
    tool = RecordingTool(name="loud", output="y" * 100)
    agent = make_agent(tool_guard=ToolGuard(limits=ToolLimits(max_output_chars=8)))
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="loud", arguments={"input": "x"}))
    assert response.succeeded()
    assert response.output.startswith("y" * 8)
    assert "truncated" in response.output


def test_call_tool_without_guard_unchanged():
    tool = RecordingTool(name="echo")
    agent = make_agent()
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="echo", arguments={"input": "hi"}))
    assert response.succeeded()
    assert response.output == "echo:hi"


def test_approval_hook_sees_calls_through_agent():
    calls = []

    def approve(call):
        calls.append(call)
        return call.arguments.get("input") != "rm -rf"

    tool = RecordingTool(name="shell")
    agent = make_agent(tool_guard=ToolGuard(approve=approve))
    agent.register_tool(tool)
    denied = agent.call_tool(ToolCall(name="shell", arguments={"input": "rm -rf"}))
    assert not denied.succeeded()
    assert tool.calls == []
    allowed = agent.call_tool(ToolCall(name="shell", arguments={"input": "ls"}))
    assert allowed.succeeded()
    assert tool.calls == ["ls"]


class ConcurrencyProbeTool(Tool):
    def __init__(self):
        super().__init__("probe", "Tracks concurrent executions")
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def use(self, task):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.04)
            return task
        finally:
            with self._lock:
                self.active -= 1


def test_isolated_process_timeout_is_hard():
    tool = RecordingTool(name="isolated", delay=1.0)
    guard = ToolGuard(limits=ToolLimits(timeout=0.05, isolate_process=True))
    started = time.monotonic()
    with pytest.raises(ToolTimeoutError, match="hard execution limit"):
        guard.invoke(tool, "slow")
    assert time.monotonic() - started < 0.8
    assert not any(
        process.name == "neva-tool-process-isolated"
        for process in __import__("multiprocessing").active_children()
    )


def test_isolated_process_truncates_before_parent_result():
    tool = RecordingTool(name="loud-process", output="z" * 10000)
    guard = ToolGuard(
        limits=ToolLimits(
            timeout=2.0,
            isolate_process=True,
            max_output_chars=7,
        )
    )
    output = guard.invoke(tool, "go")
    assert output.startswith("z" * 7)
    assert "truncated" in output
    assert len(output) < 100


def test_process_memory_limit_requires_isolation():
    with pytest.raises(ToolGuardConfigurationError, match="isolate_process"):
        ToolLimits(max_memory_bytes=1024)


def test_process_isolation_requires_timeout():
    with pytest.raises(ToolGuardConfigurationError, match="requires timeout"):
        ToolLimits(isolate_process=True)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_concurrency_limit_rejected(limit):
    with pytest.raises(ToolGuardConfigurationError, match="max_concurrency"):
        ToolLimits(max_concurrency=limit)


def test_per_tool_concurrency_quota_serializes_calls():
    tool = ConcurrencyProbeTool()
    guard = ToolGuard(limits=ToolLimits(max_concurrency=1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda value: guard.invoke(tool, value), ["a", "b", "c", "d"]))
    assert results == ["a", "b", "c", "d"]
    assert tool.max_active == 1


class TimeoutSlotTool(Tool):
    def __init__(self):
        super().__init__("slot", "Blocks the first invocation")
        self.release_first = threading.Event()
        self.started = []
        self._lock = threading.Lock()

    def use(self, task):
        with self._lock:
            self.started.append(task)
        if task == "first":
            self.release_first.wait()
        return task


def test_timed_out_thread_retains_concurrency_slot_until_worker_finishes():
    tool = TimeoutSlotTool()
    guard = ToolGuard(limits=ToolLimits(timeout=0.02, max_concurrency=1))

    with pytest.raises(ToolTimeoutError):
        guard.invoke(tool, "first")

    with ThreadPoolExecutor(max_workers=1) as pool:
        second = pool.submit(guard.invoke, tool, "second")
        time.sleep(0.04)
        assert not second.done()
        assert tool.started == ["first"]

        tool.release_first.set()
        assert second.result(timeout=1.0) == "second"
        assert tool.started == ["first", "second"]


def test_isolated_process_drains_large_result_without_pipe_deadlock():
    payload = "w" * 250_000
    tool = RecordingTool(name="large-process", output=payload)
    guard = ToolGuard(limits=ToolLimits(timeout=2.0, isolate_process=True))

    assert guard.invoke(tool, "go") == payload


def test_direct_tool_use_honours_tool_guard():
    tool = RecordingTool(name="direct")
    tool.set_tool_guard(ToolGuard(allowed_tools={"other"}))
    with pytest.raises(ToolExecutionError, match="not permitted"):
        tool.use("blocked")
    assert tool.calls == []


def test_direct_tool_use_honours_output_limit():
    tool = RecordingTool(name="direct", output="q" * 100)
    tool.set_tool_guard(ToolGuard(limits=ToolLimits(max_output_chars=6)))
    output = tool.use("go")
    assert output.startswith("q" * 6)
    assert "truncated" in output


def test_agent_and_tool_guards_both_apply():
    tool = RecordingTool(name="shared", output="r" * 50)
    tool.set_tool_guard(ToolGuard(limits=ToolLimits(max_output_chars=12)))
    agent = make_agent(
        tool_guard=ToolGuard(
            allowed_tools={"shared"},
            limits=ToolLimits(max_output_chars=5),
        )
    )
    agent.register_tool(tool)
    response = agent.call_tool(ToolCall(name="shared", arguments={"input": "go"}))
    assert response.succeeded()
    assert response.output.startswith("r" * 5)
    assert "truncated" in response.output
    assert tool.calls == ["go"]


def test_inherited_mixin_use_is_wrapped_by_tool_guard():
    class UseMixin:
        def use(self, task):
            return f"mixin:{task}"

    class MixedTool(UseMixin, Tool):
        def __init__(self):
            super().__init__("mixed", "Uses an inherited mixin implementation")

    tool = MixedTool()
    tool.set_tool_guard(ToolGuard(allowed_tools={"other"}))
    with pytest.raises(ToolExecutionError, match="not permitted"):
        tool.use("blocked")
