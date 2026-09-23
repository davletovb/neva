"""Execution guardrails for tool invocations.

Guardrails are enforced in code — allowlists, approval hooks, concurrency
quotas, and execution limits — so they hold regardless of what a model or
prompt claims. Direct :class:`~neva.agents.base.Tool` calls and
:meth:`AIAgent.call_tool` both route through the same execution path.
"""

from __future__ import annotations

import inspect
import logging
import math
import multiprocessing
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, Optional, Sequence, Tuple

from neva.utils.exceptions import (
    ToolExecutionError,
    ToolGuardConfigurationError,
    ToolResourceLimitError,
    ToolTimeoutError,
)

try:  # pragma: no cover - unavailable on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent.
    _resource = None

__all__ = ["ToolGuard", "ToolLimits"]

logger = logging.getLogger(__name__)

_TRUNCATION_MARKER = "...[tool output truncated]"


def _raw_tool_use(tool: Any, payload: str) -> Any:
    """Invoke a tool implementation without re-entering its public wrapper."""

    raw = getattr(tool, "_use_unchecked", None)
    if callable(raw):
        return raw(payload)
    return tool.use(payload)


def _truncate_output(output: Any, max_output_chars: Optional[int]) -> str:
    text = str(output)
    if max_output_chars is not None and len(text) > max_output_chars:
        return text[:max_output_chars] + _TRUNCATION_MARKER
    return text


def _send_worker_message(connection: Any, message: Tuple[Any, ...]) -> None:
    try:
        connection.send(message)
    except Exception:
        # The parent turns a missing result into a bounded worker failure.
        pass


def _apply_process_memory_limit(max_memory_bytes: Optional[int]) -> None:
    if max_memory_bytes is None:
        return
    if _resource is None or not hasattr(_resource, "RLIMIT_AS"):
        raise ToolResourceLimitError(
            "hard tool memory limits require resource.RLIMIT_AS on this platform"
        )
    _, hard = _resource.getrlimit(_resource.RLIMIT_AS)
    limit = max_memory_bytes
    if hard != _resource.RLIM_INFINITY:
        limit = min(limit, hard)
    _resource.setrlimit(_resource.RLIMIT_AS, (limit, limit))


def _isolated_tool_worker(
    tool: Any,
    payload: str,
    max_memory_bytes: Optional[int],
    max_output_chars: Optional[int],
    connection: Any,
) -> None:
    """Execute one tool call in a child process and send back a bounded result."""

    try:
        _apply_process_memory_limit(max_memory_bytes)
        try:
            output = _raw_tool_use(tool, payload)
            text = _truncate_output(output, max_output_chars)
        except MemoryError:
            if max_memory_bytes is not None:
                _send_worker_message(connection, ("resource", "memory"))
                return
            raise
        except BaseException as exc:
            try:
                connection.send(("error", exc))
            except Exception:
                _send_worker_message(connection, ("error_text", type(exc).__name__, str(exc)))
            return
        _send_worker_message(connection, ("value", text))
    except BaseException as exc:
        _send_worker_message(connection, ("error_text", type(exc).__name__, str(exc)))
    finally:
        connection.close()

@dataclass(frozen=True)
class ToolLimits:
    """Execution limits applied around a tool implementation.

    timeout bounds how long the caller waits. The compatibility default uses
    a daemon thread, which Python cannot forcibly stop after a timeout.

    isolate_process=True turns timeout into a hard process boundary: an
    over-time worker is terminated. max_memory_bytes adds an RLIMIT_AS
    address-space ceiling where the platform supports it. max_output_chars is
    applied inside an isolated worker before IPC, and max_concurrency bounds
    simultaneous executions per tool object for each guard instance.
    """

    timeout: Optional[float] = None
    max_output_chars: Optional[int] = None
    max_concurrency: Optional[int] = None
    isolate_process: bool = False
    max_memory_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.timeout is not None:
            invalid = isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
            if not invalid:
                try:
                    invalid = (
                        not math.isfinite(self.timeout)
                        or self.timeout <= 0
                        or self.timeout > threading.TIMEOUT_MAX
                    )
                except OverflowError:
                    invalid = True
            if invalid:
                raise ToolGuardConfigurationError(
                    "timeout must be a finite positive number of seconds no "
                    "greater than threading.TIMEOUT_MAX"
                )
            object.__setattr__(self, "timeout", float(self.timeout))

        if self.max_output_chars is not None:
            if (
                isinstance(self.max_output_chars, bool)
                or not isinstance(self.max_output_chars, int)
                or self.max_output_chars <= 0
            ):
                raise ToolGuardConfigurationError("max_output_chars must be a positive integer")

        if self.max_concurrency is not None:
            if (
                isinstance(self.max_concurrency, bool)
                or not isinstance(self.max_concurrency, int)
                or self.max_concurrency <= 0
            ):
                raise ToolGuardConfigurationError("max_concurrency must be a positive integer")

        if not isinstance(self.isolate_process, bool):
            raise ToolGuardConfigurationError("isolate_process must be a bool")
        if self.isolate_process and self.timeout is None:
            raise ToolGuardConfigurationError(
                "isolate_process=True requires timeout so the worker is always bounded"
            )

        if self.max_memory_bytes is not None:
            if (
                isinstance(self.max_memory_bytes, bool)
                or not isinstance(self.max_memory_bytes, int)
                or self.max_memory_bytes <= 0
            ):
                raise ToolGuardConfigurationError("max_memory_bytes must be a positive integer")
            if not self.isolate_process:
                raise ToolGuardConfigurationError(
                    "max_memory_bytes requires isolate_process=True"
                )
            if _resource is None or not hasattr(_resource, "RLIMIT_AS"):
                raise ToolGuardConfigurationError(
                    "max_memory_bytes requires resource.RLIMIT_AS on this platform"
                )


class ToolGuard:
    """Code-level guardrails for tool calls, independent of model instructions."""

    def __init__(
        self,
        *,
        allowed_tools: Optional[Iterable[str]] = None,
        approve: Optional[Callable[[Any], bool]] = None,
        limits: Optional[ToolLimits] = None,
    ) -> None:
        if allowed_tools is not None:
            if isinstance(allowed_tools, (str, bytes)):
                raise ToolGuardConfigurationError(
                    "allowed_tools must be an iterable of tool names, not a string"
                )
            names = []
            try:
                iterator = list(allowed_tools)
            except TypeError as exc:
                raise ToolGuardConfigurationError(
                    "allowed_tools must be an iterable of tool names"
                ) from exc
            for name in iterator:
                if not isinstance(name, str):
                    raise ToolGuardConfigurationError("allowed_tools must contain only strings")
                names.append(name)
            self.allowed_tools: Optional[FrozenSet[str]] = frozenset(names)
        else:
            self.allowed_tools = None
        if approve is not None and not callable(approve):
            raise ToolGuardConfigurationError("approve must be callable")
        self.approve = approve
        if limits is not None and not isinstance(limits, ToolLimits):
            raise ToolGuardConfigurationError("limits must be a ToolLimits instance")
        self.limits = limits or ToolLimits()
        self._concurrency_lock = threading.Lock()
        self._concurrency_slots: Dict[int, threading.BoundedSemaphore] = {}

    def evaluate(self, call: Any) -> Optional[str]:
        """Return a denial reason, or None when the call is permitted."""

        if self.allowed_tools is not None and call.name not in self.allowed_tools:
            return f"tool '{call.name}' is not permitted by the configured tool guard"
        if self.approve is not None:
            try:
                approved = self.approve(call)
            except Exception:
                logger.debug("Approval check for tool '%s' raised", call.name, exc_info=True)
                return f"approval check for tool '{call.name}' failed"
            if inspect.isawaitable(approved):
                close = getattr(approved, "close", None)
                if callable(close):
                    close()
                return (
                    f"approval check for tool '{call.name}' returned an awaitable; "
                    "approve must be a synchronous callable"
                )
            if not isinstance(approved, bool):
                return (
                    f"approval check for tool '{call.name}' returned "
                    f"{type(approved).__name__}, not a bool"
                )
            if not approved:
                return f"tool '{call.name}' was not approved"
        return None

    def _slot_for(self, tool: Any) -> Optional[threading.BoundedSemaphore]:
        limit = self.limits.max_concurrency
        if limit is None:
            return None
        key = id(tool)
        with self._concurrency_lock:
            slot = self._concurrency_slots.get(key)
            if slot is None:
                slot = threading.BoundedSemaphore(limit)
                self._concurrency_slots[key] = slot
            return slot

    @contextmanager
    def _execution_slot(self, tool: Any) -> Iterator[None]:
        slot = self._slot_for(tool)
        if slot is not None:
            slot.acquire()
        try:
            yield
        finally:
            if slot is not None:
                slot.release()

    def invoke(self, tool: Any, payload: str) -> str:
        """Run one tool implementation under this guard's execution limits."""

        return _execute_with_guards(tool, payload, (self,))

    @staticmethod
    def _invoke_with_timeout(tool: Any, payload: str, timeout: float) -> Any:
        outcome: Dict[str, Any] = {}

        def runner() -> None:
            try:
                outcome["value"] = _raw_tool_use(tool, payload)
            except BaseException as exc:
                outcome["error"] = exc

        worker = threading.Thread(
            target=runner,
            name=f"neva-tool-{getattr(tool, 'name', 'tool')}",
            daemon=True,
        )
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            raise ToolTimeoutError(
                f"tool '{tool.name}' exceeded the {timeout:g}s execution limit; "
                "its daemon worker thread is not forcibly stopped and may still be running"
            )
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    @staticmethod
    def _invoke_in_process(tool: Any, payload: str, limits: ToolLimits) -> str:
        timeout = limits.timeout
        if timeout is None:
            raise ToolGuardConfigurationError("process isolation requires a timeout")

        methods = multiprocessing.get_all_start_methods()
        method = "fork" if "fork" in methods else "spawn"
        context = multiprocessing.get_context(method)
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_isolated_tool_worker,
            args=(
                tool,
                payload,
                limits.max_memory_bytes,
                limits.max_output_chars,
                sender,
            ),
            name=f"neva-tool-process-{getattr(tool, 'name', 'tool')}",
        )
        try:
            try:
                process.start()
            except Exception as exc:
                raise ToolExecutionError(
                    "could not start isolated tool worker; on spawn-only platforms "
                    "the tool and its configured callables must be picklable"
                ) from exc
            finally:
                sender.close()

            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join()
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join()
                raise ToolTimeoutError(
                    f"tool '{tool.name}' exceeded the {timeout:g}s hard execution limit; "
                    "its isolated worker process was terminated"
                )

            if receiver.poll():
                message = receiver.recv()
                kind = message[0]
                if kind == "value":
                    return str(message[1])
                if kind == "error":
                    raise message[1]
                if kind == "resource":
                    raise ToolResourceLimitError(
                        f"tool '{tool.name}' exceeded its isolated memory limit"
                    )
                if kind == "error_text":
                    raise ToolExecutionError(
                        f"isolated tool '{tool.name}' failed with "
                        f"{message[1]}: {message[2]}"
                    )

            if limits.max_memory_bytes is not None:
                raise ToolResourceLimitError(
                    f"isolated tool '{tool.name}' exited without a result under its "
                    "configured memory limit"
                )
            raise ToolExecutionError(
                f"isolated tool '{tool.name}' exited without a result "
                f"(exit code {process.exitcode})"
            )
        finally:
            receiver.close()
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()


def _minimum_optional(values: Sequence[Optional[Any]]) -> Optional[Any]:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _combined_limits(guards: Sequence[ToolGuard]) -> ToolLimits:
    return ToolLimits(
        timeout=_minimum_optional([guard.limits.timeout for guard in guards]),
        max_output_chars=_minimum_optional(
            [guard.limits.max_output_chars for guard in guards]
        ),
        max_concurrency=None,
        isolate_process=any(guard.limits.isolate_process for guard in guards),
        max_memory_bytes=_minimum_optional(
            [guard.limits.max_memory_bytes for guard in guards]
        ),
    )


def _execute_with_guards(
    tool: Any,
    payload: str,
    guards: Sequence[ToolGuard],
) -> str:
    """Apply every configured quota and execute the raw tool exactly once."""

    ordered_guards = sorted({id(guard): guard for guard in guards}.values(), key=id)
    if not ordered_guards:
        return str(_raw_tool_use(tool, payload))

    limits = _combined_limits(ordered_guards)
    with ExitStack() as stack:
        for guard in ordered_guards:
            stack.enter_context(guard._execution_slot(tool))

        if limits.isolate_process:
            return ToolGuard._invoke_in_process(tool, payload, limits)
        if limits.timeout is None:
            output = _raw_tool_use(tool, payload)
        else:
            output = ToolGuard._invoke_with_timeout(tool, payload, limits.timeout)
        return _truncate_output(output, limits.max_output_chars)
