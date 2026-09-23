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
    """Execution limits applied around ``Tool.use``.

    ``timeout`` bounds how long the caller waits for a tool and may not
    exceed ``threading.TIMEOUT_MAX`` (``Thread.join`` rejects larger values).
    When the limit
    elapses the call raises :class:`ToolTimeoutError` and the tool keeps
    running on a daemon worker thread that is not forcibly stopped and not
    re-joined at interpreter exit; a timed-out tool that never returns leaks
    that one thread. ``max_output_chars`` keeps at most that many characters
    of the tool's output followed by a truncation marker, so the returned
    string can exceed the limit by the marker's length; the tool's output is
    fully materialised first, so this bounds the returned string, not peak
    memory.
    """

    timeout: Optional[float] = None
    max_output_chars: Optional[int] = None

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


class ToolGuard:
    """Code-level guardrails for tool calls, independent of model instructions.

    ``allowed_tools=None`` permits every registered tool; a set restricts
    calls to those names. ``approve`` is called with the :class:`ToolCall`
    before execution and the call proceeds only when it returns ``True``
    (identity). Any other return value — ``False``, a truthy non-bool, or an
    awaitable such as an ``async def`` hook's coroutine — denies, as does an
    exception raised inside the hook; the exception detail is logged, not
    surfaced in the denial reason. Denials surface as failed
    :class:`ToolResponse` objects carrying a reason and never execute the
    tool.

    Limits run inside :meth:`invoke`: a timeout raises
    :class:`ToolTimeoutError` (wrapped by ``call_tool`` into an error
    response) and oversized outputs are truncated with a marker. Guardrails
    apply to ``call_tool`` invocations only; direct ``Tool.use`` calls bypass
    them.
    """

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

    def evaluate(self, call: Any) -> Optional[str]:
        """Return a denial reason, or ``None`` when the call is permitted."""

        if self.allowed_tools is not None and call.name not in self.allowed_tools:
            return f"tool '{call.name}' is not permitted by the agent's tool guard"
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

    def invoke(self, tool: Any, payload: str) -> str:
        """Run ``tool.use(payload)`` under the configured limits."""

        limits = self.limits
        if limits.timeout is None:
            output = tool.use(payload)
        else:
            output = self._invoke_with_timeout(tool, payload, limits.timeout)
        text = str(output)
        if limits.max_output_chars is not None and len(text) > limits.max_output_chars:
            text = text[: limits.max_output_chars] + _TRUNCATION_MARKER
        return text

    @staticmethod
    def _invoke_with_timeout(tool: Any, payload: str, timeout: float) -> Any:
        outcome: Dict[str, Any] = {}

        def runner() -> None:
            try:
                outcome["value"] = tool.use(payload)
            except BaseException as exc:  # re-raised in the calling thread
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
                "its daemon worker thread is not forcibly stopped and may still "
                "be running"
            )
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]
