"""Core agent abstractions and coordination utilities for Neva."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from abc import ABC, abstractmethod
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from uuid import uuid4

from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.exceptions import (
    AgentActionError,
    AgentCommunicationError,
    AgentCreationError,
    AgentManagerError,
    AgentNotFoundError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchemaConfigurationError,
)
from neva.utils.metrics import ResponseTimeTracker, batch_prompt_summary, profile_memory_usage
from neva.utils.safety import PromptValidator, sanitize_input
from neva.utils.state_management import ConversationState
from neva.utils.telemetry import get_telemetry

if TYPE_CHECKING:  # pragma: no cover - import used only for typing.
    from neva.environments.base import Environment
    from neva.tools.guard import ToolGuard
    from neva.tools.loop import ToolLoopConfig, ToolLoopResult
    from neva.utils.observer import SimulationObserver


@dataclass
class ParallelExecutionConfig:
    """Configuration options for coordinating parallel agent execution."""

    enabled: bool = False
    max_concurrency: Optional[int] = None
    batch_size: Optional[int] = None


logger = logging.getLogger(__name__)


LLMBackend = Callable[[str], str]


ToolArguments = Union[str, Mapping[str, Any]]
TOOL_TEXT_ARGUMENT_KEYS = ("input", "task", "query", "text")


@dataclass(frozen=True)
class ToolCall:
    """Structured description of a tool invocation request."""

    name: str
    arguments: ToolArguments

    @classmethod
    def from_text(
        cls,
        name: str,
        text: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ToolCall":
        payload: Dict[str, Any] = {"input": text}
        if metadata:
            payload.update(metadata)
        return cls(name=name, arguments=payload)


@dataclass(frozen=True)
class ToolResponse:
    """Normalised response captured after invoking a tool."""

    name: str
    arguments: ToolArguments
    output: str
    error: Optional[str] = None

    def succeeded(self) -> bool:
        """Return ``True`` when the tool invocation completed successfully."""

        return self.error is None


_TOOL_USE_REENTRY: ContextVar[Tuple[int, ...]] = ContextVar("neva_tool_use_reentry", default=())


class Tool(ABC):
    """Base class for validated, guardable tools used by AIAgent instances."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        implementation = cls.__dict__.get("use")

        if implementation is None:
            inherited = getattr(cls, "use", None)
            if inherited is Tool.use or getattr(inherited, "_neva_tool_wrapper", False):
                return
            implementation = inherited

        if not callable(implementation):
            return

        setattr(cls, "_tool_use_impl", implementation)
        setattr(cls, "_use_unchecked", implementation)

        @wraps(implementation)
        def guarded_use(self: "Tool", task: str) -> str:
            if id(self) in _TOOL_USE_REENTRY.get():
                return implementation(self, task)
            return Tool.use(self, task)

        setattr(guarded_use, "_neva_tool_wrapper", True)
        setattr(cls, "use", guarded_use)

    def __init__(
        self,
        name: str,
        description: str,
        *,
        capabilities: Optional[Sequence[str]] = None,
        argument_schema: Optional[Any] = None,
        tool_guard: Optional["ToolGuard"] = None,
    ) -> None:
        if self.__class__ is Tool:
            raise TypeError("Tool is a base class and must be subclassed")
        if not callable(getattr(type(self), "_tool_use_impl", None)):
            raise TypeError(f"{type(self).__name__} must implement use(task)")
        self.name = name
        self.description = description
        self.capabilities: Sequence[str] = tuple(capabilities or ())
        if argument_schema is not None:
            validator = getattr(argument_schema, "validate", None)
            if not callable(validator):
                raise ToolSchemaConfigurationError(
                    "argument_schema must provide a validate(arguments) method"
                )
            if inspect.iscoroutinefunction(validator):
                raise ToolSchemaConfigurationError("argument_schema.validate must be synchronous")
            try:
                signature = inspect.signature(validator)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                try:
                    signature.bind({})
                except TypeError as exc:
                    raise ToolSchemaConfigurationError(
                        "argument_schema.validate must accept a single " "arguments mapping"
                    ) from exc
        self.argument_schema = argument_schema
        self.tool_guard: Optional["ToolGuard"] = tool_guard

    def set_tool_guard(self, guard: Optional["ToolGuard"]) -> None:
        """Configure guardrails that apply even to direct use() calls."""

        self.tool_guard = guard

    def _schema_reason(self, arguments: ToolArguments) -> Optional[str]:
        schema = self.argument_schema
        if schema is None:
            return None
        validation_arguments = (
            dict(arguments) if not isinstance(arguments, str) else {"input": arguments}
        )
        try:
            reason = schema.validate(validation_arguments)
        except Exception as exc:
            raise ToolExecutionError(f"tool schema validation for '{self.name}' failed") from exc
        if reason is None:
            return None
        if inspect.isawaitable(reason):
            close = getattr(reason, "close", None)
            if callable(close):
                close()
            return "custom schema returned an awaitable"
        if not isinstance(reason, str):
            return "custom schema reported a violation"
        return reason

    def _evaluate_guard(self, call: ToolCall, guard: Optional["ToolGuard"]) -> Optional[str]:
        if guard is None:
            return None
        try:
            return guard.evaluate(call)
        except Exception as exc:
            raise ToolExecutionError(f"tool guard evaluation for '{self.name}' failed") from exc

    def _invoke_unchecked(self, task: str) -> Any:
        """Call the effective implementation while allowing normal super().use() delegation."""

        implementation = getattr(type(self), "_tool_use_impl", None)
        if not callable(implementation):
            raise ToolExecutionError(f"tool '{self.name}' has no executable use implementation")
        active = _TOOL_USE_REENTRY.get()
        token = _TOOL_USE_REENTRY.set(active + (id(self),))
        try:
            return implementation(self, task)
        finally:
            _TOOL_USE_REENTRY.reset(token)

    def _execute(
        self,
        payload: str,
        *,
        additional_guards: Sequence["ToolGuard"] = (),
    ) -> str:
        from neva.tools.guard import _execute_with_guards

        guards = []
        if self.tool_guard is not None:
            guards.append(self.tool_guard)
        guards.extend(additional_guards)
        return _execute_with_guards(self, payload, tuple(guards))

    def use(self, task: str) -> str:
        """Execute a direct tool call through schema validation and tool guardrails."""

        if id(self) in _TOOL_USE_REENTRY.get():
            raise ToolExecutionError(
                f"tool '{self.name}' delegated use() past its last concrete implementation"
            )

        call = ToolCall(name=self.name, arguments={"input": task})
        reason = self._evaluate_guard(call, self.tool_guard)
        if reason is not None:
            raise ToolExecutionError(reason)
        schema_reason = self._schema_reason(task)
        if schema_reason is not None:
            raise ToolExecutionError(f"invalid arguments for tool '{self.name}': {schema_reason}")
        return self._execute(task)

    def metadata(self) -> Dict[str, Sequence[str]]:
        """Return metadata describing the tool's capabilities."""

        return {"name": self.name, "capabilities": self.capabilities}


class AIAgent(ABC):
    """Abstract base class describing the behaviour of an AI agent."""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        memory: Optional[MemoryModule] = None,
        cache: Optional[LLMCache] = None,
        prompt_validator: Optional[PromptValidator] = None,
        conversation_state: Optional[ConversationState] = None,
        response_time_tracker: Optional[ResponseTimeTracker] = None,
        tool_guard: Optional["ToolGuard"] = None,
    ) -> None:
        self.id = uuid4()
        self.name = name or f"agent-{str(self.id)[:8]}"
        self.environment: Optional["Environment"] = None
        self.tools: List[Tool] = []
        self.tool_guard: Optional["ToolGuard"] = tool_guard
        self.attributes: Dict[str, str] = {}
        self._llm_backend = llm_backend
        self._memory: Optional[MemoryModule] = None
        self._cache = cache
        self._prompt_validator = prompt_validator or PromptValidator()
        self._conversation_state = conversation_state or ConversationState(agent_name=self.name)
        self._response_time_tracker = response_time_tracker or ResponseTimeTracker()
        if memory is not None:
            self.set_memory(memory)

    # ------------------------------------------------------------------
    # Tool and attribute management utilities
    # ------------------------------------------------------------------
    def set_environment(self, environment: "Environment") -> None:
        self.environment = environment
        observer = self._resolve_observer()
        if observer is not None:
            observer.watch_agent(self)

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)
        observer = self._resolve_observer()
        if observer is not None:
            observer.watch_tool(self, tool)

    def clear_tools(self) -> None:
        self.tools.clear()

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value

    def _resolve_observer(self) -> Optional["SimulationObserver"]:
        if self.environment is None:
            return None
        scheduler = getattr(self.environment, "scheduler", None)
        if scheduler is None:
            return None
        return getattr(scheduler, "simulation_observer", None)

    def generate_attribute_summary(self) -> str:
        if not self.attributes:
            return f"Agent {self.name} has no additional attributes."

        joined = ", ".join(f"{k}: {v}" for k, v in sorted(self.attributes.items()))
        return f"Agent {self.name} attributes -> {joined}."

    def generate_tool_summary(self) -> str:
        if not self.tools:
            return "No tools are currently available."

        descriptions = ", ".join(f"{tool.name} ({tool.description})" for tool in self.tools)
        return f"Available tools: {descriptions}."

    def get_tool(self, name: str) -> Tool:
        """Return the registered tool matching ``name``.

        Raises
        ------
        ToolNotFoundError
            If the agent does not expose a tool with the requested name.
        """

        for tool in self.tools:
            if tool.name == name:
                return tool
        available = ", ".join(tool.name for tool in self.tools) or "<none>"
        raise ToolNotFoundError(
            f"Agent {self.name} does not have a tool named '{name}'. "
            f"Available tools: {available}."
        )

    def _normalise_tool_input(self, arguments: ToolArguments) -> str:
        """Convert structured ``arguments`` into the string expected by tools."""

        if isinstance(arguments, str):
            return arguments

        for key in TOOL_TEXT_ARGUMENT_KEYS:
            value = arguments.get(key)
            if isinstance(value, str):
                return value

        if len(arguments) == 1:
            value = next(iter(arguments.values()))
            if isinstance(value, str):
                return value

        return json.dumps(arguments, sort_keys=True)

    def call_tool(self, call: ToolCall) -> ToolResponse:
        """Invoke a registered tool through policy, schema, and execution limits."""

        tool = self.get_tool(call.name)
        if isinstance(call.arguments, str):
            arguments: ToolArguments = call.arguments
        else:
            arguments = dict(call.arguments)

        guards = []
        if self.tool_guard is not None:
            guards.append(self.tool_guard)
        if tool.tool_guard is not None and tool.tool_guard is not self.tool_guard:
            guards.append(tool.tool_guard)

        for guard in guards:
            try:
                reason = guard.evaluate(ToolCall(name=call.name, arguments=arguments))
            except Exception:
                logger.exception("Tool guard evaluation failed for tool '%s'", call.name)
                reason = f"tool guard evaluation for '{call.name}' failed"
            if reason is not None:
                logger.warning(
                    "Tool '%s' denied for agent '%s': %s",
                    call.name,
                    self.name,
                    reason,
                )
                return ToolResponse(
                    name=tool.name,
                    arguments=arguments,
                    output="",
                    error=reason,
                )

        try:
            schema_reason = tool._schema_reason(arguments)
        except ToolExecutionError as exc:
            logger.warning(
                "Tool '%s' schema failed for agent '%s': %s",
                call.name,
                self.name,
                exc,
            )
            return ToolResponse(
                name=tool.name,
                arguments=arguments,
                output="",
                error=str(exc),
            )
        if schema_reason is not None:
            logger.warning(
                "Tool '%s' rejected for agent '%s': invalid arguments: %s",
                call.name,
                self.name,
                schema_reason,
            )
            return ToolResponse(
                name=tool.name,
                arguments=arguments,
                output="",
                error=f"invalid arguments for tool '{call.name}': {schema_reason}",
            )

        try:
            payload = self._normalise_tool_input(arguments)
        except Exception:
            logger.exception("Tool payload normalisation failed for tool '%s'", call.name)
            return ToolResponse(
                name=tool.name,
                arguments=arguments,
                output="",
                error=f"could not normalise arguments for tool '{call.name}'",
            )

        observer = self._resolve_observer()
        started = perf_counter()
        try:
            output = tool._execute(payload, additional_guards=tuple(guards))
        except ToolExecutionError as exc:
            logger.warning(
                "Tool '%s' failed for agent '%s': %s",
                tool.name,
                self.name,
                exc,
            )
            return ToolResponse(
                name=tool.name,
                arguments=arguments,
                output="",
                error=str(exc),
            )
        except Exception as exc:
            logger.exception(
                "Tool '%s' raised an unexpected error for agent '%s'",
                tool.name,
                self.name,
            )
            return ToolResponse(
                name=tool.name,
                arguments=arguments,
                output="",
                error=str(exc),
            )
        finally:
            if observer is not None:
                observer.record_tool_usage(
                    self,
                    tool,
                    duration=perf_counter() - started,
                )

        return ToolResponse(
            name=tool.name,
            arguments=arguments,
            output=str(output),
        )

    def run_tool_loop(
        self,
        task: str,
        *,
        model: Optional[Callable[[str], str]] = None,
        config: Optional["ToolLoopConfig"] = None,
    ) -> "ToolLoopResult":
        """Run a bounded model -> tool -> feedback loop for ``task``.

        Every selected tool invocation is delegated back through call_tool(),
        so schemas, permission checks, execution limits, and observer accounting
        remain the single enforcement path.
        """

        from neva.tools.loop import run_tool_loop

        return run_tool_loop(self, task, model=model, config=config)

    # ------------------------------------------------------------------
    # Memory utilities
    # ------------------------------------------------------------------
    @property
    def memory(self) -> Optional[MemoryModule]:
        return self._memory

    def set_memory(self, memory: Optional[MemoryModule]) -> None:
        self._memory = memory

    @property
    def cache(self) -> Optional[LLMCache]:
        return self._cache

    def set_cache(self, cache: Optional[LLMCache]) -> None:
        self._cache = cache

    def _cache_lookup(self, prompt: str) -> Optional[str]:
        if self._cache is None:
            return None
        return self._cache.get(prompt)

    def _cache_store(self, prompt: str, response: str) -> None:
        if self._cache is None:
            return
        self._cache.set(prompt, response)

    def _remember(self, speaker: str, message: str) -> None:
        cleaned = sanitize_input(message)
        if self._memory is not None:
            self._memory.remember(speaker, cleaned)
        self._conversation_state.record_turn(speaker, cleaned)

    def recall_memory(self, *, query: Optional[str] = None) -> str:
        if self._memory is None:
            return ""
        return self._memory.recall(query=query)

    @property
    def conversation_state(self) -> ConversationState:
        return self._conversation_state

    def set_conversation_state(self, state: ConversationState) -> None:
        self._conversation_state = state

    @property
    def prompt_validator(self) -> PromptValidator:
        return self._prompt_validator

    @property
    def response_time_tracker(self) -> ResponseTimeTracker:
        return self._response_time_tracker

    # ------------------------------------------------------------------
    # Communication helpers
    # ------------------------------------------------------------------
    @property
    def llm_backend(self) -> Optional[LLMBackend]:
        return self._llm_backend

    def set_llm_backend(self, backend: Optional[LLMBackend]) -> None:
        self._llm_backend = backend

    def generate_model_output(self, prompt: str) -> str:
        """Generate from an already composed prompt without adding agent context.

        Subclasses with built-in provider/model backends should override this
        method. The base implementation supports agents configured with an
        explicit llm_backend and preserves prompt validation plus caching.
        """

        validated_prompt = self.prompt_validator.validate(prompt)
        if self.llm_backend is None:
            raise AgentCommunicationError(
                "Agent has no raw model backend; pass model= to run_tool_loop() "
                "or override generate_model_output()."
            )
        cached = self._cache_lookup(validated_prompt)
        if cached is not None:
            return cached
        response = self.llm_backend(validated_prompt)
        self._cache_store(validated_prompt, response)
        return response

    def prepare_prompt(self, message: str) -> str:
        """Compose a prompt enriched with agent attributes and tool context."""

        validated = self._prompt_validator.validate(message)
        query = validated or None
        memory_context = self.recall_memory(query=query)
        if not memory_context and query is not None:
            memory_context = self.recall_memory()
        parts = [self.generate_attribute_summary(), self.generate_tool_summary()]
        if memory_context:
            parts.append(f"Relevant memory -> {memory_context}")
        if validated:
            parts.append(validated)
        return " ".join(part for part in parts if part).strip()

    def communicate(self, receiver: "AIAgent", message: str) -> str:
        """Send ``message`` to ``receiver`` and return their response."""

        if not isinstance(receiver, AIAgent):
            raise AgentCommunicationError("receiver must be an instance of AIAgent")
        validated = self._prompt_validator.validate(message)
        formatted = f"{self.name} says: {validated}"
        self._remember(self.name, validated)
        reply = receiver.receive(formatted, sender=self.name)
        self._remember(receiver.name, reply)
        return reply

    async def acommunicate(self, receiver: "AIAgent", message: str) -> str:
        """Asynchronously send ``message`` to ``receiver`` and return the reply."""

        if not isinstance(receiver, AIAgent):
            raise AgentCommunicationError("receiver must be an instance of AIAgent")
        validated = self._prompt_validator.validate(message)
        formatted = f"{self.name} says: {validated}"
        self._remember(self.name, validated)
        reply = await receiver.areceive(formatted, sender=self.name)
        self._remember(receiver.name, reply)
        return reply

    def process_input(self, message: str) -> str:
        """Alias for :meth:`respond` retained for backwards compatibility."""

        return self.receive(message)

    def step(self, observation: Optional[str] = None) -> str:
        """Perform a single agent step and return the generated response."""

        if observation is None and self.environment is not None:
            observation = self.environment.context()
        observation = observation or ""
        return self.receive(observation, sender="environment")

    @abstractmethod
    def respond(self, message: str) -> str:
        """Return the agent's response to ``message``."""
        raise NotImplementedError

    def _respond_with_cancel(
        self,
        message: str,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Cancellation-aware response hook for synchronous backends.

        The base implementation preserves compatibility for agents that do not
        implement cooperative cancellation. Provider-backed agents can override
        this hook and observe ``cancel_event`` in their own waits/retries.
        """

        return self.respond(message)

    async def arespond(self, message: str) -> str:
        """Asynchronously return the agent's response to ``message``.

        Cancelling the asyncio task sets a threading Event observed by agents
        that implement cooperative cancellation. The worker thread itself
        cannot be forcibly killed by Python.
        """

        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()
        future = loop.run_in_executor(
            None,
            lambda: self._respond_with_cancel(message, cancel_event=cancel_event),
        )
        try:
            return await future
        except asyncio.CancelledError:
            cancel_event.set()
            raise

    def receive(
        self,
        message: str,
        *,
        sender: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Generate a response and persist the exchange in memory."""

        speaker = sender or "system"
        validated = self._prompt_validator.validate(message)
        context = (
            self._response_time_tracker.track()
            if self._response_time_tracker is not None
            else nullcontext()
        )
        try:
            with context:
                response = self._respond_with_cancel(
                    validated,
                    cancel_event=cancel_event,
                )
        except Exception:
            self._remember(speaker, validated)
            raise
        self._remember(speaker, validated)
        self._remember(self.name, response)
        telemetry = get_telemetry()
        if telemetry is not None:
            try:
                environment = self.environment
                conversation_id = getattr(environment, "conversation_id", f"agent-{self.id}")
                telemetry.record_agent_turn(
                    conversation_id=conversation_id,
                    agent_name=self.name,
                    prompt=validated,
                    response=response,
                    latency=(
                        self._response_time_tracker.latest()
                        if self._response_time_tracker is not None
                        else None
                    ),
                    model=getattr(self, "model", None),
                    metadata={"sender": speaker},
                    conversation_state=self._conversation_state,
                )
            except Exception:  # pragma: no cover - telemetry must not break message flow.
                logger.debug("Failed to emit telemetry for agent turn", exc_info=True)
        return response

    async def areceive(self, message: str, *, sender: Optional[str] = None) -> str:
        """Asynchronously generate a response and persist the exchange.

        Async cancellation is propagated to cooperative synchronous agent
        backends through a threading Event.
        """

        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()
        future = loop.run_in_executor(
            None,
            lambda: self.receive(
                message,
                sender=sender,
                cancel_event=cancel_event,
            ),
        )
        try:
            return await future
        except asyncio.CancelledError:
            cancel_event.set()
            raise


class AgentManager:
    """Create and coordinate agents within a simulation."""

    def __init__(self, parallel_config: Optional[ParallelExecutionConfig] = None) -> None:
        self.agents: Dict[str, AIAgent] = {}
        self.groups: Dict[str, List[str]] = {}
        self.parallel_config = parallel_config or ParallelExecutionConfig()
        self._concurrency_semaphore: Optional[asyncio.Semaphore] = None
        self._loop_semaphores: Dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    @staticmethod
    def profile_population_memory(
        agent_factory: Callable[[], AIAgent], population_size: int
    ) -> Tuple[int, int]:
        """Return memory usage for instantiating ``population_size`` agents."""

        def _populate() -> None:
            manager = AgentManager()
            for _ in range(population_size):
                agent = agent_factory()
                manager.agents[str(agent.id)] = agent

        return profile_memory_usage(_populate)

    def _batched(self, items: List[str], batch_size: Optional[int]) -> List[List[str]]:
        if not items:
            return []
        if batch_size is None or batch_size <= 0:
            return [items[:]]
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def _execute_coroutine(self, coro: Coroutine[Any, Any, Dict[str, str]]) -> Dict[str, str]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        coro.close()
        raise AgentManagerError(
            "batch_communicate(concurrent=True) cannot run inside an active event loop. "
            "Await batch_communicate_async instead."
        )

    def create_agent(self, agent_type: str, **kwargs) -> AIAgent:
        agent_type = agent_type.lower()
        agent: AIAgent
        if agent_type == "transformer":
            from .transformer import TransformerAgent

            agent = TransformerAgent(**kwargs)
        elif agent_type in {"gpt", "openai"}:
            from .gpt import GPTAgent

            agent = GPTAgent(**kwargs)
        else:
            raise AgentCreationError(f"Invalid agent type: {agent_type}")

        self.agents[str(agent.id)] = agent
        return agent

    def get_agent(self, agent_id: str) -> AIAgent:
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise AgentNotFoundError(f"Unknown agent id: {agent_id}") from exc

    def remove_agent(self, agent_id: str) -> None:
        try:
            del self.agents[agent_id]
        except KeyError as exc:
            raise AgentNotFoundError(f"Unknown agent id: {agent_id}") from exc

    def communicate(self, sender_id: str, receiver_id: str, message: str) -> str:
        sender = self.get_agent(sender_id)
        receiver = self.get_agent(receiver_id)
        return sender.communicate(receiver, message)

    def batch_communicate(
        self,
        sender_id: str,
        receiver_ids: Iterable[str],
        message: str,
        *,
        concurrent: Optional[bool] = None,
    ) -> Dict[str, str]:
        receiver_list = list(receiver_ids)
        if not receiver_list:
            return {}

        concurrent = self.parallel_config.enabled if concurrent is None else concurrent
        responses: Dict[str, str] = {}
        if not concurrent:
            for receiver_id in receiver_list:
                responses[receiver_id] = self.communicate(sender_id, receiver_id, message)
            return responses
        batches = self._batched(receiver_list, self.parallel_config.batch_size)
        for batch in batches:
            batch_responses = self._execute_coroutine(
                self.batch_communicate_async(sender_id, batch, message)
            )
            responses.update(batch_responses)
        return responses

    async def batch_communicate_async(
        self, sender_id: str, receiver_ids: Iterable[str], message: str
    ) -> Dict[str, str]:
        receiver_list = list(receiver_ids)
        if not receiver_list:
            return {}

        sender = self.get_agent(sender_id)
        # Asyncio primitives belong to a loop. Share the limit between batches
        # on that loop, but never reuse it in a later asyncio.run() invocation.
        loop = asyncio.get_running_loop()
        self._loop_semaphores = {
            owner: limit for owner, limit in self._loop_semaphores.items() if not owner.is_closed()
        }
        semaphore = None
        if self.parallel_config.max_concurrency:
            semaphore = self._loop_semaphores.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.parallel_config.max_concurrency)
                self._loop_semaphores[loop] = semaphore
        self._concurrency_semaphore = semaphore

        async def _communicate(receiver_id: str) -> Tuple[str, str]:
            receiver = self.get_agent(receiver_id)
            if semaphore is None:
                reply = await sender.acommunicate(receiver, message)
            else:
                async with semaphore:
                    reply = await sender.acommunicate(receiver, message)
            return receiver_id, reply

        responses: Dict[str, str] = {}
        batches = self._batched(receiver_list, self.parallel_config.batch_size)
        for batch in batches:
            tasks = [asyncio.create_task(_communicate(receiver_id)) for receiver_id in batch]
            for receiver_id, reply in await asyncio.gather(*tasks):
                responses[receiver_id] = reply
        return responses

    def create_group(self, group_id: str, agent_ids: Sequence[str]) -> None:
        self.groups[group_id] = list(agent_ids)

    def add_to_group(self, group_id: str, agent_id: str) -> None:
        self.groups.setdefault(group_id, []).append(agent_id)

    def remove_from_group(self, group_id: str, agent_id: str) -> None:
        members = self.groups.get(group_id)
        if not members or agent_id not in members:
            raise AgentNotFoundError(f"Agent {agent_id} is not in group '{group_id}'.")
        members.remove(agent_id)

    def schedule_action(self, agent_id: str, action: str, *args, **kwargs) -> None:
        agent = self.get_agent(agent_id)
        try:
            method = getattr(agent, action)
        except AttributeError as exc:
            raise AgentActionError(f"Agent {agent_id} has no action named '{action}'") from exc
        try:
            method(*args, **kwargs)
        except Exception as exc:
            raise AgentActionError(
                f"Failed to execute action '{action}' on agent {agent_id}: {exc}"
            ) from exc

    def handle_error(self, error: Exception) -> None:
        logging.getLogger(self.__class__.__name__).error(
            "agent_manager_error", extra={"error": str(error)}
        )
        raise AgentManagerError(f"An error occurred: {error}") from error

    def conversation_summary(self) -> Dict[str, Dict[str, int]]:
        summaries: Dict[str, Dict[str, int]] = {}
        for agent in self.agents.values():
            states = [turn.message for turn in agent.conversation_state.turns]
            summaries[agent.name] = batch_prompt_summary(states)
        return summaries


class AgentFactory:
    """Factory helper mirroring :class:`AgentManager` creation logic."""

    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> AIAgent:
        manager = AgentManager()
        return manager.create_agent(agent_type, **kwargs)


class InteractionHistory:
    """Collect and store the history of interactions between agents."""

    def __init__(self) -> None:
        self.history: List[Dict[str, object]] = []

    def record(self, sender_id: str, receiver_id: str, message: str) -> None:
        interaction = {
            "time": datetime.now(),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message": message,
        }
        self.history.append(interaction)


__all__ = [
    "AIAgent",
    "AgentFactory",
    "AgentManager",
    "InteractionHistory",
    "LLMBackend",
    "ParallelExecutionConfig",
    "Tool",
]
