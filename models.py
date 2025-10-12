"""Core models and abstractions for Neva agents and environments."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import logging
from time import perf_counter, sleep
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from uuid import uuid4

import openai

from memory import MemoryModule
from caching import LLMCache
from metrics import (
    CostTracker,
    ResponseTimeTracker,
    TokenUsageTracker,
    batch_prompt_summary,
    profile_memory_usage,
)
from safety import PromptValidator, RateLimiter, sanitize_input
from state_management import ConversationState, SimulationSnapshot, create_snapshot
from exceptions import (
    AgentActionError,
    AgentCommunicationError,
    AgentCreationError,
    AgentManagerError,
    AgentNotFoundError,
    BackendError,
    BackendUnavailableError,
    ConfigurationError,
)

if TYPE_CHECKING:  # pragma: no cover - import used for typing only.
    from observer import SimulationObserver

# Importing transformers can be expensive and is not always required for tests or
# light-weight experimentation.  The concrete agent classes therefore import the
# library lazily, but type-checkers still benefit from the symbol being present
# here when available.  ``_TRANSFORMERS_AVAILABLE`` indicates whether the import
# succeeded and allows the agents to surface clearer error messages.
try:  # pragma: no cover - exercised implicitly when transformers is installed.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - handled by runtime guard.
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore



@dataclass
class ParallelExecutionConfig:
    """Configuration options for coordinating parallel agent execution."""

    enabled: bool = False
    max_concurrency: Optional[int] = None
    batch_size: Optional[int] = None


LLMBackend = Callable[[str], str]


class Tool(ABC):
    """An abstract base class for tools used by :class:`AIAgent` instances."""

    def __init__(
        self,
        name: str,
        description: str,
        *,
        capabilities: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.capabilities: Sequence[str] = tuple(capabilities or ())

    @abstractmethod
    def use(self, task: str) -> str:
        """Execute the tool for the given task description."""

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
    ) -> None:
        self.id = uuid4()
        self.name = name or f"agent-{str(self.id)[:8]}"
        self.environment: Optional["Environment"] = None
        self.tools: List[Tool] = []
        self.attributes: Dict[str, str] = {}
        self._llm_backend = llm_backend
        self._memory: Optional[MemoryModule] = None
        self._cache = cache
        self._prompt_validator = prompt_validator or PromptValidator()
        self._conversation_state = conversation_state or ConversationState(
            agent_name=self.name
        )
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

        descriptions = ", ".join(
            f"{tool.name} ({tool.description})" for tool in self.tools
        )
        return f"Available tools: {descriptions}."

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

    async def arespond(self, message: str) -> str:
        """Asynchronously return the agent's response to ``message``."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.respond(message))

    def receive(self, message: str, *, sender: Optional[str] = None) -> str:
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
                response = self.respond(validated)
        except Exception:
            self._remember(speaker, validated)
            raise
        self._remember(speaker, validated)
        self._remember(self.name, response)
        return response

    async def areceive(self, message: str, *, sender: Optional[str] = None) -> str:
        """Asynchronously generate a response and persist the exchange."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.receive(message, sender=sender)
        )


class TransformerAgent(AIAgent):
    """Agent powered by a Hugging Face transformer model."""

    def __init__(
        self,
        model_name: str = "t5-small",
        *,
        name: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        memory: Optional[MemoryModule] = None,
        cache: Optional[LLMCache] = None,
        model_loader: Optional[Callable[[str], object]] = None,
        tokenizer_loader: Optional[Callable[[str], object]] = None,
    ) -> None:
        super().__init__(
            name=name, llm_backend=llm_backend, memory=memory, cache=cache
        )
        self.model_name = model_name
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._model = None
        self._tokenizer = None

    def _load_transformer(self) -> None:
        if self.llm_backend is not None:
            return
        if self._model is not None and self._tokenizer is not None:
            return

        loader = self._model_loader
        tokenizer_loader = self._tokenizer_loader

        if loader is None or tokenizer_loader is None:
            if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
                raise BackendUnavailableError(
                    "Transformers library is not available. Provide an ``llm_backend`` "
                    "or install `transformers` to use TransformerAgent."
                )

            loader = AutoModelForSeq2SeqLM.from_pretrained
            tokenizer_loader = AutoTokenizer.from_pretrained

        self._model = loader(self.model_name)
        self._tokenizer = tokenizer_loader(self.model_name)

    def respond(self, message: str) -> str:
        prompt = self.prepare_prompt(message)
        validated_prompt = self.prompt_validator.validate(prompt)

        cached = self._cache_lookup(validated_prompt)
        if cached is not None:
            return cached

        if self.llm_backend is not None:
            response = self.llm_backend(validated_prompt)
            self._cache_store(validated_prompt, response)
            return response

        self._load_transformer()
        assert self._model is not None and self._tokenizer is not None

        inputs = self._tokenizer(
            validated_prompt, return_tensors="pt", truncation=True, padding=True
        )
        output_tokens = self._model.generate(**inputs, max_length=200)
        decoded = self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        self._cache_store(validated_prompt, decoded)
        return decoded


class GPTAgent(AIAgent):
    """Agent that communicates with an OpenAI-compatible large language model."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        name: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        memory: Optional[MemoryModule] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[LLMCache] = None,
        token_tracker: Optional[TokenUsageTracker] = None,
        cost_tracker: Optional[CostTracker] = None,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> None:
        resolved_cache = cache or LLMCache(max_size=256)
        super().__init__(
            name=name,
            llm_backend=llm_backend,
            memory=memory,
            cache=resolved_cache,
        )
        self.api_key = api_key
        self.model = model
        self._rate_limiter = rate_limiter or RateLimiter(rate=60, per=60.0)
        self._cache = resolved_cache
        self._token_tracker = token_tracker or TokenUsageTracker()
        self._cost_tracker = cost_tracker or CostTracker()
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._logger = logging.getLogger(self.__class__.__name__)

    def _default_backend(self) -> LLMBackend:
        if not self.api_key:
            raise ConfigurationError(
                "No API key configured for GPTAgent. Provide `llm_backend` or set `api_key`."
            )

        def _call_openai(prompt: str) -> str:
            cached = self._cache_lookup(prompt)
            if cached is not None:
                return cached

            attempt = 0
            last_error: Optional[Exception] = None
            while attempt <= self._max_retries:
                attempt += 1
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()
                start = perf_counter()
                try:
                    openai.api_key = self.api_key
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                    )
                    content = response.choices[0].message["content"].strip()
                    duration = perf_counter() - start
                    if self._token_tracker is not None:
                        prompt_tokens, response_tokens = self._token_tracker.record(
                            prompt, content
                        )
                        total_tokens = prompt_tokens + response_tokens
                    else:
                        total_tokens = 0
                    if self._cost_tracker is not None and total_tokens:
                        self._cost_tracker.add_usage(self.model, total_tokens)
                    self._cache_store(prompt, content)
                    self._logger.debug(
                        "llm_call",
                        extra={
                            "model": self.model,
                            "duration": duration,
                            "prompt_tokens": locals().get("prompt_tokens", 0),
                            "response_tokens": locals().get("response_tokens", 0),
                        },
                    )
                    return content
                except Exception as exc:  # pragma: no cover - network error path.
                    last_error = exc
                    if attempt > self._max_retries:
                        raise BackendError("LLM call failed") from exc
                    sleep_time = min(30.0, self._retry_backoff ** attempt)
                    self._logger.warning(
                        "Retrying LLM call due to error", extra={"error": str(exc)}
                    )
                    sleep(sleep_time)

            if last_error is not None:
                raise BackendError("LLM call failed") from last_error
            raise BackendError("LLM call failed")

        return _call_openai

    def respond(self, message: str) -> str:
        prompt = self.prepare_prompt(message)
        validated_prompt = self.prompt_validator.validate(prompt)
        cached = self._cache_lookup(validated_prompt)
        if cached is not None:
            return cached

        backend = self.llm_backend or self._default_backend()
        response = backend(validated_prompt)
        self._cache_store(validated_prompt, response)
        return response


class AgentManager:
    """Create and coordinate agents within a simulation."""

    def __init__(
        self, parallel_config: Optional[ParallelExecutionConfig] = None
    ) -> None:
        self.agents: Dict[str, AIAgent] = {}
        self.groups: Dict[str, List[str]] = {}
        self.parallel_config = parallel_config or ParallelExecutionConfig()

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

    def _execute_coroutine(self, coro: Awaitable[Dict[str, str]]) -> Dict[str, str]:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.run_until_complete(loop.shutdown_asyncgens())
            return result
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def create_agent(self, agent_type: str, **kwargs) -> AIAgent:
        agent_type = agent_type.lower()
        if agent_type == "transformer":
            agent = TransformerAgent(**kwargs)
        elif agent_type in {"gpt", "openai"}:
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

        concurrent = (
            self.parallel_config.enabled if concurrent is None else concurrent
        )
        if not concurrent:
            responses: Dict[str, str] = {}
            for receiver_id in receiver_list:
                responses[receiver_id] = self.communicate(
                    sender_id, receiver_id, message
                )
            return responses

        responses: Dict[str, str] = {}
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
        semaphore = (
            asyncio.Semaphore(self.parallel_config.max_concurrency)
            if self.parallel_config.max_concurrency
            else None
        )

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
        self.groups.setdefault(group_id, []).remove(agent_id)

    def schedule_action(self, agent_id: str, action: str, *args, **kwargs) -> None:
        agent = self.get_agent(agent_id)
        try:
            method = getattr(agent, action)
        except AttributeError as exc:
            raise AgentActionError(
                f"Agent {agent_id} has no action named '{action}'"
            ) from exc
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


class Scheduler(ABC):
    """Base class for iterating over a collection of agents."""

    def __init__(self) -> None:
        self.agents: List[AIAgent] = []
        self.environment: Optional["Environment"] = None
        self._paused_agents: Set[AIAgent] = set()
        self._termination_hooks: List[Callable[[AIAgent], None]] = []

    def set_environment(self, environment: "Environment") -> None:
        self.environment = environment

    def record_metrics(self, active_agent: Optional[AIAgent] = None) -> None:
        observer = getattr(self, "simulation_observer", None)
        if observer is not None:
            try:
                observer.collect_data(
                    list(self.agents), self.environment, active_agent=active_agent
                )
            except TypeError:
                # Backwards compatibility for observers that do not yet accept the
                # ``active_agent`` keyword argument (for example, monkeypatched
                # tests).
                observer.collect_data(list(self.agents), self.environment)

    # ------------------------------------------------------------------
    # Agent lifecycle controls
    # ------------------------------------------------------------------
    def register_termination_hook(self, hook: Callable[[AIAgent], None]) -> None:
        """Invoke ``hook`` whenever an agent is terminated."""

        self._termination_hooks.append(hook)

    def pause(self, agent: AIAgent) -> None:
        """Temporarily remove ``agent`` from the active scheduling pool."""

        if agent in self.agents:
            self._paused_agents.add(agent)

    def resume(self, agent: AIAgent) -> None:
        """Return ``agent`` to the active scheduling pool."""

        self._paused_agents.discard(agent)

    def is_paused(self, agent: AIAgent) -> bool:
        """Return ``True`` when ``agent`` is currently paused."""

        return agent in self._paused_agents

    def terminate(self, agent: AIAgent) -> None:
        """Remove ``agent`` from the scheduler and trigger termination hooks."""

        if agent not in self.agents:
            return

        self.agents = [existing for existing in self.agents if existing is not agent]
        self._paused_agents.discard(agent)
        self._handle_agent_removal(agent)
        for hook in list(self._termination_hooks):
            hook(agent)

    def _handle_agent_removal(self, agent: AIAgent) -> None:
        """Allow subclasses to update internal state after ``agent`` removal."""

        # Subclasses override as necessary. The base implementation intentionally
        # does nothing.
        return

    def _active_agents(self) -> List[AIAgent]:
        """Return the list of agents that are not currently paused."""

        return [agent for agent in self.agents if agent not in self._paused_agents]

    @abstractmethod
    def add(self, agent: AIAgent, **kwargs) -> None:
        """Register an agent with the scheduler."""

    @abstractmethod
    def get_next_agent(self) -> Optional[AIAgent]:
        """Return the next agent to act."""


class Environment(ABC):
    """Coordinate agents and schedulers while maintaining shared state."""

    def __init__(self, scheduler: Optional[Scheduler] = None) -> None:
        self.state: Dict[str, object] = {}
        self.scheduler = scheduler
        self.agents: List[AIAgent] = []
        if self.scheduler is not None:
            self.scheduler.set_environment(self)

    def register_agent(self, agent: AIAgent) -> None:
        agent.set_environment(self)
        self.agents.append(agent)
        if self.scheduler is not None:
            self.scheduler.add(agent)

    def context(self) -> str:
        """Return a textual description of the environment state."""

        return ""

    def step(self) -> Optional[str]:
        if self.scheduler is None or not self.agents:
            return None

        agent = self.scheduler.get_next_agent()
        if agent is None:
            return None
        return agent.step(self.context())

    def run(self, steps: int) -> List[Optional[str]]:
        return [self.step() for _ in range(steps)]

    def snapshot(self) -> SimulationSnapshot:
        return create_snapshot(
            environment_state=dict(self.state),
            agent_states=(agent.conversation_state for agent in self.agents),
        )

    def restore(self, snapshot: SimulationSnapshot) -> None:
        self.state = dict(snapshot.environment_state)
        name_to_state = {
            name: ConversationState.from_dict(state)
            for name, state in snapshot.agent_states.items()
        }
        for agent in self.agents:
            if agent.name in name_to_state:
                agent.set_conversation_state(name_to_state[agent.name])

