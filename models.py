"""Core models and abstractions for Neva agents and environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set
from uuid import uuid4

import openai

from memory import MemoryModule

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
    ) -> None:
        self.id = uuid4()
        self.name = name or f"agent-{str(self.id)[:8]}"
        self.environment: Optional["Environment"] = None
        self.tools: List[Tool] = []
        self.attributes: Dict[str, str] = {}
        self._llm_backend = llm_backend
        self._memory: Optional[MemoryModule] = None
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

    def _remember(self, speaker: str, message: str) -> None:
        if self._memory is not None:
            self._memory.remember(speaker, message)

    def recall_memory(self, *, query: Optional[str] = None) -> str:
        if self._memory is None:
            return ""
        return self._memory.recall(query=query)

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

        query = message or None
        memory_context = self.recall_memory(query=query)
        parts = [self.generate_attribute_summary(), self.generate_tool_summary()]
        if memory_context:
            parts.append(f"Relevant memory -> {memory_context}")
        if message:
            parts.append(message)
        return " ".join(part for part in parts if part).strip()

    def communicate(self, receiver: "AIAgent", message: str) -> str:
        """Send ``message`` to ``receiver`` and return their response."""

        if not isinstance(receiver, AIAgent):
            raise TypeError("receiver must be an instance of AIAgent")
        formatted = f"{self.name} says: {message}"
        self._remember(self.name, message)
        reply = receiver.receive(formatted, sender=self.name)
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

    def receive(self, message: str, *, sender: Optional[str] = None) -> str:
        """Generate a response and persist the exchange in memory."""

        speaker = sender or "system"
        try:
            response = self.respond(message)
        except Exception:
            self._remember(speaker, message)
            raise
        self._remember(speaker, message)
        self._remember(self.name, response)
        return response


class TransformerAgent(AIAgent):
    """Agent powered by a Hugging Face transformer model."""

    def __init__(
        self,
        model_name: str = "t5-small",
        *,
        name: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        memory: Optional[MemoryModule] = None,
        model_loader: Optional[Callable[[str], object]] = None,
        tokenizer_loader: Optional[Callable[[str], object]] = None,
    ) -> None:
        super().__init__(name=name, llm_backend=llm_backend, memory=memory)
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
                raise RuntimeError(
                    "Transformers library is not available. Provide an ``llm_backend`` "
                    "or install `transformers` to use TransformerAgent."
                )

            loader = AutoModelForSeq2SeqLM.from_pretrained
            tokenizer_loader = AutoTokenizer.from_pretrained

        self._model = loader(self.model_name)
        self._tokenizer = tokenizer_loader(self.model_name)

    def respond(self, message: str) -> str:
        prompt = self.prepare_prompt(message)

        if self.llm_backend is not None:
            return self.llm_backend(prompt)

        self._load_transformer()
        assert self._model is not None and self._tokenizer is not None

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True
        )
        output_tokens = self._model.generate(**inputs, max_length=200)
        return self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)


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
    ) -> None:
        super().__init__(name=name, llm_backend=llm_backend, memory=memory)
        self.api_key = api_key
        self.model = model

    def _default_backend(self) -> LLMBackend:
        if not self.api_key:
            raise RuntimeError(
                "No API key configured for GPTAgent. Provide `llm_backend` or set "
                "`api_key`."
            )

        def _call_openai(prompt: str) -> str:
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message["content"].strip()

        return _call_openai

    def respond(self, message: str) -> str:
        prompt = self.prepare_prompt(message)
        backend = self.llm_backend or self._default_backend()
        return backend(prompt)


class AgentManager:
    """Create and coordinate agents within a simulation."""

    def __init__(self) -> None:
        self.agents: Dict[str, AIAgent] = {}
        self.groups: Dict[str, List[str]] = {}

    def create_agent(self, agent_type: str, **kwargs) -> AIAgent:
        agent_type = agent_type.lower()
        if agent_type == "transformer":
            agent = TransformerAgent(**kwargs)
        elif agent_type in {"gpt", "openai"}:
            agent = GPTAgent(**kwargs)
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        self.agents[str(agent.id)] = agent
        return agent

    def get_agent(self, agent_id: str) -> AIAgent:
        return self.agents[agent_id]

    def remove_agent(self, agent_id: str) -> None:
        del self.agents[agent_id]

    def communicate(self, sender_id: str, receiver_id: str, message: str) -> str:
        sender = self.get_agent(sender_id)
        receiver = self.get_agent(receiver_id)
        return sender.communicate(receiver, message)

    def create_group(self, group_id: str, agent_ids: Sequence[str]) -> None:
        self.groups[group_id] = list(agent_ids)

    def add_to_group(self, group_id: str, agent_id: str) -> None:
        self.groups.setdefault(group_id, []).append(agent_id)

    def remove_from_group(self, group_id: str, agent_id: str) -> None:
        self.groups.setdefault(group_id, []).remove(agent_id)

    def schedule_action(self, agent_id: str, action: str, *args, **kwargs) -> None:
        agent = self.get_agent(agent_id)
        getattr(agent, action)(*args, **kwargs)

    def handle_error(self, error: Exception) -> None:
        raise RuntimeError(f"An error occurred: {error}")


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

