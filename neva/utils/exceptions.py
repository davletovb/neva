"""Project-wide custom exception hierarchy for Neva."""

from __future__ import annotations


class NevaError(Exception):
    """Base class for all custom exceptions raised by Neva."""


class ConfigurationError(NevaError):
    """Raised when a component receives an invalid configuration."""


class ValidationError(NevaError):
    """Raised when input data fails validation checks."""


class DependencyError(NevaError):
    """Raised when optional dependencies are unavailable or fail to load."""


class MissingDependencyError(DependencyError):
    """Raised when an optional dependency is not installed."""


class BackendError(NevaError):
    """Raised when an external service or model backend fails."""


class BackendUnavailableError(BackendError):
    """Raised when a required backend cannot be initialised."""


class CacheError(NevaError):
    """Base class for cache related errors."""


class CacheConfigurationError(CacheError, ConfigurationError):
    """Raised when cache parameters are invalid."""


class MemoryModuleError(NevaError):
    """Base class for memory module related errors."""


class MemoryConfigurationError(MemoryModuleError, ConfigurationError):
    """Raised when memory modules receive invalid parameters."""


class PromptValidationError(ValidationError):
    """Raised when a prompt fails validation checks."""


class RateLimiterConfigurationError(ConfigurationError):
    """Raised when rate limiter parameters are invalid."""


class AgentError(NevaError):
    """Base class for agent related errors."""


class AgentCommunicationError(AgentError):
    """Raised when agents cannot communicate with one another."""


class AgentCreationError(AgentError):
    """Raised when the manager cannot create the requested agent type."""


class AgentNotFoundError(AgentError):
    """Raised when the requested agent identifier is unknown."""


class AgentActionError(AgentError):
    """Raised when scheduling an action on an agent fails."""


class AgentManagerError(AgentError):
    """Raised when the agent manager encounters an unrecoverable error."""


class SchedulingError(NevaError):
    """Raised when a scheduler cannot select an agent to run."""


class ToolError(NevaError):
    """Base class for tool related failures."""


class ToolExecutionError(ToolError):
    """Raised when a tool invocation fails."""


class ToolNotFoundError(ToolError):
    """Raised when attempting to use a tool that is not registered."""

