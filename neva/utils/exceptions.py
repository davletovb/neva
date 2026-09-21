"""Project-wide custom exception hierarchy for Neva."""

from __future__ import annotations

from concurrent.futures import CancelledError


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


class CircuitOpenError(BackendError):
    """Raised when a circuit breaker is open and the provider should not be called."""


class CircuitBreakerConfigurationError(ConfigurationError):
    """Raised when circuit breaker parameters are invalid."""


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


class SpendBudgetConfigurationError(ConfigurationError):
    """Raised when a spend budget is configured with invalid parameters."""


class SpendBudgetExceededError(NevaError):
    """Raised when a spend budget has been exhausted or would be exceeded."""


class RateLimiterCancelledError(CancelledError):
    """Raised when a rate limiter token wait is cancelled.

    Subclasses ``concurrent.futures.CancelledError`` (itself an ``Exception``,
    distinct from ``asyncio.CancelledError``) so retry handlers that already
    special-case cancellation keep working; catch this type to distinguish a
    deliberate cancellation from an unrelated cancellation elsewhere.
    """


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


class ToolGuardConfigurationError(ConfigurationError):
    """Raised when tool guard or limit parameters are invalid."""


class ToolSchemaConfigurationError(ConfigurationError):
    """Raised when a tool argument schema or spec is invalid."""


class ToolExecutionError(ToolError):
    """Raised when a tool invocation fails."""


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool exceeds its configured execution time limit."""


class ToolNotFoundError(ToolError):
    """Raised when attempting to use a tool that is not registered."""
