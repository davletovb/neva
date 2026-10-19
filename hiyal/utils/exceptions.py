"""Project-wide custom exception hierarchy for Hiyal."""

from __future__ import annotations

from concurrent.futures import CancelledError


class HiyalError(Exception):
    """Base class for all custom exceptions raised by Hiyal."""


class ConfigurationError(HiyalError):
    """Raised when a component receives an invalid configuration."""


class ValidationError(HiyalError):
    """Raised when input data fails validation checks."""


class DependencyError(HiyalError):
    """Raised when optional dependencies are unavailable or fail to load."""


class MissingDependencyError(DependencyError):
    """Raised when an optional dependency is not installed."""


class BackendError(HiyalError):
    """Raised when an external service or model backend fails."""


class BackendUnavailableError(BackendError):
    """Raised when a required backend cannot be initialised."""


class CircuitOpenError(BackendError):
    """Raised when a circuit breaker is open and the provider should not be called."""


class CircuitBreakerConfigurationError(ConfigurationError):
    """Raised when circuit breaker parameters are invalid."""


class CacheError(HiyalError):
    """Base class for cache related errors."""


class CacheConfigurationError(CacheError, ConfigurationError):
    """Raised when cache parameters are invalid."""


class MemoryModuleError(HiyalError):
    """Base class for memory module related errors."""


class MemoryConfigurationError(MemoryModuleError, ConfigurationError):
    """Raised when memory modules receive invalid parameters."""


class PromptValidationError(ValidationError):
    """Raised when a prompt fails validation checks."""


class RateLimiterConfigurationError(ConfigurationError):
    """Raised when rate limiter parameters are invalid."""


class SpendBudgetConfigurationError(ConfigurationError):
    """Raised when a spend budget is configured with invalid parameters."""


class SpendBudgetExceededError(HiyalError):
    """Raised when a spend budget has been exhausted or would be exceeded."""


class RateLimiterCancelledError(CancelledError):
    """Raised when a rate limiter token wait is cancelled.

    Subclasses ``concurrent.futures.CancelledError`` (itself an ``Exception``,
    distinct from ``asyncio.CancelledError``) so retry handlers that already
    special-case cancellation keep working; catch this type to distinguish a
    deliberate cancellation from an unrelated cancellation elsewhere.
    """


class AgentError(HiyalError):
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


class SchedulingError(HiyalError):
    """Raised when a scheduler cannot select an agent to run."""


class ToolError(HiyalError):
    """Base class for tool related failures."""


class ToolGuardConfigurationError(ConfigurationError):
    """Raised when tool guard or limit parameters are invalid."""


class ToolSchemaConfigurationError(ConfigurationError):
    """Raised when a tool argument schema or spec is invalid."""


class ToolExecutionError(ToolError):
    """Raised when a tool invocation fails."""


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool exceeds its configured execution time limit."""


class ToolResourceLimitError(ToolExecutionError):
    """Raised when an isolated tool exceeds or cannot apply a hard resource limit."""


class ToolNotFoundError(ToolError):
    """Raised when attempting to use a tool that is not registered."""


class ToolLoopConfigurationError(ConfigurationError):
    """Raised when the model-driven tool loop is configured unsafely."""


class ToolLoopLimitError(ToolError):
    """Raised when a bounded tool-loop resource envelope cannot be satisfied."""


class ReproducibilityError(HiyalError):
    """Base class for experiment reproducibility failures."""


class ReplayMismatchError(ReproducibilityError):
    """Raised when offline replay diverges from the recorded prompt sequence."""


class RecordedReplayError(ReproducibilityError):
    """Raised when replay reaches a model failure recorded in the tape."""
