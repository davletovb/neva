"""Large language model agent with support for multiple providers."""

from __future__ import annotations

import importlib
import json
import logging
import math
import threading
from contextvars import ContextVar
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional

import requests

from neva.agents.base import AIAgent, LLMBackend
from neva.agents.streaming import StreamEvent, StreamInterruptedError, StreamSession
from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.context_budget import ModelContextBudget, RequestContent
from neva.utils.exceptions import (
    BackendError,
    CircuitOpenError,
    ConfigurationError,
    RateLimiterCancelledError,
    SpendBudgetExceededError,
)
from neva.utils.metrics import (
    CostTracker,
    ResponseTimeTracker,
    SpendBudget,
    SpendReservation,
    TokenUsageTracker,
    estimate_token_count,
)
from neva.utils.observability.telemetry import get_telemetry
from neva.utils.provider_resources import (
    ProviderPermit,
    ProviderResourceCoordinator,
    shared_provider_resources,
)
from neva.utils.safety import CircuitBreaker, RateLimiter

# Provider SDKs are imported lazily. Route them through a module-level alias so
# tests can stub the resolver: patching ``importlib.import_module`` directly
# would mutate the stdlib module for the whole process (and break pytest's own
# monkeypatch resolution on pytest 9+).
_import_module = importlib.import_module

if TYPE_CHECKING:  # pragma: no cover - import used only for typing.
    from neva.tools.guard import ToolGuard

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-1.5-flash",
    "google": "gemini-1.5-flash",
    "google-gemini": "gemini-1.5-flash",
    "xai": "grok-4.5",
    "grok": "grok-4.5",
}

_GEMINI_PROVIDERS = {"gemini", "google", "google-gemini"}

_CHAT_COMPLETION_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "grok": "https://api.x.ai/v1/chat/completions",
}

# ~6k tokens at 4 chars/token; leaves headroom for the current prompt and completion.
_DEFAULT_MAX_CONTEXT_CHARS = 24000
_RAW_PROMPT_MODE: ContextVar[bool] = ContextVar("neva_gpt_raw_prompt_mode", default=False)


def _chat_completions_url(api_base: Optional[str], default_url: str) -> str:
    """Resolve a Chat Completions URL from an optional base or full endpoint.

    ``api_base`` is treated as a full endpoint only when it already ends with
    ``chat/completions``. Every other value is a base URL and gets that path
    appended, matching the previous OpenAI SDK behaviour for custom gateways.
    """

    if not api_base:
        return default_url
    trimmed = api_base.rstrip("/")
    if trimmed.endswith("chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def _is_retryable_error(exc: Exception) -> bool:
    """Retry only transport failures, throttling, conflicts and server errors."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        return status in {408, 409, 429} or 500 <= status < 600
    return isinstance(
        exc, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)
    )


def _extract_chat_content(data: Dict[str, Any]) -> Optional[str]:
    """Parse OpenAI-compatible Chat Completions JSON into a text reply."""

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message") or choice.get("delta") or {}
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("value")
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                content = "".join(parts)
            if content:
                return str(content).strip()
            text = choice.get("text")
            if text:
                return str(text).strip()
    for key in ("content", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _provider_usage_from_gemini(response: Any) -> Optional[Dict[str, Any]]:
    """Map Gemini ``usage_metadata`` onto the Chat Completions usage shape."""

    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_token_count") or usage.get("prompt_tokens")
        completion_tokens = usage.get("candidates_token_count") or usage.get("completion_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_token_count", None) or getattr(
            usage, "prompt_tokens", None
        )
        completion_tokens = getattr(usage, "candidates_token_count", None) or getattr(
            usage, "completion_tokens", None
        )
    if prompt_tokens is None and completion_tokens is None:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
    }


class GPTAgent(AIAgent):
    """Agent that communicates with a large language model provider."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "openai",
        name: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        memory: Optional[MemoryModule] = None,
        rate_limiter: Optional[RateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        cache: Optional[LLMCache] = None,
        token_tracker: Optional[TokenUsageTracker] = None,
        cost_tracker: Optional[CostTracker] = None,
        spend_budget: Optional[SpendBudget] = None,
        provider_spend_limit: Optional[float] = None,
        provider_scope: Optional[str] = None,
        provider_coordination_path: Optional[str] = None,
        provider_rate: Optional[int] = 60,
        provider_rate_period: float = 60.0,
        max_provider_concurrency: Optional[int] = 8,
        share_provider_resources: bool = True,
        billing_reconciler: Optional[Callable[[], float]] = None,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        response_time_tracker: Optional[ResponseTimeTracker] = None,
        max_output_tokens: int = 1024,
        api_base: Optional[str] = None,
        request_timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
        context_budget: Optional[ModelContextBudget] = None,
        tool_guard: Optional["ToolGuard"] = None,
    ) -> None:
        resolved_cache = cache or LLMCache(max_size=256)
        super().__init__(
            name=name,
            llm_backend=llm_backend,
            memory=memory,
            cache=resolved_cache,
            response_time_tracker=response_time_tracker,
            tool_guard=tool_guard,
        )
        if max_context_chars <= 0:
            raise ConfigurationError("max_context_chars must be positive")
        selected_model = model or _DEFAULT_MODELS.get(provider.lower(), "gpt-4o-mini")
        if context_budget is not None:
            if not isinstance(context_budget, ModelContextBudget):
                raise ConfigurationError("context_budget must be a ModelContextBudget")
            if (
                context_budget.provider != provider.lower()
                or context_budget.model != selected_model
            ):
                raise ConfigurationError(
                    "context budget provider/model must match the selected provider/model"
                )
            if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
                raise ConfigurationError("max_output_tokens must be a positive integer")
            if max_output_tokens <= 0 or max_output_tokens >= context_budget.max_tokens:
                raise ConfigurationError(
                    "max_output_tokens must be positive and below context budget"
                )
        if spend_budget is not None:
            if llm_backend is not None:
                raise ConfigurationError(
                    "spend_budget requires the built-in provider backend; "
                    "remove llm_backend or the spend budget"
                )
            if not isinstance(spend_budget, SpendBudget):
                raise ConfigurationError("spend_budget must be a SpendBudget instance")
        if provider_spend_limit is not None and spend_budget is not None:
            raise ConfigurationError(
                "configure either spend_budget or provider_spend_limit, not both"
            )
        if provider_spend_limit is not None and llm_backend is not None:
            raise ConfigurationError("provider_spend_limit requires the built-in provider backend")
        if billing_reconciler is not None and not callable(billing_reconciler):
            raise ConfigurationError("billing_reconciler must be callable")
        if billing_reconciler is not None and spend_budget is None and provider_spend_limit is None:
            raise ConfigurationError(
                "billing_reconciler requires spend_budget or provider_spend_limit"
            )
        if not share_provider_resources and (
            provider_spend_limit is not None
            or provider_coordination_path is not None
            or provider_scope is not None
        ):
            raise ConfigurationError(
                "provider scope/spend/process coordination requires share_provider_resources=True"
            )
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model or _DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.api_base = api_base
        self._provider_resources: Optional[ProviderResourceCoordinator] = None
        if llm_backend is None and share_provider_resources:
            self._provider_resources = shared_provider_resources(
                provider=self.provider,
                api_key=api_key,
                api_base=api_base,
                provider_scope=provider_scope,
                rate=None if rate_limiter is not None else provider_rate,
                per=provider_rate_period,
                max_concurrency=max_provider_concurrency,
                max_cost=provider_spend_limit,
                state_path=provider_coordination_path,
            )
        self._rate_limiter = rate_limiter
        if llm_backend is None and not share_provider_resources and self._rate_limiter is None:
            self._rate_limiter = RateLimiter(rate=60, per=60.0)
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._cache = resolved_cache
        self._token_tracker = token_tracker or TokenUsageTracker()
        self._cost_tracker = cost_tracker or CostTracker()
        self._spend_budget = spend_budget
        self._billing_reconciler = billing_reconciler
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._logger = logging.getLogger(self.__class__.__name__)
        self._max_output_tokens = max_output_tokens
        self._request_timeout = request_timeout
        self._extra_headers = dict(extra_headers or {})
        self._max_context_chars = max_context_chars
        self._context_budget = context_budget
        self._last_provider_usage: Optional[Dict[str, Any]] = None

    def _call_cost(self, *, prompt_tokens: int, response_tokens: int) -> float:
        cost = self._cost_tracker.cost_for(
            self.model,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
        )
        if cost is None:
            raise ConfigurationError(
                f"Spend enforcement is enabled but model '{self.model}' has no "
                "pricing entry; add pricing or disable the spend limit."
            )
        if not math.isfinite(cost) or cost < 0:
            raise ConfigurationError(
                "Spend enforcement is enabled but model pricing produced a "
                "non-finite or negative cost."
            )
        return float(cost)

    def _reservation_cost(self, prompt: str) -> float:
        if self._spend_budget is None and (
            self._provider_resources is None or self._provider_resources.max_cost is None
        ):
            return 0.0
        if self._context_budget is not None:
            prompt_tokens = self._count_request_tokens(self._history_window(prompt), prompt)
        else:
            prompt_tokens = estimate_token_count(self._request_text(prompt))
        return self._call_cost(
            prompt_tokens=prompt_tokens,
            response_tokens=self._max_output_tokens,
        )

    def _reconcile_billing(self) -> None:
        if self._billing_reconciler is None:
            return
        actual = self._billing_reconciler()
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or actual < 0
        ):
            raise ConfigurationError(
                "billing_reconciler must return a finite non-negative spend total"
            )
        if self._spend_budget is not None:
            self._spend_budget.reconcile(float(actual))
        elif self._provider_resources is not None:
            self._provider_resources.reconcile_spend(float(actual))

    def _spend_preflight(self, prompt: str) -> float:
        """Reconcile billing and calculate the worst-case reservation for a call."""

        try:
            self._reconcile_billing()
            reserve_cost = self._reservation_cost(prompt)
            if self._spend_budget is not None:
                self._spend_budget.check(reserve_cost)
            return reserve_cost
        except (SpendBudgetExceededError, ConfigurationError):
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_rejected()
            raise

    @staticmethod
    def _wait_retry(
        delay: float,
        cancel_event: Optional[threading.Event],
    ) -> None:
        if cancel_event is None:
            sleep(delay)
            return
        if cancel_event.wait(delay):
            raise RateLimiterCancelledError("LLM retry wait cancelled")

    def _release_attempt_resources(
        self,
        permit: Optional[ProviderPermit],
        reservation: Optional[SpendReservation],
        *,
        actual_cost: Optional[float],
    ) -> None:
        error: Optional[Exception] = None
        if reservation is not None and self._spend_budget is not None:
            try:
                if actual_cost is None:
                    self._spend_budget.release(reservation)
                else:
                    self._spend_budget.settle(reservation, actual_cost)
            except Exception as exc:
                error = exc
        if permit is not None and self._provider_resources is not None:
            try:
                self._provider_resources.release(
                    permit,
                    actual_cost=(
                        actual_cost if self._provider_resources.max_cost is not None else None
                    ),
                )
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def _default_backend(
        self,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> LLMBackend:
        if not self.api_key:
            raise ConfigurationError(
                "No API key configured for GPTAgent. Provide `llm_backend` or set `api_key`."
            )

        def _call_model(prompt: str) -> str:
            cached = self._cache_lookup(prompt)
            if cached is not None:
                return cached

            attempt = 0
            last_error: Optional[Exception] = None
            while attempt <= self._max_retries:
                attempt += 1
                permit: Optional[ProviderPermit] = None
                reservation: Optional[SpendReservation] = None
                resources_released = False
                provider_succeeded = False
                actual_cost: Optional[float] = None
                try:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RateLimiterCancelledError("LLM call cancelled")
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.allow()

                    reserve_cost = self._spend_preflight(prompt)
                    if self._rate_limiter is not None:
                        self._rate_limiter.acquire(cancel_event=cancel_event)
                    if self._provider_resources is not None:
                        permit = self._provider_resources.acquire(
                            reserve_cost=(
                                reserve_cost
                                if self._provider_resources.max_cost is not None
                                else 0.0
                            ),
                            cancel_event=cancel_event,
                        )
                    if self._spend_budget is not None:
                        reservation = self._spend_budget.reserve(reserve_cost)

                    if cancel_event is not None and cancel_event.is_set():
                        raise RateLimiterCancelledError("LLM call cancelled")

                    start = perf_counter()
                    with self._response_time_tracker.track():
                        self._last_provider_usage = None
                        content = self._invoke_provider(prompt)
                    provider_succeeded = True
                    duration = perf_counter() - start

                    prompt_tokens = response_tokens = total_tokens = 0
                    if self._token_tracker is not None:
                        prompt_tokens, response_tokens = self._token_tracker.record(
                            self._request_text(prompt),
                            content,
                            usage=self._last_provider_usage,
                        )
                        total_tokens = prompt_tokens + response_tokens
                    if self._cost_tracker is not None and total_tokens:
                        self._cost_tracker.add_usage(
                            self.model,
                            total_tokens,
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                        )
                    if self._spend_budget is not None or (
                        self._provider_resources is not None
                        and self._provider_resources.max_cost is not None
                    ):
                        actual_cost = self._call_cost(
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                        )
                    try:
                        self._release_attempt_resources(
                            permit,
                            reservation,
                            actual_cost=actual_cost,
                        )
                    finally:
                        resources_released = True

                    self._cache_store(prompt, content)
                    self._logger.debug(
                        "llm_call",
                        extra={
                            "model": self.model,
                            "duration": duration,
                            "prompt_tokens": prompt_tokens,
                            "response_tokens": response_tokens,
                        },
                    )
                    telemetry = get_telemetry()
                    if telemetry is not None:
                        try:
                            conversation_id = getattr(
                                self.environment, "conversation_id", f"agent-{self.id}"
                            )
                            telemetry.record_llm_api_call(
                                conversation_id=conversation_id,
                                agent_name=self.name,
                                prompt=prompt,
                                completion=content,
                                provider=self.provider,
                                model=self.model,
                                latency=duration,
                                prompt_tokens=prompt_tokens or None,
                                completion_tokens=response_tokens or None,
                                total_tokens=total_tokens or None,
                                metadata={"attempt": attempt, "cache_hit": False},
                                conversation_state=self.conversation_state,
                            )
                        except Exception:
                            self._logger.debug(
                                "Failed to emit telemetry for LLM call",
                                exc_info=True,
                            )
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_success()
                    if cancel_event is not None and cancel_event.is_set():
                        raise RateLimiterCancelledError(
                            "LLM call cancelled after provider completion"
                        )
                    return content
                except Exception as exc:
                    last_error = exc
                    if not resources_released:
                        try:
                            self._release_attempt_resources(
                                permit,
                                reservation,
                                actual_cost=(actual_cost if provider_succeeded else None),
                            )
                        except Exception as settlement_error:
                            if provider_succeeded:
                                exc = settlement_error
                                last_error = exc

                    if isinstance(exc, RateLimiterCancelledError):
                        if self._circuit_breaker is not None:
                            if provider_succeeded:
                                self._circuit_breaker.record_success()
                            else:
                                self._circuit_breaker.record_rejected()
                        raise
                    if isinstance(exc, (CircuitOpenError, SpendBudgetExceededError)):
                        if self._circuit_breaker is not None:
                            if provider_succeeded:
                                self._circuit_breaker.record_success()
                            elif isinstance(exc, SpendBudgetExceededError):
                                self._circuit_breaker.record_rejected()
                        raise
                    if isinstance(exc, ConfigurationError):
                        if self._circuit_breaker is not None:
                            self._circuit_breaker.record_rejected()
                        raise
                    if _is_retryable_error(exc):
                        if self._circuit_breaker is not None:
                            self._circuit_breaker.record_failure()
                        if attempt > self._max_retries:
                            raise BackendError("LLM call failed") from exc
                        if self._circuit_breaker is not None:
                            self._circuit_breaker.allow()
                        sleep_time = min(30.0, self._retry_backoff**attempt)
                        self._logger.warning(
                            "Retrying LLM call due to error",
                            extra={"error": str(exc)},
                        )
                        self._wait_retry(sleep_time, cancel_event)
                        continue
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_rejected()
                    raise BackendError("LLM call failed") from exc

            if last_error is not None:
                raise BackendError("LLM call failed") from last_error
            raise BackendError("LLM call failed")

        return _call_model

    # ------------------------------------------------------------------
    # Provider specific implementations
    # ------------------------------------------------------------------

    def _invoke_provider(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._invoke_openai(prompt)
        if self.provider == "anthropic":
            return self._invoke_anthropic(prompt)
        if self.provider in _GEMINI_PROVIDERS:
            return self._invoke_gemini(prompt)
        if self.provider in {"xai", "grok"}:
            return self._invoke_grok(prompt)
        raise ConfigurationError(f"Unsupported provider '{self.provider}'.")

    def _invoke_chat_completions(self, prompt: str, *, default_url: str, empty_error: str) -> str:
        url = _chat_completions_url(self.api_base, default_url)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        payload = {
            "model": self.model,
            "messages": self._chat_messages(prompt),
            "max_tokens": self._max_output_tokens,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self._request_timeout)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendError("Provider returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise BackendError("Provider returned a malformed JSON response.")
        usage = data.get("usage")
        if isinstance(usage, dict):
            self._last_provider_usage = usage
        content = _extract_chat_content(data)
        if not content:
            raise BackendError(empty_error)
        return content

    def _invoke_openai(self, prompt: str) -> str:
        return self._invoke_chat_completions(
            prompt,
            default_url=_CHAT_COMPLETION_URLS["openai"],
            empty_error="Empty response from OpenAI provider.",
        )

    def _invoke_anthropic(self, prompt: str) -> str:
        try:
            anthropic = _import_module("anthropic")
        except ImportError as exc:  # pragma: no cover - import guard
            raise ConfigurationError(
                "Anthropic provider requires the 'anthropic' package to be installed."
            ) from exc

        client_cls = getattr(anthropic, "Anthropic", None)
        if client_cls is None:  # pragma: no cover - defensive guard
            raise ConfigurationError("Invalid anthropic client; update the 'anthropic' package.")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.api_base is not None:
            client_kwargs["base_url"] = self.api_base
        client = client_cls(**client_kwargs)
        request: Dict[str, object] = {
            "model": self.model,
            "messages": self._chat_messages(prompt),
            "max_tokens": self._max_output_tokens,
        }
        request["timeout"] = self._request_timeout
        response = client.messages.create(**request)
        content_blocks = getattr(response, "content", [])
        if not content_blocks:
            raise BackendError("Anthropic provider returned no content.")
        parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        content = "".join(parts).strip()
        if not content:
            raise BackendError("Anthropic provider returned empty content.")
        return content

    def _invoke_gemini(self, prompt: str) -> str:
        try:
            generative_ai = _import_module("google.generativeai")
        except ImportError as exc:  # pragma: no cover - import guard
            raise ConfigurationError(
                "Gemini provider requires the 'google-generativeai' package to be installed."
            ) from exc

        generative_ai.configure(api_key=self.api_key)
        model = generative_ai.GenerativeModel(self.model)
        request_text = self._prompt_with_history(prompt)
        response = model.generate_content(
            request_text,
            generation_config={"max_output_tokens": self._max_output_tokens},
            request_options={"timeout": self._request_timeout, "retry": None},
        )
        self._last_provider_usage = _provider_usage_from_gemini(response)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    texts = [getattr(part, "text", "") for part in content.parts]
                    joined = "".join(texts).strip()
                    if joined:
                        return joined
        raise BackendError("Gemini provider returned empty content.")

    def _invoke_grok(self, prompt: str) -> str:
        return self._invoke_chat_completions(
            prompt,
            default_url=_CHAT_COMPLETION_URLS["grok"],
            empty_error="Grok provider returned empty content.",
        )

    def _stream_chat_completions(self, prompt: str, usage_state: Dict[str, Any]) -> Iterator[str]:
        """Parse bounded SSE frames, closing the socket on every exit path."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._chat_messages(prompt),
            "max_tokens": self._max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        response = requests.post(
            _chat_completions_url(self.api_base, _CHAT_COMPLETION_URLS[self.provider]),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self._extra_headers,
            },
            json=payload,
            timeout=self._request_timeout,
            stream=True,
        )
        try:
            response.raise_for_status()
            buffer = bytearray()
            data_lines: List[str] = []
            event_bytes = 0
            done = False
            # read1 returns available bytes without waiting for the requested
            # size; iter_content(None) can wait for EOF on non-chunked SSE.
            for chunk in iter(lambda: response.raw.read1(65536, decode_content=True), b""):
                segments = chunk.split(b"\n")
                for index, segment in enumerate(segments):
                    buffer.extend(segment)
                    if len(buffer) > 1_048_576:
                        raise BackendError("Provider SSE frame exceeds 1 MiB")
                    if index == len(segments) - 1:
                        continue
                    line = bytes(buffer)
                    buffer.clear()
                    line = line.rstrip(b"\r")
                    if not line:
                        if not data_lines:
                            continue
                        event_payload = "\n".join(data_lines)
                        data_lines.clear()
                        event_bytes = 0
                        if event_payload == "[DONE]":
                            done = True
                            break
                        try:
                            event = json.loads(event_payload)
                        except (ValueError, UnicodeError) as exc:
                            raise BackendError("Provider returned malformed SSE JSON") from exc
                        if not isinstance(event, dict):
                            raise BackendError("Provider returned malformed SSE event")
                        if "error" in event:
                            raise BackendError("Provider returned a stream error")
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            usage_state["value"] = {
                                "prompt_tokens": usage.get(
                                    "prompt_tokens", usage.get("input_tokens")
                                ),
                                "completion_tokens": usage.get(
                                    "completion_tokens", usage.get("output_tokens")
                                ),
                            }
                        choices = event.get("choices", [])
                        if not isinstance(choices, list):
                            raise BackendError("Provider returned malformed stream choices")
                        if choices:
                            choice = choices[0]
                            if not isinstance(choice, dict) or not isinstance(
                                choice.get("delta"), dict
                            ):
                                raise BackendError("Provider returned malformed stream delta")
                            content = choice["delta"].get("content")
                            if content is not None:
                                if not isinstance(content, str):
                                    raise BackendError("Provider returned non-text stream delta")
                                if content:
                                    yield content
                    elif line.startswith(b"data:"):
                        event_bytes += len(line)
                        if event_bytes > 1_048_576:
                            raise BackendError("Provider SSE event exceeds 1 MiB")
                        try:
                            data_lines.append(line[5:].lstrip(b" ").decode("utf-8"))
                        except UnicodeError as exc:
                            raise BackendError("Provider returned invalid UTF-8 SSE data") from exc
                if done:
                    break
            if not done:
                raise requests.ConnectionError("Provider stream ended without a completion marker")
        finally:
            response.close()

    def _stream_provider(self, prompt: str, usage_state: Dict[str, Any]) -> Iterator[str]:
        if self.provider in _CHAT_COMPLETION_URLS:
            yield from self._stream_chat_completions(prompt, usage_state)
        elif self.provider == "anthropic":
            try:
                anthropic = _import_module("anthropic")
            except ImportError as exc:
                raise ConfigurationError(
                    "Anthropic streaming requires the 'anthropic' package"
                ) from exc
            kwargs: Dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
            if self.api_base is not None:
                kwargs["base_url"] = self.api_base
            client = anthropic.Anthropic(**kwargs)
            with client.messages.stream(
                model=self.model,
                messages=self._chat_messages(prompt),
                max_tokens=self._max_output_tokens,
                timeout=self._request_timeout,
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
                usage = getattr(final, "usage", None)
                if usage is not None:
                    usage_state["value"] = {
                        "prompt_tokens": getattr(usage, "input_tokens", 0),
                        "completion_tokens": getattr(usage, "output_tokens", 0),
                    }
        elif self.provider in _GEMINI_PROVIDERS:
            try:
                generative_ai = _import_module("google.generativeai")
            except ImportError as exc:
                raise ConfigurationError("Gemini streaming requires 'google-generativeai'") from exc
            generative_ai.configure(api_key=self.api_key)
            model = generative_ai.GenerativeModel(self.model)
            response = model.generate_content(
                self._prompt_with_history(prompt),
                generation_config={"max_output_tokens": self._max_output_tokens},
                request_options={"timeout": self._request_timeout, "retry": None},
                stream=True,
            )
            try:
                for chunk in response:
                    usage = _provider_usage_from_gemini(chunk)
                    if usage:
                        usage_state["value"] = usage
                    try:
                        content = chunk.text
                    except ValueError:
                        content = None
                    if content:
                        yield content
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        else:
            raise ConfigurationError(f"Unsupported provider '{self.provider}'.")

    def _turn_text(self, turn: Any) -> str:
        if turn.speaker == self.name or turn.speaker in {"user", "system"}:
            return str(turn.message)
        return f"{turn.speaker}: {turn.message}"

    def _format_request(self, turns: List[Any], prompt: str) -> str:
        """Serialize ``turns`` plus ``prompt`` the way this provider will send them."""

        if self.provider in _GEMINI_PROVIDERS:
            if not turns:
                return prompt
            lines = [f"{turn.speaker}: {turn.message}" for turn in turns]
            return "Conversation so far:\n" + "\n".join(lines) + "\n\n" + prompt
        contents = [self._turn_text(turn) for turn in turns]
        contents.append(prompt)
        return "\n".join(contents)

    def _messages_from_turns(self, turns: List[Any], prompt: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for turn in turns:
            content = self._turn_text(turn)
            role = "assistant" if turn.speaker == self.name else "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request_for_tokens(self, turns: List[Any], prompt: str) -> RequestContent:
        if self.provider in _GEMINI_PROVIDERS:
            return self._format_request(turns, prompt)
        return self._messages_from_turns(turns, prompt)

    def _count_request_tokens(self, turns: List[Any], prompt: str) -> int:
        budget = self._context_budget
        if budget is None:
            raise ConfigurationError("context token counting requires a context budget")
        if budget.provider != self.provider or budget.model != self.model:
            raise ConfigurationError("context budget provider/model no longer matches the agent")
        return budget.count(self._request_for_tokens(turns, prompt))

    def _history_window(self, prompt: str) -> List[Any]:
        """Return a recent contiguous suffix fitting both configured budgets.

        The character budget uses provider-specific serialized text; the
        opt-in token budget uses the provider-shaped request. ConversationState
        itself is unbounded. Chat Completions
        and Anthropic drop a leading assistant turn so the request still starts
        with a user message. Raw model-generation mode intentionally excludes
        history because its caller already composed the complete bounded prompt.
        """

        if len(self._format_request([], prompt)) > self._max_context_chars:
            raise ConfigurationError(
                f"Current prompt is {len(prompt)} characters, which exceeds "
                f"max_context_chars={self._max_context_chars}"
            )
        budget = self._context_budget
        if budget is not None:
            current_tokens = self._count_request_tokens([], prompt)
            if current_tokens + self._max_output_tokens > budget.max_tokens:
                raise ConfigurationError(
                    f"Current request needs {current_tokens} input tokens + "
                    f"{self._max_output_tokens} reserved output tokens, exceeding "
                    f"context budget of {budget.max_tokens} for {self.model}"
                )
        if _RAW_PROMPT_MODE.get():
            return []

        turns = self.conversation_state.turns
        if budget is None:
            # Preserve the legacy character-only window selection.
            window: List[Any] = []
            for turn in reversed(turns):
                candidate = [turn, *window]
                if len(self._format_request(candidate, prompt)) > self._max_context_chars:
                    break
                window = candidate
        else:
            # The caller's counter should be nondecreasing when older turns
            # are prepended. Count only logarithmically many complete requests
            # rather than re-encoding each growing suffix (quadratic work).
            low, high = 0, len(turns)
            while low < high:
                middle = (low + high + 1) // 2
                candidate = turns[-middle:]
                if len(self._format_request(candidate, prompt)) > self._max_context_chars or (
                    self._count_request_tokens(candidate, prompt) + self._max_output_tokens
                    > budget.max_tokens
                ):
                    high = middle - 1
                else:
                    low = middle
            window = turns[-low:] if low else []
        if self.provider not in _GEMINI_PROVIDERS:
            while window and window[0].speaker == self.name:
                window.pop(0)
        return window

    def _chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build Chat Completions messages from recorded turns plus ``prompt``."""

        return self._messages_from_turns(self._history_window(prompt), prompt)

    def _prompt_with_history(self, prompt: str) -> str:
        """Flatten recorded turns in front of ``prompt`` for string-only APIs."""

        return self._format_request(self._history_window(prompt), prompt)

    def _request_text(self, prompt: str) -> str:
        """Text actually sent to the provider, used for token estimates."""

        return self._format_request(self._history_window(prompt), prompt)

    def _scoped_key(self, prompt: str) -> str:
        """Cache key bound to the provider/model configuration in effect."""

        history = [(turn.speaker, turn.message) for turn in self._history_window(prompt)]
        scope = {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "max_output_tokens": self._max_output_tokens,
            "max_context_chars": self._max_context_chars,
            "context_budget": (
                {
                    "max_tokens": self._context_budget.max_tokens,
                    "counter_id": self._context_budget.counter_id,
                    "provider": self._context_budget.provider,
                    "model": self._context_budget.model,
                }
                if self._context_budget is not None
                else None
            ),
            "history": history,
        }
        return json.dumps({"scope": scope, "prompt": prompt}, sort_keys=True)

    def _cache_lookup(self, prompt: str) -> Optional[str]:
        return super()._cache_lookup(self._scoped_key(prompt))

    def _cache_store(self, prompt: str, response: str) -> None:
        super()._cache_store(self._scoped_key(prompt), response)

    def replay_identity_resolver(self) -> Callable[[str], str]:
        """Capture the effective provider-request identity for replay validation."""

        if self.llm_backend is not None:
            return super().replay_identity_resolver()
        return self._scoped_key

    def replayable_backend(
        self,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> LLMBackend:
        """Return a cancellation-aware provider/custom model boundary."""

        backend = self.llm_backend or self._default_backend(cancel_event=cancel_event)
        return self._wrap_model_backend(backend)

    def generate_model_output(self, prompt: str) -> str:
        """Generate from an already composed prompt without adding history/context."""

        validated_prompt = self.prompt_validator.validate(prompt)
        token = _RAW_PROMPT_MODE.set(True)
        try:
            cached = self._cache_lookup(validated_prompt)
            if cached is not None:
                return cached
            response = self.replayable_backend()(validated_prompt)
            self._cache_store(validated_prompt, response)
            return response
        finally:
            _RAW_PROMPT_MODE.reset(token)

    def _respond_with_cancel(
        self,
        message: str,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        return self.respond(message, cancel_event=cancel_event)

    def respond(
        self,
        message: str,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Return a cached or freshly generated response for ``message``."""
        prompt = self.prepare_prompt(message)
        validated_prompt = self.prompt_validator.validate(prompt)
        cached = self._cache_lookup(validated_prompt)
        if cached is not None:
            return cached

        response = self.replayable_backend(cancel_event=cancel_event)(validated_prompt)
        self._cache_store(validated_prompt, response)
        return response

    def stream_response(
        self, message: str, *, max_queue_size: int = 8, max_response_chars: int = 1_000_000
    ) -> StreamSession:
        """Stream a reply with bounded queued deltas and a terminal completion event.

        Only completed responses enter conversation state and cache. An error after
        a delta raises ``StreamInterruptedError`` carrying the uncommitted text.
        The session supports synchronous and asynchronous iteration; call ``close``
        or ``aclose`` when abandoning it early.
        """

        if self.llm_backend is not None or self._model_backend_wrapper is not None:
            raise ConfigurationError(
                "Streaming requires a built-in provider without a model wrapper"
            )
        if not self.api_key:
            raise ConfigurationError("Streaming requires an API key")
        if (
            isinstance(max_response_chars, bool)
            or not isinstance(max_response_chars, int)
            or max_response_chars < 1
        ):
            raise ConfigurationError("max_response_chars must be a positive integer")
        validated_message = self.prompt_validator.validate(message)
        prompt = self.prompt_validator.validate(self.prepare_prompt(validated_message))
        cache_key = self._scoped_key(prompt)
        completed: Dict[str, Any] = {}

        def produce(emit: Callable[[str], None], cancel: threading.Event) -> StreamEvent:
            """Emit the cached or streamed completion text for this request."""
            cached = AIAgent._cache_lookup(self, cache_key)
            if cached is not None:
                if len(cached) > max_response_chars:
                    raise BackendError("Cached response exceeds max_response_chars")
                completed["cache_hit"] = True
                emit(cached)
                return StreamEvent("complete", cached, 0.0, 0.0)

            last_error: Optional[Exception] = None
            for attempt in range(1, self._max_retries + 2):
                permit: Optional[ProviderPermit] = None
                reservation: Optional[SpendReservation] = None
                released = False
                accounted = False
                actual_cost: Optional[float] = None
                parts: List[str] = []
                usage_state: Dict[str, Any] = {}
                chars = 0
                first: Optional[float] = None
                started = perf_counter()
                try:
                    if cancel.is_set():
                        raise RateLimiterCancelledError("LLM stream cancelled")
                    self._circuit_breaker.allow()
                    reserve_cost = self._spend_preflight(prompt)
                    if self._rate_limiter is not None:
                        self._rate_limiter.acquire(cancel_event=cancel)
                    if self._provider_resources is not None:
                        permit = self._provider_resources.acquire(
                            reserve_cost=(
                                reserve_cost
                                if self._provider_resources.max_cost is not None
                                else 0.0
                            ),
                            cancel_event=cancel,
                        )
                    if self._spend_budget is not None:
                        reservation = self._spend_budget.reserve(reserve_cost)
                    if cancel.is_set():
                        raise RateLimiterCancelledError("LLM stream cancelled")
                    started = perf_counter()
                    with self._response_time_tracker.track():
                        for delta in self._stream_provider(prompt, usage_state):
                            if cancel.is_set():
                                raise RateLimiterCancelledError("LLM stream cancelled")
                            if not isinstance(delta, str):
                                raise BackendError("Provider returned non-text stream delta")
                            if delta:
                                if chars + len(delta) > max_response_chars:
                                    raise BackendError("Provider stream exceeds max_response_chars")
                                emit(delta)  # bounded queue; blocks producer on slow consumer
                                parts.append(delta)
                                chars += len(delta)
                                if first is None:
                                    first = perf_counter() - started
                        if cancel.is_set():
                            raise RateLimiterCancelledError("LLM stream cancelled")
                    content = "".join(parts)
                    if not content.strip():
                        raise BackendError("Provider stream returned no text")
                    duration = perf_counter() - started
                    prompt_tokens, response_tokens = self._token_tracker.record(
                        self._request_text(prompt), content, usage=usage_state.get("value")
                    )
                    accounted = True
                    self._cost_tracker.add_usage(
                        self.model,
                        prompt_tokens + response_tokens,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                    )
                    actual_cost = (
                        self._call_cost(
                            prompt_tokens=prompt_tokens, response_tokens=response_tokens
                        )
                        if reservation is not None
                        or (
                            self._provider_resources is not None
                            and self._provider_resources.max_cost is not None
                        )
                        else None
                    )
                    try:
                        self._release_attempt_resources(
                            permit, reservation, actual_cost=actual_cost
                        )
                    finally:
                        released = True
                    self._circuit_breaker.record_success()
                    completed.update(
                        attempt=attempt,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                    )
                    return StreamEvent("complete", content, first, duration)
                except Exception as exc:
                    last_error = exc
                    if not released:
                        # Settle known usage/partial output, otherwise release
                        # the reservation as in the ordinary failure path.
                        if not accounted and (parts or usage_state.get("value") is not None):
                            try:
                                p, r = self._token_tracker.record(
                                    self._request_text(prompt),
                                    "".join(parts),
                                    usage=usage_state.get("value"),
                                )
                                self._cost_tracker.add_usage(
                                    self.model, p + r, prompt_tokens=p, response_tokens=r
                                )
                                if reservation is not None or (
                                    self._provider_resources is not None
                                    and self._provider_resources.max_cost is not None
                                ):
                                    actual_cost = self._call_cost(
                                        prompt_tokens=p, response_tokens=r
                                    )
                            except Exception:
                                self._logger.debug(
                                    "Failed to estimate partial stream cost", exc_info=True
                                )
                        self._release_attempt_resources(
                            permit, reservation, actual_cost=actual_cost
                        )
                    if isinstance(exc, RateLimiterCancelledError):
                        self._circuit_breaker.record_rejected()
                        raise
                    if parts:
                        self._circuit_breaker.record_failure()
                        telemetry = get_telemetry()
                        if telemetry is not None:
                            try:
                                telemetry.record_llm_api_call(
                                    conversation_id=getattr(
                                        self.environment, "conversation_id", f"agent-{self.id}"
                                    ),
                                    agent_name=self.name,
                                    prompt=prompt,
                                    completion="".join(parts),
                                    provider=self.provider,
                                    model=self.model,
                                    latency=perf_counter() - started,
                                    first_token_seconds=first,
                                    metadata={
                                        "attempt": attempt,
                                        "stream": True,
                                        "stream_status": "interrupted",
                                    },
                                )
                            except Exception:
                                self._logger.debug(
                                    "Failed to emit partial telemetry", exc_info=True
                                )
                        raise StreamInterruptedError(
                            "Provider stream failed after emitting text",
                            "".join(parts),
                            first,
                        ) from exc
                    if isinstance(
                        exc, (ConfigurationError, CircuitOpenError, SpendBudgetExceededError)
                    ):
                        self._circuit_breaker.record_rejected()
                        raise
                    if _is_retryable_error(exc):
                        self._circuit_breaker.record_failure()
                        if attempt <= self._max_retries:
                            self._wait_retry(min(30.0, self._retry_backoff**attempt), cancel)
                            continue
                    else:
                        self._circuit_breaker.record_rejected()
                    raise BackendError("LLM stream failed") from exc
            raise BackendError("LLM stream failed") from last_error

        def accept(event: StreamEvent) -> None:
            """Record the settled stream: conversation turns, cache, and usage."""
            self._remember("system", validated_message)
            self._remember(self.name, event.text)
            if not completed.get("cache_hit"):
                AIAgent._cache_store(self, cache_key, event.text)
            if completed.get("cache_hit"):
                return
            attempt = completed["attempt"]
            prompt_tokens = completed["prompt_tokens"]
            response_tokens = completed["response_tokens"]
            duration = event.completion_seconds
            first = event.first_token_seconds
            content = event.text
            telemetry = get_telemetry()
            if telemetry is not None:
                try:
                    conversation_id = getattr(
                        self.environment, "conversation_id", f"agent-{self.id}"
                    )
                    telemetry.record_llm_api_call(
                        conversation_id=conversation_id,
                        agent_name=self.name,
                        prompt=prompt,
                        completion=content,
                        provider=self.provider,
                        model=self.model,
                        latency=duration,
                        first_token_seconds=first,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=response_tokens,
                        metadata={
                            "attempt": attempt,
                            "stream": True,
                            "first_token_seconds": first,
                            "completion_seconds": duration,
                        },
                        conversation_state=self.conversation_state,
                    )
                    telemetry.record_agent_turn(
                        conversation_id=conversation_id,
                        agent_name=self.name,
                        prompt=validated_message,
                        response=content,
                        latency=duration,
                        model=self.model,
                        metadata={"stream": True, "first_token_seconds": first},
                        conversation_state=self.conversation_state,
                    )
                except Exception:
                    self._logger.debug("Failed to emit stream telemetry", exc_info=True)

        return StreamSession(produce, max_queue_size=max_queue_size, on_complete=accept)


__all__ = ["GPTAgent"]
