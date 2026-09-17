"""Large language model agent with support for multiple providers."""

from __future__ import annotations

import importlib
import json
import logging
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional

import requests

from neva.agents.base import AIAgent, LLMBackend
from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.exceptions import BackendError, ConfigurationError
from neva.utils.metrics import CostTracker, ResponseTimeTracker, TokenUsageTracker
from neva.utils.safety import RateLimiter
from neva.utils.telemetry import get_telemetry

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-1.5-flash",
    "google": "gemini-1.5-flash",
    "google-gemini": "gemini-1.5-flash",
    "xai": "grok-4.5",
    "grok": "grok-4.5",
}

_CHAT_COMPLETION_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "grok": "https://api.x.ai/v1/chat/completions",
}


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
        cache: Optional[LLMCache] = None,
        token_tracker: Optional[TokenUsageTracker] = None,
        cost_tracker: Optional[CostTracker] = None,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        response_time_tracker: Optional[ResponseTimeTracker] = None,
        max_output_tokens: int = 1024,
        api_base: Optional[str] = None,
        request_timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        resolved_cache = cache or LLMCache(max_size=256)
        super().__init__(
            name=name,
            llm_backend=llm_backend,
            memory=memory,
            cache=resolved_cache,
            response_time_tracker=response_time_tracker,
        )
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model or _DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.api_base = api_base
        self._rate_limiter = rate_limiter or RateLimiter(rate=60, per=60.0)
        self._cache = resolved_cache
        self._token_tracker = token_tracker or TokenUsageTracker()
        self._cost_tracker = cost_tracker or CostTracker()
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._logger = logging.getLogger(self.__class__.__name__)
        self._max_output_tokens = max_output_tokens
        self._request_timeout = request_timeout
        self._extra_headers = dict(extra_headers or {})
        self._last_provider_usage: Optional[Dict[str, Any]] = None

    def _default_backend(self) -> LLMBackend:
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
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()
                start = perf_counter()
                try:
                    with self._response_time_tracker.track():
                        self._last_provider_usage = None
                        content = self._invoke_provider(prompt)
                    duration = perf_counter() - start
                    prompt_tokens = response_tokens = total_tokens = 0
                    if self._token_tracker is not None:
                        prompt_tokens, response_tokens = self._token_tracker.record(
                            prompt, content, usage=self._last_provider_usage
                        )
                        total_tokens = prompt_tokens + response_tokens
                    if self._cost_tracker is not None and total_tokens:
                        self._cost_tracker.add_usage(
                            self.model,
                            total_tokens,
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                        )
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
                        except Exception:  # pragma: no cover - telemetry must not break retries.
                            self._logger.debug(
                                "Failed to emit telemetry for LLM call", exc_info=True
                            )
                    return content
                except Exception as exc:  # pragma: no cover - network error path.
                    last_error = exc
                    if isinstance(exc, ConfigurationError):
                        raise
                    if not _is_retryable_error(exc) or attempt > self._max_retries:
                        raise BackendError("LLM call failed") from exc
                    sleep_time = min(30.0, self._retry_backoff**attempt)
                    self._logger.warning(
                        "Retrying LLM call due to error", extra={"error": str(exc)}
                    )
                    sleep(sleep_time)

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
        if self.provider in {"gemini", "google", "google-gemini"}:
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
        data = response.json()
        usage = data.get("usage") if isinstance(data, dict) else None
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
            anthropic = importlib.import_module("anthropic")
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
            generative_ai = importlib.import_module("google.generativeai")
        except ImportError as exc:  # pragma: no cover - import guard
            raise ConfigurationError(
                "Gemini provider requires the 'google-generativeai' package to be installed."
            ) from exc

        generative_ai.configure(api_key=self.api_key)
        model = generative_ai.GenerativeModel(self.model)
        response = model.generate_content(
            self._prompt_with_history(prompt),
            generation_config={"max_output_tokens": self._max_output_tokens},
            request_options={"timeout": self._request_timeout, "retry": None},
        )
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

    def _chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build Chat Completions messages from recorded turns plus ``prompt``."""

        messages: List[Dict[str, str]] = []
        for turn in self.conversation_state.turns:
            if turn.speaker == self.name:
                messages.append({"role": "assistant", "content": turn.message})
            elif turn.speaker in {"user", "system"}:
                messages.append({"role": "user", "content": turn.message})
            else:
                messages.append({"role": "user", "content": f"{turn.speaker}: {turn.message}"})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _prompt_with_history(self, prompt: str) -> str:
        """Flatten recorded turns in front of ``prompt`` for string-only APIs."""

        if not self.conversation_state.turns:
            return prompt
        lines = [f"{turn.speaker}: {turn.message}" for turn in self.conversation_state.turns]
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n" + prompt

    def _scoped_key(self, prompt: str) -> str:
        """Cache key bound to the provider/model configuration in effect."""

        history = [(turn.speaker, turn.message) for turn in self.conversation_state.turns]
        scope = {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "max_output_tokens": self._max_output_tokens,
            "history": history,
        }
        return json.dumps({"scope": scope, "prompt": prompt}, sort_keys=True)

    def _cache_lookup(self, prompt: str) -> Optional[str]:
        return super()._cache_lookup(self._scoped_key(prompt))

    def _cache_store(self, prompt: str, response: str) -> None:
        super()._cache_store(self._scoped_key(prompt), response)

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


__all__ = ["GPTAgent"]
