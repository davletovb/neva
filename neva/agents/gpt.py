"""Large language model agent with support for multiple providers."""

from __future__ import annotations

import importlib
import logging
from time import perf_counter, sleep
from typing import Dict, Optional

import openai
import requests

from neva.agents.base import AIAgent, LLMBackend
from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.exceptions import BackendError, ConfigurationError
from neva.utils.metrics import CostTracker, ResponseTimeTracker, TokenUsageTracker
from neva.utils.safety import RateLimiter
from neva.utils.telemetry import get_telemetry


class GPTAgent(AIAgent):
    """Agent that communicates with a large language model provider."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
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
        self.model = model
        self.provider = provider.lower()
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
                        content = self._invoke_provider(prompt)
                    duration = perf_counter() - start
                    prompt_tokens = response_tokens = total_tokens = 0
                    if self._token_tracker is not None:
                        prompt_tokens, response_tokens = self._token_tracker.record(prompt, content)
                        total_tokens = prompt_tokens + response_tokens
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
                    if attempt > self._max_retries:
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

    def _invoke_openai(self, prompt: str) -> str:
        openai.api_key = self.api_key
        if self.api_base is not None:
            openai.api_base = self.api_base
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_output_tokens,
            request_timeout=self._request_timeout,
        )
        content = response.choices[0].message["content"].strip()
        if not content:
            raise BackendError("Empty response from OpenAI provider.")
        return content

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

        client_kwargs = {"api_key": self.api_key}
        if self.api_base is not None:
            client_kwargs["base_url"] = self.api_base
        client = client_cls(**client_kwargs)
        request: Dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_output_tokens,
        }
        response = client.messages.create(**request)
        content_blocks = getattr(response, "content", [])
        if not content_blocks:
            raise BackendError("Anthropic provider returned no content.")
        parts = []
        for block in content_blocks:
            text = (
                getattr(block, "text", None) or block.get("text")
                if isinstance(block, dict)
                else None
            )
            if text:
                parts.append(text)
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
        response = model.generate_content(prompt)
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
        url = self.api_base or "https://api.x.ai/v1/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self._request_timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or data.get("messages")
        content: Optional[str] = None
        if isinstance(message, list) and message:
            message = message[-1]
        if isinstance(message, dict):
            content = message.get("content")
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
        if not content:
            content = data.get("content") or data.get("text")
        if not content:
            raise BackendError("Grok provider returned empty content.")
        return str(content).strip()

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
