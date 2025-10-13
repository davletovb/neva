"""OpenAI-compatible GPT agent implementation."""

from __future__ import annotations

import logging
from time import perf_counter, sleep
from typing import Optional

import openai

from neva.agents.base import AIAgent, LLMBackend
from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.exceptions import BackendError, ConfigurationError
from neva.utils.metrics import CostTracker, ResponseTimeTracker, TokenUsageTracker
from neva.utils.safety import RateLimiter


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
        response_time_tracker: Optional[ResponseTimeTracker] = None,
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


__all__ = ["GPTAgent"]
