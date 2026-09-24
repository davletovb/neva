"""Opt-in, model-bound context envelopes for complete provider requests."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Union

from .exceptions import ConfigurationError

RequestContent = Union[str, Sequence[Mapping[str, str]]]
RequestTokenCounter = Callable[[RequestContent], int]


@dataclass(frozen=True)
class ModelContextBudget:
    """Limit input plus reserved output tokens for one provider/model.

    ``count_request_tokens`` receives the actual provider-shaped input: a
    sequence of role/content messages for chat APIs, or the flattened request
    string for Gemini. It must include any message framing, priming, and
    provider-specific overhead. The caller supplies the model's context size
    and an appropriate counter; Neva cannot verify provider-side tokenization.
    ``counter_id`` versions the counter for cache/replay identity.
    """

    provider: str
    model: str
    max_tokens: int
    count_request_tokens: RequestTokenCounter
    counter_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ConfigurationError("context budget provider must be nonempty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError("context budget model must be nonempty")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ConfigurationError("context budget max_tokens must be a positive integer")
        if self.max_tokens <= 0:
            raise ConfigurationError("context budget max_tokens must be a positive integer")
        if not callable(self.count_request_tokens):
            raise ConfigurationError("context budget count_request_tokens must be callable")
        if not isinstance(self.counter_id, str) or not self.counter_id.strip():
            raise ConfigurationError("context budget counter_id must be nonempty")

    def count(self, request: RequestContent) -> int:
        """Reject broken counters instead of silently undercounting."""

        try:
            count = self.count_request_tokens(request)
        except Exception as exc:
            raise ConfigurationError("context budget token counter failed") from exc
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ConfigurationError("context budget token counter must return a positive integer")
        return count


def openai_chat_counter(model: str) -> RequestTokenCounter:
    """Estimate text-only OpenAI chat tokens using the model's tiktoken encoding.

    Uses the message-framing estimate in OpenAI's token-counting cookbook for
    the listed models. It is not a provider context guarantee: aliases and
    server-side accounting can change. Install ``tiktoken`` separately.
    Other models/providers must supply their own request counter.
    """

    if model not in {"gpt-4o-mini", "gpt-4o-mini-2024-07-18"}:
        raise ConfigurationError("openai_chat_counter does not support this model")
    try:
        tiktoken = importlib.import_module("tiktoken")
        encoding = tiktoken.encoding_for_model(model)
    except (ImportError, KeyError) as exc:
        raise ConfigurationError(
            "OpenAI context counting requires tiktoken for this model"
        ) from exc

    def count(request: RequestContent) -> int:
        if isinstance(request, str):
            raise ConfigurationError("OpenAI chat counter requires role/content messages")
        return 3 + sum(
            3 + len(encoding.encode(message["role"])) + len(encoding.encode(message["content"]))
            for message in request
        )

    return count
