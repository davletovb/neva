"""Transformer-based agent implementation."""

from __future__ import annotations

from typing import Callable, Optional

from neva.agents.base import AIAgent, LLMBackend
from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.exceptions import BackendUnavailableError
from neva.utils.safety import PromptValidator

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


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
        prompt_validator: Optional[PromptValidator] = None,
    ) -> None:
        super().__init__(
            name=name,
            llm_backend=llm_backend,
            memory=memory,
            cache=cache,
            prompt_validator=prompt_validator,
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


__all__ = ["TransformerAgent"]
