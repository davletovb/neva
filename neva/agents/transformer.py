"""Transformer-based agent implementation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, cast

from neva.agents.base import AIAgent, LLMBackend
from neva.memory import MemoryModule
from neva.utils.caching import LLMCache
from neva.utils.exceptions import BackendUnavailableError
from neva.utils.safety import PromptValidator


class _TransformerModel(Protocol):
    """Protocol for transformer models used by :class:`TransformerAgent`."""

    def generate(self, **kwargs: Any) -> Sequence[Any]:
        """Generate tokens from the provided model inputs."""


class _TransformerTokenizer(Protocol):
    """Protocol describing the tokenizer surface used by the agent."""

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        """Tokenize ``text`` and return the model inputs."""

    def decode(self, token_ids: Any, *, skip_special_tokens: bool = ...) -> str:
        """Convert ``token_ids`` back into natural language text."""


ModelLoader = Callable[[str], _TransformerModel]
TokenizerLoader = Callable[[str], _TransformerTokenizer]


try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLM
    from transformers import AutoTokenizer as _AutoTokenizer
except Exception:  # pragma: no cover - handled at runtime
    _DEFAULT_MODEL_LOADER: Optional[ModelLoader] = None
    _DEFAULT_TOKENIZER_LOADER: Optional[TokenizerLoader] = None
else:  # pragma: no cover - exercised when transformers is installed
    _DEFAULT_MODEL_LOADER = cast(ModelLoader, _AutoModelForSeq2SeqLM.from_pretrained)
    _DEFAULT_TOKENIZER_LOADER = cast(TokenizerLoader, _AutoTokenizer.from_pretrained)


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
        model_loader: Optional[ModelLoader] = None,
        tokenizer_loader: Optional[TokenizerLoader] = None,
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
        self._model_loader: Optional[ModelLoader] = model_loader
        self._tokenizer_loader: Optional[TokenizerLoader] = tokenizer_loader
        self._model: Optional[_TransformerModel] = None
        self._tokenizer: Optional[_TransformerTokenizer] = None

    def _load_transformer(self) -> None:
        if self.llm_backend is not None:
            return
        if self._model is not None and self._tokenizer is not None:
            return

        model_loader = self._model_loader
        tokenizer_loader = self._tokenizer_loader

        if model_loader is None or tokenizer_loader is None:
            if _DEFAULT_MODEL_LOADER is None or _DEFAULT_TOKENIZER_LOADER is None:
                raise BackendUnavailableError(
                    "Transformers library is not available. Provide an ``llm_backend`` "
                    "or install `transformers` to use TransformerAgent."
                )

            model_loader = _DEFAULT_MODEL_LOADER
            tokenizer_loader = _DEFAULT_TOKENIZER_LOADER

        if model_loader is None or tokenizer_loader is None:
            raise BackendUnavailableError(
                "Transformer loader functions are unavailable despite transformers being installed."
            )

        self._model = model_loader(self.model_name)
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
        if self._model is None or self._tokenizer is None:
            raise BackendUnavailableError(
                "Transformer model components failed to load; provide an ``llm_backend`` "
                "or ensure the transformers dependencies are available."
            )

        inputs = self._tokenizer(
            validated_prompt, return_tensors="pt", truncation=True, padding=True
        )
        output_tokens = self._model.generate(**inputs, max_length=200)
        decoded = self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        self._cache_store(validated_prompt, decoded)
        return decoded


__all__ = ["TransformerAgent"]
