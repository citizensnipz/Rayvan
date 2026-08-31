from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable


DEFAULT_TOKENIZER_IDENTIFIER = "gpt2"


@runtime_checkable
class TextTokenizer(Protocol):
    @property
    def identifier(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...

    def to_config(self) -> dict[str, Any]: ...


class HuggingFaceTokenizer:
    """Thin adapter around a standard fast Hugging Face tokenizer."""

    def __init__(self, backend: Any, identifier: str) -> None:
        if backend.eos_token_id is None:
            raise ValueError(f"tokenizer {identifier!r} does not define an EOS token")
        self._backend = backend
        self._identifier = identifier

    @classmethod
    def from_pretrained(
        cls, identifier: str = DEFAULT_TOKENIZER_IDENTIFIER
    ) -> HuggingFaceTokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                'subword tokenization requires: python -m pip install -e ".[data]"'
            ) from error
        backend = AutoTokenizer.from_pretrained(identifier, use_fast=True)
        if not backend.is_fast:
            raise RuntimeError(f"tokenizer {identifier!r} has no fast implementation")
        return cls(backend, identifier)

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def vocab_size(self) -> int:
        return len(self._backend)

    @property
    def eos_token_id(self) -> int:
        return int(self._backend.eos_token_id)

    def encode(self, text: str) -> list[int]:
        raw_backend = getattr(self._backend, "backend_tokenizer", None)
        if raw_backend is not None:
            encoding = raw_backend.encode(text, add_special_tokens=False)
            return list(encoding.ids)
        return list(self._backend.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Iterable[int]) -> str:
        return str(
            self._backend.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    def to_config(self) -> dict[str, Any]:
        return {"kind": "huggingface", "identifier": self.identifier}


def tokenizer_from_config(config: dict[str, Any]) -> TextTokenizer:
    kind = config.get("kind")
    if kind == "huggingface":
        return HuggingFaceTokenizer.from_pretrained(str(config["identifier"]))
    if kind == "character":
        from .data import CharacterTokenizer

        return CharacterTokenizer.from_config(config)
    raise ValueError(f"unknown tokenizer configuration kind: {kind!r}")
