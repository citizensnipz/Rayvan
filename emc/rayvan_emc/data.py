from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import Tensor

from .tokenization import (
    DEFAULT_TOKENIZER_IDENTIFIER,
    HuggingFaceTokenizer,
    TextTokenizer,
)


TINY_OVERFIT_TEXTS = (
    "the moon is bright tonight.",
    "the small fox runs through the garden.",
    "a quiet river flows under the old bridge.",
    "the red bird sings in the morning.",
    "we watch the stars from the hill.",
    "the little boat returns before sunset.",
)


class CharacterTokenizer:
    unknown_symbol = "<unk>"

    def __init__(self, characters: Iterable[str]) -> None:
        symbols = (self.unknown_symbol, *sorted(set(characters)))
        self._set_symbols(symbols)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> CharacterTokenizer:
        return cls(character for text in texts for character in text)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CharacterTokenizer:
        tokenizer = cls.__new__(cls)
        tokenizer._set_symbols(tuple(str(value) for value in config["symbols"]))
        return tokenizer

    def _set_symbols(self, symbols: tuple[str, ...]) -> None:
        self.symbols = symbols
        self._token_to_id = {
            symbol: index for index, symbol in enumerate(self.symbols)
        }

    @property
    def identifier(self) -> str:
        return "rayvan-character-v1"

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id["\n"]

    def encode(self, text: str) -> list[int]:
        unknown_id = self._token_to_id[self.unknown_symbol]
        return [self._token_to_id.get(character, unknown_id) for character in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            symbol = self.symbols[int(token_id)]
            decoded.append("�" if symbol == self.unknown_symbol else symbol)
        return "".join(decoded)

    def to_config(self) -> dict[str, Any]:
        return {"kind": "character", "symbols": list(self.symbols)}


@dataclass(frozen=True)
class LanguageCorpus:
    tokenizer: TextTokenizer
    train_tokens: Tensor
    validation_tokens: Tensor

    @classmethod
    def from_texts(
        cls,
        train_texts: Iterable[str],
        validation_texts: Iterable[str] | None = None,
        *,
        tokenizer: TextTokenizer | None = None,
    ) -> LanguageCorpus:
        train_documents = tuple(train_texts)
        validation_documents = (
            tuple(validation_texts)
            if validation_texts is not None
            else train_documents
        )
        if not train_documents or not validation_documents:
            raise ValueError("training and validation text collections cannot be empty")

        resolved_tokenizer = tokenizer or CharacterTokenizer.from_texts(
            (*train_documents, *validation_documents, "\n")
        )
        return cls(
            tokenizer=resolved_tokenizer,
            train_tokens=_tokenize_documents(train_documents, resolved_tokenizer),
            validation_tokens=_tokenize_documents(
                validation_documents, resolved_tokenizer
            ),
        )

    def sample_batch(
        self,
        split: Literal["train", "validation"],
        batch_size: int,
        sequence_length: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        tokens = self.train_tokens if split == "train" else self.validation_tokens
        maximum_start = tokens.numel() - sequence_length - 1
        if maximum_start < 0:
            raise ValueError(
                f"{split} corpus needs at least {sequence_length + 1} tokens"
            )
        starts = torch.randint(
            maximum_start + 1, (batch_size,), generator=generator
        ).tolist()
        inputs = torch.stack(
            [tokens[start : start + sequence_length] for start in starts]
        )
        targets = torch.stack(
            [tokens[start + 1 : start + sequence_length + 1] for start in starts]
        )
        return inputs.to(device), targets.to(device)

    def fixed_sequences(
        self,
        split: Literal["train", "validation"],
        sequence_length: int,
    ) -> tuple[Tensor, Tensor]:
        tokens = self.train_tokens if split == "train" else self.validation_tokens
        sequence_count = (tokens.numel() - 1) // sequence_length
        if sequence_count == 0:
            raise ValueError(
                f"{split} corpus needs at least {sequence_length + 1} tokens"
            )
        packed_length = sequence_count * sequence_length
        inputs = tokens[:packed_length].reshape(sequence_count, sequence_length)
        targets = tokens[1 : packed_length + 1].reshape(
            sequence_count, sequence_length
        )
        return inputs, targets


def tiny_overfit_corpus() -> LanguageCorpus:
    return LanguageCorpus.from_texts(TINY_OVERFIT_TEXTS, TINY_OVERFIT_TEXTS)


def load_tinystories(
    *,
    max_train_stories: int = 10_000,
    max_validation_stories: int = 1_000,
    tokenizer_identifier: str = DEFAULT_TOKENIZER_IDENTIFIER,
) -> LanguageCorpus:
    """Stream deterministic TinyStories subsets into packed GPT-2 token tensors."""
    if max_train_stories <= 0 or max_validation_stories <= 0:
        raise ValueError("TinyStories subset sizes must be positive")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'TinyStories requires: python -m pip install -e ".[data]"'
        ) from error

    tokenizer = HuggingFaceTokenizer.from_pretrained(tokenizer_identifier)
    train_stream = load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True
    )
    validation_stream = load_dataset(
        "roneneldan/TinyStories", split="validation", streaming=True
    )
    train_documents = (
        str(row["text"]) for row in train_stream.take(max_train_stories)
    )
    validation_documents = (
        str(row["text"])
        for row in validation_stream.take(max_validation_stories)
    )
    return LanguageCorpus(
        tokenizer=tokenizer,
        train_tokens=_tokenize_documents(train_documents, tokenizer),
        validation_tokens=_tokenize_documents(validation_documents, tokenizer),
    )


def _tokenize_documents(
    documents: Iterable[str],
    tokenizer: TextTokenizer,
    *,
    chunk_size: int = 1_000_000,
) -> Tensor:
    chunks: list[Tensor] = []
    buffer: list[int] = []
    document_count = 0
    for document in documents:
        document_count += 1
        buffer.extend(tokenizer.encode(document))
        buffer.append(tokenizer.eos_token_id)
        if len(buffer) >= chunk_size:
            chunks.append(torch.tensor(buffer, dtype=torch.long))
            buffer.clear()
    if not document_count:
        raise ValueError("cannot tokenize an empty document collection")
    if buffer:
        chunks.append(torch.tensor(buffer, dtype=torch.long))
    return torch.cat(chunks)
