from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor


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
        symbols = sorted(set(characters))
        self.symbols = (self.unknown_symbol, *symbols)
        self._token_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> CharacterTokenizer:
        return cls(character for text in texts for character in text)

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    def encode(self, text: str) -> list[int]:
        unknown_id = self._token_to_id[self.unknown_symbol]
        return [self._token_to_id.get(character, unknown_id) for character in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            symbol = self.symbols[int(token_id)]
            decoded.append("�" if symbol == self.unknown_symbol else symbol)
        return "".join(decoded)


@dataclass(frozen=True)
class LanguageCorpus:
    tokenizer: CharacterTokenizer
    train_tokens: Tensor
    validation_tokens: Tensor

    @classmethod
    def from_texts(
        cls,
        train_texts: Iterable[str],
        validation_texts: Iterable[str] | None = None,
    ) -> LanguageCorpus:
        train_documents = tuple(train_texts)
        validation_documents = (
            tuple(validation_texts)
            if validation_texts is not None
            else train_documents
        )
        if not train_documents or not validation_documents:
            raise ValueError("training and validation text collections cannot be empty")

        tokenizer = CharacterTokenizer.from_texts(
            (*train_documents, *validation_documents, "\n")
        )
        train_text = "\n".join(train_documents) + "\n"
        validation_text = "\n".join(validation_documents) + "\n"
        return cls(
            tokenizer=tokenizer,
            train_tokens=torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
            validation_tokens=torch.tensor(
                tokenizer.encode(validation_text), dtype=torch.long
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


def tiny_overfit_corpus() -> LanguageCorpus:
    return LanguageCorpus.from_texts(TINY_OVERFIT_TEXTS, TINY_OVERFIT_TEXTS)


def load_tinystories(
    *, max_train_stories: int = 2_000, max_validation_stories: int = 200
) -> LanguageCorpus:
    """Load bounded TinyStories subsets through the optional datasets package."""
    if max_train_stories <= 0 or max_validation_stories <= 0:
        raise ValueError("TinyStories subset sizes must be positive")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'TinyStories requires: python -m pip install -e ".[data]"'
        ) from error

    train_stream = load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True
    )
    validation_stream = load_dataset(
        "roneneldan/TinyStories", split="validation", streaming=True
    )
    train_texts = [row["text"] for row in train_stream.take(max_train_stories)]
    validation_texts = [
        row["text"] for row in validation_stream.take(max_validation_stories)
    ]
    return LanguageCorpus.from_texts(train_texts, validation_texts)
