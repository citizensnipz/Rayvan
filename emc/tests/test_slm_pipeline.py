from pathlib import Path

import torch

from rayvan_emc import (
    EMCConfig,
    EMCModel,
    HuggingFaceTokenizer,
    TrainingConfig,
    generate_text,
    load_model_checkpoint,
    load_training_checkpoint,
    next_token_loss,
    save_training_checkpoint,
    steps_for_token_budget,
    tiny_overfit_corpus,
    train_model,
)


torch.set_num_threads(1)


class FakeFastTokenizer:
    is_fast = True
    eos_token_id = 2

    def __init__(self) -> None:
        self.symbols = tuple("abcdefghijklmnopqrstuvwxyz .")
        self.to_id = {symbol: index + 3 for index, symbol in enumerate(self.symbols)}
        self.from_id = {index: symbol for symbol, index in self.to_id.items()}

    def __len__(self) -> int:
        return len(self.symbols) + 3

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.to_id.get(character, 0) for character in text.lower()]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.from_id.get(token_id, "") for token_id in token_ids)


def tiny_model() -> tuple[EMCModel, object]:
    corpus = tiny_overfit_corpus()
    model = EMCModel(
        EMCConfig(
            latent_dim=8,
            num_modules=4,
            modules_per_cycle=2,
            num_cycles=2,
            vocab_size=corpus.tokenizer.vocab_size,
            max_sequence_length=8,
            module_hidden_dim=16,
            attention_heads=2,
        )
    )
    return model, corpus


def populate_optimizer(model: EMCModel, optimizer: torch.optim.Optimizer) -> None:
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    logits = model(tokens[:, :-1])
    assert isinstance(logits, torch.Tensor)
    next_token_loss(logits, tokens[:, 1:]).backward()
    optimizer.step()


def test_subword_tokenizer_adapter_encodes_and_decodes() -> None:
    tokenizer = HuggingFaceTokenizer(FakeFastTokenizer(), "fake-fast")

    encoded = tokenizer.encode("tiny story.")
    decoded = tokenizer.decode(encoded)

    assert encoded
    assert decoded == "tiny story."
    assert tokenizer.identifier == "fake-fast"
    assert tokenizer.eos_token_id == 2


def test_tokenizer_vocabulary_size_reaches_emc_config() -> None:
    tokenizer = HuggingFaceTokenizer(FakeFastTokenizer(), "fake-fast")
    config = EMCConfig(vocab_size=tokenizer.vocab_size)
    model = EMCModel(config)

    assert model.config.vocab_size == tokenizer.vocab_size
    assert model.token_embedding.num_embeddings == tokenizer.vocab_size
    assert model.output_projection.out_features == tokenizer.vocab_size


def test_packed_text_becomes_fixed_length_causal_sequences() -> None:
    tokenizer = HuggingFaceTokenizer(FakeFastTokenizer(), "fake-fast")
    from rayvan_emc import LanguageCorpus

    corpus = LanguageCorpus.from_texts(
        ["once upon a time.", "there was a dog."],
        ["a little story."],
        tokenizer=tokenizer,
    )
    inputs, targets = corpus.fixed_sequences("train", sequence_length=5)

    assert inputs.ndim == 2
    assert inputs.shape == targets.shape
    assert inputs.shape[1] == 5
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_token_budget_calculation_rounds_up_to_whole_steps() -> None:
    assert steps_for_token_budget(1_000_000, 8, 256) == 489
    assert steps_for_token_budget(4_096, 4, 256) == 4
    config = TrainingConfig(
        steps=None,
        train_tokens=10_000_000,
        batch_size=8,
        sequence_length=256,
    )
    assert config.planned_steps == 4_883


def test_checkpoint_round_trip_reproduces_model_state(tmp_path: Path) -> None:
    model, corpus = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "model.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        tokenizer=corpus.tokenizer,
        step=2,
        tokens_processed=32,
        validation_loss=1.5,
        best_validation_loss=1.5,
        training_config={"router_balance_coefficient": 0.01},
    )

    loaded = load_model_checkpoint(checkpoint)

    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, loaded.model.state_dict()[name])
    assert loaded.progress.step == 2
    assert loaded.progress.tokens_processed == 32
    assert loaded.tokenizer.to_config() == corpus.tokenizer.to_config()


def test_resume_restores_optimizer_and_training_progress(tmp_path: Path) -> None:
    model, corpus = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    populate_optimizer(model, optimizer)
    checkpoint = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        tokenizer=corpus.tokenizer,
        step=3,
        tokens_processed=48,
        validation_loss=1.4,
        best_validation_loss=1.2,
        training_config={"precision": "fp32"},
    )
    restored_model, _ = tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)

    progress = load_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert progress.step == 3
    assert progress.tokens_processed == 48
    assert progress.best_validation_loss == 1.2
    assert restored_optimizer.state
    assert restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()[
        "state"
    ].keys()


def test_training_tracks_latest_and_best_checkpoints(tmp_path: Path) -> None:
    model, corpus = tiny_model()
    result = train_model(
        model,
        corpus,
        TrainingConfig(
            steps=2,
            batch_size=2,
            sequence_length=8,
            evaluation_interval=1,
            evaluation_batches=1,
            checkpoint_directory=str(tmp_path),
            checkpoint_prefix="emc",
            seed=9,
        ),
        print_progress=False,
    )

    assert result.latest_checkpoint is not None
    assert result.best_checkpoint is not None
    assert Path(result.latest_checkpoint).exists()
    assert Path(result.best_checkpoint).exists()
    best = load_model_checkpoint(result.best_checkpoint)
    latest = load_model_checkpoint(result.latest_checkpoint)
    assert best.progress.validation_loss == best.progress.best_validation_loss
    assert best.progress.best_validation_loss <= latest.progress.validation_loss


def test_generation_runs_from_saved_checkpoint(tmp_path: Path) -> None:
    model, corpus = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "generation.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        tokenizer=corpus.tokenizer,
        step=1,
        tokens_processed=16,
        validation_loss=2.0,
        best_validation_loss=2.0,
        training_config={},
    )

    loaded = load_model_checkpoint(checkpoint)
    generated = generate_text(
        loaded.model,
        loaded.tokenizer,
        "the ",
        max_new_tokens=4,
        greedy=True,
    )

    assert generated.startswith("the ")
    assert len(generated) == 8
