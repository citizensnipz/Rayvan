from pathlib import Path

import torch

from rayvan_emc import (
    EMCDiagnostics,
    EMCOutput,
    LanguageCorpus,
    count_parameters,
    load_model_checkpoint,
    next_token_loss,
    parameter_breakdown,
    parameter_counts,
    save_training_checkpoint,
    steps_for_token_budget,
    tiny_overfit_corpus,
)
from rayvan_emc.experiments.common import create_emc_model
from rayvan_emc.tokenization import HuggingFaceTokenizer


torch.set_num_threads(1)


class RawEncoding:
    def __init__(self, token_ids: list[int]) -> None:
        self.ids = token_ids


class UnlimitedRawTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> RawEncoding:
        assert not add_special_tokens
        return RawEncoding([3 + ord(character) % 32 for character in text])


class LengthWarningBackend:
    is_fast = True
    eos_token_id = 2
    backend_tokenizer = UnlimitedRawTokenizer()

    def __len__(self) -> int:
        return 64

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("model-length-limited encode path must not be used")

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        return "".join("x" for _ in token_ids)


def research_model(vocab_size: int = 50_257, context: int = 256):
    return create_emc_model(
        vocab_size,
        "research",
        maximum_sequence_length=context,
        seed=11,
        tie_embeddings=True,
    )

def test_research_preset_is_in_target_parameter_range() -> None:
    model = research_model()
    counts = parameter_counts(model)

    assert 25_000_000 <= counts.total <= 35_000_000
    assert model.config.latent_dim == 256
    assert model.config.num_modules == 4
    assert model.config.modules_per_cycle == 2
    assert model.config.num_cycles == 2
    assert model.config.module_hidden_dim == 6_144


def test_parameter_breakdown_matches_unique_model_total() -> None:
    model = research_model()
    breakdown = parameter_breakdown(model)
    categorized_total = (
        breakdown.token_embeddings
        + breakdown.position_embeddings
        + breakdown.router
        + breakdown.modules_combined
        + breakdown.integrator
        + breakdown.final_normalization
        + breakdown.output_projection_unique
    )

    assert categorized_total == count_parameters(model)
    assert breakdown.total_parameters == count_parameters(model)
    assert breakdown.trainable_parameters == count_parameters(model)
    assert sum(breakdown.modules_individual) == breakdown.modules_combined
    assert breakdown.modules_combined > breakdown.token_embeddings
    assert breakdown.output_projection_unique == model.config.vocab_size


def test_research_preset_really_ties_vocabulary_weights() -> None:
    model = research_model()
    breakdown = parameter_breakdown(model)

    assert model.output_projection.weight is model.token_embedding.weight
    assert breakdown.output_weight_tied


def test_research_preset_completes_forward_and_backward() -> None:
    model = research_model(context=8)
    tokens = torch.randint(0, model.config.vocab_size, (1, 5))

    output = model(tokens[:, :-1], return_trace=True)
    assert isinstance(output, EMCOutput)
    loss = next_token_loss(output.logits, tokens[:, 1:])
    assert output.router_balance_loss is not None
    (loss + 0.01 * output.router_balance_loss).backward()

    assert model.token_embedding.weight.grad is not None
    assert model.integrator.update_projection.weight.grad is not None
    assert any(
        parameter.grad is not None
        for module in model.emc_modules
        for parameter in module.parameters()
    )


def test_checkpoint_round_trip_handles_large_module_configuration(
    tmp_path: Path,
) -> None:
    corpus = tiny_overfit_corpus()
    model = research_model(corpus.tokenizer.vocab_size, context=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "research.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        tokenizer=corpus.tokenizer,
        step=1,
        tokens_processed=8,
        validation_loss=3.0,
        best_validation_loss=3.0,
        training_config={"gradient_accumulation_steps": 1},
    )

    loaded = load_model_checkpoint(checkpoint)

    assert loaded.model.config.module_hidden_dim == 6_144
    assert count_parameters(loaded.model) == count_parameters(model)
    assert torch.equal(
        loaded.model.emc_modules[0].feed_forward[0].weight,
        model.emc_modules[0].feed_forward[0].weight,
    )


def test_routing_diagnostics_work_with_scaled_four_module_preset() -> None:
    model = research_model(vocab_size=128, context=8)
    diagnostics = EMCDiagnostics(model)
    output = model(torch.randint(0, 128, (1, 8)), return_trace=True)
    assert isinstance(output, EMCOutput)

    diagnostics.observe_trace(output.trace)
    report = diagnostics.report(model)

    assert len(report.traffic_fraction) == 4
    assert len(report.routing_distribution_per_cycle) == 2
    assert 0.0 <= report.top_1_traffic_share <= 1.0
    assert 0.0 <= report.normalized_routing_entropy <= 1.0
    assert 1.0 <= report.effective_active_modules <= 4.0


def test_long_documents_are_chunked_before_model_context() -> None:
    tokenizer = HuggingFaceTokenizer(LengthWarningBackend(), "unlimited-test")
    long_story = "a" * 5_000
    corpus = LanguageCorpus.from_texts(
        [long_story],
        [long_story],
        tokenizer=tokenizer,
    )

    inputs, targets = corpus.fixed_sequences("train", sequence_length=256)
    sampled_inputs, sampled_targets = corpus.sample_batch(
        "train",
        batch_size=2,
        sequence_length=256,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )

    assert inputs.shape[1] == 256
    assert targets.shape == inputs.shape
    assert sampled_inputs.shape == (2, 256)
    assert sampled_targets.shape == sampled_inputs.shape
    assert corpus.train_tokens.numel() > 256


def test_token_budget_includes_gradient_accumulation() -> None:
    assert steps_for_token_budget(1_000_000, 1, 256, 4) == 977
    assert steps_for_token_budget(10_000_000, 1, 256, 4) == 9_766
