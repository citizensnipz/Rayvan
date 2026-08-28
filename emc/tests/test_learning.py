import math

import torch
from torch import nn

from rayvan_emc import (
    EMCConfig,
    EMCCycleTrace,
    EMCDiagnostics,
    EMCModel,
    TrainingConfig,
    TransformerConfig,
    TransformerLanguageModel,
    count_parameters,
    evaluate_model,
    generate_text,
    next_token_loss,
    parameter_counts,
    tiny_overfit_corpus,
    train_model,
)


torch.set_num_threads(1)


def emc_config(**overrides: int) -> EMCConfig:
    values = {
        "latent_dim": 16,
        "num_modules": 3,
        "modules_per_cycle": 2,
        "num_cycles": 2,
        "vocab_size": 23,
        "max_sequence_length": 16,
        "module_hidden_dim": 32,
        "attention_heads": 2,
    }
    values.update(overrides)
    return EMCConfig(**values)


def baseline_config(**overrides: int) -> TransformerConfig:
    values = {
        "vocab_size": 23,
        "latent_dim": 16,
        "num_layers": 2,
        "attention_heads": 2,
        "feed_forward_dim": 32,
        "max_sequence_length": 16,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def random_next_token_batch(vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab_size, (3, 9))
    return tokens[:, :-1], tokens[:, 1:]


def backward_emc() -> EMCModel:
    torch.manual_seed(3)
    model = EMCModel(emc_config())
    inputs, targets = random_next_token_batch(model.config.vocab_size)
    logits = model(inputs)
    assert isinstance(logits, torch.Tensor)
    next_token_loss(logits, targets).backward()
    return model

def test_language_models_do_not_leak_future_tokens() -> None:
    torch.manual_seed(2)
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = torch.tensor([[1, 2, 3, 4, 9, 10, 11, 12]])
    models = (
        EMCModel(emc_config()),
        TransformerLanguageModel(baseline_config()),
    )

    for model in models:
        model.eval()
        first_logits = model(first)
        second_logits = model(second)
        assert isinstance(first_logits, torch.Tensor)
        assert isinstance(second_logits, torch.Tensor)
        torch.testing.assert_close(first_logits[:, :4], second_logits[:, :4])


def has_nonzero_gradient(parameters: object) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )


def test_next_token_loss_computes_for_emc() -> None:
    model = EMCModel(emc_config())
    inputs, targets = random_next_token_batch(model.config.vocab_size)

    logits = model(inputs)
    assert isinstance(logits, torch.Tensor)
    loss = next_token_loss(logits, targets)

    assert loss.ndim == 0
    assert math.isfinite(loss.item())


def test_next_token_loss_computes_for_baseline() -> None:
    model = TransformerLanguageModel(baseline_config())
    inputs, targets = random_next_token_batch(model.config.vocab_size)

    loss = next_token_loss(model(inputs), targets)

    assert loss.ndim == 0
    assert math.isfinite(loss.item())


def test_training_step_updates_emc_parameters() -> None:
    torch.manual_seed(4)
    model = EMCModel(emc_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs, targets = random_next_token_batch(model.config.vocab_size)
    before = model.output_projection.weight.detach().clone()

    logits = model(inputs)
    assert isinstance(logits, torch.Tensor)
    loss = next_token_loss(logits, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, model.output_projection.weight)


def test_router_parameters_receive_gradients() -> None:
    model = backward_emc()

    assert has_nonzero_gradient(model.router.parameters())


def test_selected_module_parameters_receive_gradients() -> None:
    model = backward_emc()

    assert has_nonzero_gradient(
        parameter
        for module in model.emc_modules
        for parameter in module.parameters()
    )


def test_integrator_parameters_receive_gradients() -> None:
    model = backward_emc()

    assert has_nonzero_gradient(model.integrator.parameters())


def test_tiny_corpus_overfit_substantially_reduces_loss() -> None:
    corpus = tiny_overfit_corpus()
    torch.manual_seed(5)
    model = EMCModel(
        emc_config(
            vocab_size=corpus.tokenizer.vocab_size,
            num_cycles=1,
            max_sequence_length=16,
        )
    )
    config = TrainingConfig(
        steps=80,
        batch_size=8,
        sequence_length=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        evaluation_interval=80,
        evaluation_batches=3,
        seed=5,
    )
    initial_loss, _ = evaluate_model(model, corpus, config)

    result = train_model(model, corpus, config, print_progress=False)

    assert result.final_training_loss < initial_loss * 0.65
    assert result.final_validation_loss < initial_loss * 0.70


def test_generation_runs_for_both_models() -> None:
    corpus = tiny_overfit_corpus()
    emc = EMCModel(
        emc_config(
            vocab_size=corpus.tokenizer.vocab_size,
            max_sequence_length=16,
        )
    )
    baseline = TransformerLanguageModel(
        baseline_config(
            vocab_size=corpus.tokenizer.vocab_size,
            max_sequence_length=16,
        )
    )

    for model in (emc, baseline):
        generated = generate_text(
            model,
            corpus.tokenizer,
            "the ",
            max_new_tokens=5,
            greedy=True,
        )
        assert generated.startswith("the ")
        assert len(generated) == 9


def test_parameter_count_utilities_match_model_parameters() -> None:
    linear = nn.Linear(3, 2)
    assert count_parameters(linear) == 8
    assert parameter_counts(linear).total == 8

    baseline = TransformerLanguageModel(baseline_config())
    baseline_counts = parameter_counts(baseline)
    assert baseline_counts.total == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert baseline_counts.approximate_active_per_cycle == baseline_counts.total

    emc = EMCModel(emc_config())
    emc_counts = parameter_counts(emc)
    assert emc_counts.total == sum(parameter.numel() for parameter in emc.parameters())
    assert emc_counts.approximate_active_per_cycle < emc_counts.total
    assert (
        emc_counts.approximate_parameter_uses_per_forward
        > emc_counts.approximate_active_per_cycle
    )


def test_routing_statistics_are_collected_correctly() -> None:
    model = EMCModel(
        emc_config(
            latent_dim=4,
            num_modules=3,
            modules_per_cycle=1,
            num_cycles=2,
            module_hidden_dim=8,
            attention_heads=1,
        )
    )
    diagnostics = EMCDiagnostics(model)
    latent_shape = (1, 4, 4)
    diagnostics.observe_trace(
        (
            EMCCycleTrace(1, (0,), torch.tensor([2.0, 0.0, 0.0]), torch.ones(1), latent_shape),
            EMCCycleTrace(2, (1,), torch.tensor([0.0, 2.0, 0.0]), torch.ones(1), latent_shape),
        )
    )
    diagnostics.observe_trace(
        (
            EMCCycleTrace(1, (2,), torch.tensor([0.0, 0.0, 2.0]), torch.ones(1), latent_shape),
            EMCCycleTrace(2, (1,), torch.tensor([0.0, 3.0, 0.0]), torch.ones(1), latent_shape),
        )
    )
    for parameter in model.router.parameters():
        parameter.grad = torch.ones_like(parameter)
    diagnostics.observe_router_gradients(model)
    with torch.no_grad():
        next(model.emc_modules[0].parameters()).add_(0.5)

    report = diagnostics.report(model)

    assert report.selection_counts == ((1, 0, 1), (0, 2, 0))
    assert report.traffic_fraction == (0.25, 0.5, 0.25)
    assert report.all_modules_used
    assert report.routing_differs_across_inputs
    assert report.routing_differs_across_cycles
    assert report.maximum_router_gradient_norm > 0
    assert report.module_update_norms[0] > 0
    assert report.module_updates_diverged
