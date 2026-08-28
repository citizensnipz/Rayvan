import math

import torch

from rayvan_emc import (
    EMCConfig,
    EMCCycleTrace,
    EMCDiagnostics,
    EMCModel,
    EMCOutput,
    TrainingConfig,
    TransformerConfig,
    TransformerLanguageModel,
    next_token_loss,
    router_balance_metrics,
    tiny_overfit_corpus,
    train_model,
)


torch.set_num_threads(1)


def balance_case(counts: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module_count = len(counts)
    assignments = torch.tensor(
        [module for module, count in enumerate(counts) for _ in range(count)],
        dtype=torch.long,
    ).reshape(1, -1, 1)
    scores = torch.zeros(
        1, assignments.size(1), module_count, requires_grad=True
    )
    metrics = router_balance_metrics(scores, assignments, entropy_floor=0.75)
    return metrics.loss, metrics.normalized_entropy, scores


def test_balanced_routing_has_no_balance_penalty() -> None:
    loss, normalized_entropy, _ = balance_case([10] * 8)

    assert math.isclose(normalized_entropy.item(), 1.0, rel_tol=1e-6)
    assert loss.item() == 0.0


def test_extreme_concentration_has_larger_balance_penalty() -> None:
    balanced_loss, _, _ = balance_case([10] * 8)
    concentrated_loss, concentrated_entropy, _ = balance_case(
        [50, 43, 2, 1, 1, 1, 1, 1]
    )

    assert concentrated_entropy.item() < 0.75
    assert concentrated_loss.item() > balanced_loss.item()
    assert concentrated_loss.item() > 0.01


def test_moderate_specialization_is_penalized_far_less_than_collapse() -> None:
    moderate_loss, moderate_entropy, _ = balance_case(
        [25, 20, 15, 10, 10, 8, 7, 5]
    )
    severe_loss, _, _ = balance_case([50, 43, 2, 1, 1, 1, 1, 1])

    assert moderate_entropy.item() > 0.75
    assert moderate_loss.item() < severe_loss.item() * 0.1


def test_balance_loss_produces_router_gradients() -> None:
    config = EMCConfig(
        latent_dim=8,
        num_modules=4,
        modules_per_cycle=2,
        num_cycles=2,
        vocab_size=17,
        max_sequence_length=8,
        module_hidden_dim=16,
        attention_heads=2,
    )
    model = EMCModel(config)
    with torch.no_grad():
        model.router.score_projection.weight.zero_()
        model.router.score_projection.bias.copy_(
            torch.tensor([6.0, 5.0, -6.0, -6.0])
        )

    output = model(
        torch.randint(0, config.vocab_size, (3, 8)),
        return_trace=True,
    )
    assert isinstance(output, EMCOutput)
    assert output.router_balance_loss is not None
    output.router_balance_loss.backward()

    router_gradient = model.router.score_projection.bias.grad
    assert router_gradient is not None
    assert torch.count_nonzero(router_gradient).item() > 0


def test_language_model_gradients_still_propagate_normally() -> None:
    config = EMCConfig(
        latent_dim=8,
        num_modules=4,
        modules_per_cycle=2,
        num_cycles=2,
        vocab_size=17,
        max_sequence_length=8,
        module_hidden_dim=16,
        attention_heads=2,
    )
    model = EMCModel(config)
    tokens = torch.randint(0, config.vocab_size, (3, 9))
    output = model(tokens[:, :-1])
    assert isinstance(output, torch.Tensor)

    next_token_loss(output, tokens[:, 1:]).backward()

    assert model.token_embedding.weight.grad is not None
    assert model.integrator.update_projection.weight.grad is not None
    assert model.output_projection.weight.grad is not None
    assert any(
        parameter.grad is not None
        for module in model.emc_modules
        for parameter in module.parameters()
    )


def test_dead_zone_allows_nonuniform_router_probabilities() -> None:
    counts = [25, 20, 15, 10, 10, 8, 7, 5]
    loss, _, scores = balance_case(counts)
    probabilities = torch.tensor(counts, dtype=torch.float32)
    probabilities /= probabilities.sum()
    with torch.no_grad():
        scores.copy_(probabilities.log().reshape(1, 1, -1))

    metrics = router_balance_metrics(
        scores,
        torch.tensor(
            [module for module, count in enumerate(counts) for _ in range(count)]
        ).reshape(1, -1, 1),
        entropy_floor=0.75,
    )

    assert loss.item() == 0.0
    assert metrics.loss.item() == 0.0
    assert not torch.allclose(probabilities, torch.full_like(probabilities, 1 / 8))


def test_diagnostics_flag_top_two_above_ninety_percent() -> None:
    config = EMCConfig(
        latent_dim=8,
        num_modules=8,
        modules_per_cycle=2,
        num_cycles=1,
        vocab_size=17,
        max_sequence_length=100,
        module_hidden_dim=16,
        attention_heads=2,
    )
    model = EMCModel(config)
    diagnostics = EMCDiagnostics(model)
    decisions = [[0, 1] for _ in range(93)]
    decisions.extend([[2 + index % 6, 2 + (index + 1) % 6] for index in range(7)])
    selected_indices = torch.tensor(decisions).reshape(1, 100, 2)
    diagnostics.observe_trace(
        (
            EMCCycleTrace(
                cycle=1,
                selected_modules=tuple(range(8)),
                router_scores=torch.zeros(1, 100, 8),
                router_weights=torch.full((1, 100, 2), 0.5),
                latent_shape=(1, 100, 8),
                selected_indices=selected_indices,
            ),
        )
    )

    report = diagnostics.report(model)

    assert report.all_modules_used
    assert report.top_2_traffic_share > 0.90
    assert report.top_1_traffic_share < report.top_2_traffic_share
    assert report.minimum_module_share > 0
    assert report.normalized_routing_entropy < 0.75
    assert report.effective_active_modules < 4
    assert report.severe_collapse
    assert report.routing_collapsed


def test_baseline_receives_no_balance_contribution() -> None:
    corpus = tiny_overfit_corpus()
    model = TransformerLanguageModel(
        TransformerConfig(
            vocab_size=corpus.tokenizer.vocab_size,
            latent_dim=8,
            num_layers=1,
            attention_heads=2,
            feed_forward_dim=16,
            max_sequence_length=8,
        )
    )
    result = train_model(
        model,
        corpus,
        TrainingConfig(
            steps=1,
            batch_size=2,
            sequence_length=8,
            evaluation_interval=1,
            evaluation_batches=1,
            router_balance_coefficient=1.0,
        ),
        print_progress=False,
    )

    assert math.isclose(result.average_router_balance_loss, 0.0)
    assert math.isclose(result.average_weighted_balance_contribution, 0.0)
