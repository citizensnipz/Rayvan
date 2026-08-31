import math

import torch

from rayvan_emc import (
    EMCConfig,
    EMCDiagnostics,
    EMCModel,
    EMCModule,
    EMCOutput,
    Integrator,
    ModuleAwareNexusRouter,
    RecurrentEMCModule,
    StateSpaceEMCModule,
    TrainingConfig,
    WeightedAverageIntegrator,
    evaluate_emc_cycles,
    tiny_overfit_corpus,
)
from rayvan_emc.experiments.common import (
    MODULE_POPULATIONS,
    create_emc_model,
)


torch.set_num_threads(1)


def n1_config(**overrides: object) -> EMCConfig:
    values: dict[str, object] = {
        "latent_dim": 8,
        "num_modules": 4,
        "modules_per_cycle": 2,
        "num_cycles": 2,
        "vocab_size": 17,
        "max_sequence_length": 8,
        "module_hidden_dim": 16,
        "attention_heads": 2,
        "module_families": ("gpt", "ssm", "recurrent", "gpt"),
        "state_space_dim": 12,
        "recurrent_dim": 8,
        "router_type": "module_aware",
        "router_descriptor_dim": 4,
        "integrator_type": "proposal_attention",
        "integrator_heads": 2,
    }
    values.update(overrides)
    return EMCConfig(**values)


def test_weighted_average_integrator_remains_reproducible_baseline() -> None:
    config = n1_config(integrator_type="weighted_average")
    integrator = WeightedAverageIntegrator(config)
    latent = torch.randn(2, 4, config.latent_dim)
    proposals = torch.randn(2, 4, 2, config.latent_dim)
    weights = torch.softmax(torch.randn(2, 4, 2), dim=-1)

    output = integrator(latent, proposals, weights)

    assert isinstance(output, torch.Tensor)
    assert output.shape == latent.shape


def test_proposal_aware_integrator_preserves_and_adjudicates_proposals() -> None:
    config = n1_config()
    integrator = Integrator(config)
    latent = torch.randn(1, 3, config.latent_dim)
    first = torch.ones(1, 3, config.latent_dim)
    second = -torch.ones(1, 3, config.latent_dim) * 2
    proposals = torch.stack((first, second), dim=2).requires_grad_()
    routing_weights = torch.full((1, 3, 2), 0.5)

    result = integrator(
        latent,
        proposals,
        routing_weights,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple)
    output, trace = result
    output.square().mean().backward()

    assert output.shape == latent.shape
    assert trace.proposal_acceptance.shape == (1, 3, 2)
    torch.testing.assert_close(
        trace.proposal_acceptance.sum(dim=-1), torch.ones(1, 3)
    )
    assert torch.all(trace.proposal_norms[..., 1] > trace.proposal_norms[..., 0])
    assert torch.all(trace.proposal_similarity[..., 0, 1] < -0.99)
    assert trace.gate_magnitude.shape == (1, 3)
    assert proposals.grad is not None
    assert torch.count_nonzero(proposals.grad[..., 0, :]).item() > 0
    assert torch.count_nonzero(proposals.grad[..., 1, :]).item() > 0


def test_all_module_families_share_the_same_proposal_contract() -> None:
    config = n1_config()
    latent = torch.randn(2, 6, config.latent_dim)
    modules = (
        EMCModule(config),
        StateSpaceEMCModule(config),
        RecurrentEMCModule(config),
    )

    for module in modules:
        first = module(latent)
        second = module(latent)
        assert first.shape == latent.shape
        torch.testing.assert_close(first, second)


def test_homogeneous_and_mixed_population_configs_are_supported() -> None:
    expected = {
        "gpt-only": {"gpt"},
        "ssm-only": {"ssm"},
        "recurrent-only": {"recurrent"},
        "gpt-ssm": {"gpt", "ssm"},
        "gpt-recurrent": {"gpt", "recurrent"},
        "ssm-recurrent": {"ssm", "recurrent"},
        "mixed": {"gpt", "ssm", "recurrent"},
    }
    assert set(expected).issubset(MODULE_POPULATIONS)

    for population, families in expected.items():
        model = create_emc_model(
            31,
            "quick",
            maximum_sequence_length=4,
            seed=2,
            n1_stage="heterogeneous",
            module_population=population,
        )
        assert set(model.module_families) == families
        logits = model(torch.randint(0, 31, (1, 4)))
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 4, 31)


def test_staged_configs_isolate_three_n1_hypotheses() -> None:
    baseline = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=4,
        seed=1,
        n1_stage="baseline",
    )
    integrator = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=4,
        seed=1,
        n1_stage="integrator",
    )
    heterogeneous = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=4,
        seed=1,
        n1_stage="heterogeneous",
    )
    n1 = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=4,
        seed=1,
        n1_stage="n1",
    )

    assert baseline.config.integrator_type == "weighted_average"
    assert baseline.config.router_type == "fixed_index"
    assert set(baseline.module_families) == {"gpt"}
    assert integrator.config.integrator_type == "proposal_attention"
    assert integrator.config.router_type == "fixed_index"
    assert set(integrator.module_families) == {"gpt"}
    assert heterogeneous.config.integrator_type == "proposal_attention"
    assert heterogeneous.config.router_type == "fixed_index"
    assert len(set(heterogeneous.module_families)) == 3
    assert n1.config.integrator_type == "proposal_attention"
    assert n1.config.router_type == "module_aware"
    assert len(set(n1.module_families)) == 3


def test_module_aware_nexus_scores_learned_descriptors() -> None:
    config = n1_config()
    router = ModuleAwareNexusRouter(config)
    with torch.no_grad():
        router.query_projection.weight.zero_()
        router.query_projection.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        router.module_descriptors.zero_()
        router.module_descriptors[0, 0] = 2.0
        router.module_descriptors[1, 0] = 1.0
    latent = torch.randn(1, 3, config.latent_dim)

    routing = router(latent)
    routing.scores[..., 0].sum().backward()

    assert routing.selected_indices.shape == (1, 3, 2)
    assert torch.all(routing.selected_indices[..., 0] == 0)
    assert router.module_descriptors.grad is not None
    assert torch.count_nonzero(router.module_descriptors.grad).item() > 0


def test_module_availability_mask_excludes_unavailable_modules() -> None:
    model = EMCModel(n1_config())
    mask = torch.tensor([False, True, True, False])
    result = model(
        torch.randint(0, model.config.vocab_size, (1, 4)),
        return_trace=True,
        availability_mask=mask,
    )
    assert isinstance(result, EMCOutput)

    for cycle in result.trace:
        assert cycle.selected_indices is not None
        assert set(cycle.selected_indices.unique().tolist()) == {1, 2}


def test_n1_instrumentation_reports_modules_families_and_integrator() -> None:
    model = EMCModel(n1_config())
    diagnostics = EMCDiagnostics(model)
    tokens = torch.randint(0, model.config.vocab_size, (2, 5))
    output = model(tokens, return_trace=True)
    assert isinstance(output, EMCOutput)
    diagnostics.observe_trace(output.trace)
    output.logits.square().mean().backward()
    diagnostics.observe_router_gradients(model)
    diagnostics.observe_module_gradients(model)

    report = diagnostics.report(model)

    assert len(report.average_routing_probability) == 4
    assert len(report.average_integrator_acceptance) == 4
    assert len(report.average_proposal_norm) == 4
    assert len(report.module_gradient_norms) == 4
    assert len(report.module_parameter_counts) == 4
    assert {family for family, _ in report.family_traffic_fraction} == {
        "gpt",
        "ssm",
        "recurrent",
    }
    assert math.isfinite(report.mean_proposal_similarity)
    assert report.mean_integrated_update_norm > 0
    assert 0 < report.mean_gate_magnitude < 1


def test_cycle_logits_and_cycle_limited_evaluation_are_available() -> None:
    corpus = tiny_overfit_corpus()
    model = create_emc_model(
        corpus.tokenizer.vocab_size,
        "quick",
        maximum_sequence_length=8,
        seed=5,
        n1_stage="n1",
    )
    model.eval()
    output = model(
        torch.randint(0, corpus.tokenizer.vocab_size, (1, 8)),
        return_cycle_logits=True,
    )
    assert isinstance(output, EMCOutput)
    assert output.cycle_logits is not None
    assert len(output.cycle_logits) == model.config.num_cycles

    cycle_metrics = evaluate_emc_cycles(
        model,
        corpus,
        TrainingConfig(
            steps=1,
            batch_size=1,
            sequence_length=8,
            evaluation_batches=1,
        ),
    )
    assert len(cycle_metrics) == model.config.num_cycles
    assert [metric.cycle for metric in cycle_metrics] == [1, 2]
    assert all(metric.validation_loss > 0 for metric in cycle_metrics)
    assert all(metric.seconds_per_batch > 0 for metric in cycle_metrics)
