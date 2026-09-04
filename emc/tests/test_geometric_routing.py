from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rayvan_emc.integrator import SequentialAcceptanceIntegrator
from rayvan_emc.evaluate import _aggregate_geometric_counterfactuals
from rayvan_emc.model import EMCConfig, SequentialEMCModel
from rayvan_emc.nexus import GeometricNexusRouter
from rayvan_emc.research_config import ExperimentConfig, RoutingConfig
from rayvan_emc.research_runner import _build_model, _routing_warnings
from rayvan_emc.training import next_token_loss


def geometric_config(**overrides: object) -> EMCConfig:
    values: dict[str, object] = {
        "latent_dim": 8,
        "num_modules": 3,
        "modules_per_cycle": 1,
        "num_cycles": 3,
        "trajectory_steps": 3,
        "vocab_size": 16,
        "max_sequence_length": 8,
        "module_hidden_dim": 16,
        "attention_heads": 2,
        "integrator_heads": 2,
        "module_families": ("gpt", "recurrent", "delta"),
        "router_type": "geometric",
        "integrator_type": "acceptance_gate",
        "architecture_stage": "n1_sequential",
        "routing_geometry_dim": 4,
        "loss_free_balance_enabled": False,
        "switch_cost": 0.0,
        "persistence_bonus": 0.0,
        "counterfactual_probe_fixed_rate": 1.0,
        "counterfactual_max_probes_per_forward": 8,
        "counterfactual_probe_seed": 17,
    }
    values.update(overrides)
    return EMCConfig(**values)


def test_need_embedding_is_normalized_and_prototypes_are_trainable() -> None:
    router = GeometricNexusRouter(geometric_config())
    need, raw_norm = router.encode_need(torch.randn(2, 1, 8))
    torch.testing.assert_close(need.norm(dim=-1), torch.ones(2, 1))
    assert torch.all(raw_norm > 0)
    assert router.competence_prototypes.shape == (3, 1, 4)
    assert router.competence_prototypes.requires_grad


def test_distance_itself_determines_preference_without_classifier_logits() -> None:
    config = geometric_config(num_modules=2, module_families=("gpt", "recurrent"))
    router = GeometricNexusRouter(config)
    assert not hasattr(router, "score_projection")
    with torch.no_grad():
        for parameter in router.need_encoder.parameters():
            parameter.zero_()
        router.need_encoder[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        router.competence_prototypes.copy_(
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]])
        )
    decision = router.route_one(torch.randn(1, 1, 8))
    assert int(decision.selected_indices.item()) == 0
    torch.testing.assert_close(decision.base_actions[0, 0], torch.tensor([0.0, 2.0]))


def test_refractory_changes_effective_not_base_action() -> None:
    router = GeometricNexusRouter(
        geometric_config(num_modules=2, module_families=("gpt", "recurrent"))
    )
    latent = torch.randn(1, 1, 8)
    base = router.route_one(latent)
    penalty = torch.zeros(1, 1, 2)
    penalty[..., int(base.selected_indices.item())] = 10.0
    inhibited = router.route_one(latent, refractory_penalty=penalty)
    torch.testing.assert_close(base.base_actions, inhibited.base_actions)
    assert not torch.equal(base.selected_indices, inhibited.selected_indices)


def test_one_real_expert_executes_and_geometry_recomputes(monkeypatch) -> None:
    model = SequentialEMCModel(geometric_config(counterfactual_calibration_enabled=False))
    calls = [0, 0, 0]
    needs: list[torch.Tensor] = []
    original_route = model.router.route_one
    for index, expert in enumerate(model.emc_modules):
        original = expert.forward

        def counted(latent, *, _index=index, _original=original):
            calls[_index] += 1
            return _original(latent)

        monkeypatch.setattr(expert, "forward", counted)

    def observed(latent, **kwargs):
        needs.append(latent.detach().clone())
        return original_route(latent, **kwargs)

    monkeypatch.setattr(model.router, "route_one", observed)
    output = model(torch.randint(0, 16, (1, 5)), return_trace=True)
    assert sum(calls) == 3
    assert len(output.trace) == 3
    assert len(needs) == 3 and not torch.equal(needs[0], needs[1])


def test_counterfactual_probe_evaluates_every_expert_from_same_state(monkeypatch) -> None:
    model = SequentialEMCModel(geometric_config(trajectory_steps=1, num_cycles=1))
    states: list[torch.Tensor] = []
    for expert in model.emc_modules:
        original = expert.forward

        def observed(latent, *, _original=original):
            if not torch.is_grad_enabled():
                states.append(latent.detach().clone())
            return _original(latent)

        monkeypatch.setattr(expert, "forward", observed)
    inputs = torch.randint(0, 16, (2, 5))
    targets = torch.randint(0, 16, (2, 5))
    output = model(inputs, return_trace=True, counterfactual_targets=targets, training_step=2)
    probe = output.trace[0].counterfactual
    assert probe is not None
    assert probe.candidate_losses.shape == (2, 3)
    assert len(states) == 3
    assert all(torch.equal(states[0], state) for state in states[1:])


def test_regret_and_soft_calibration_direction_are_correct() -> None:
    model = SequentialEMCModel(geometric_config(trajectory_steps=1, num_cycles=1))
    inputs = torch.randint(0, 16, (2, 5))
    targets = torch.randint(0, 16, (2, 5))
    output = model(inputs, return_trace=True, counterfactual_targets=targets, training_step=1)
    probe = output.trace[0].counterfactual
    assert probe is not None
    expected = probe.candidate_losses.gather(
        1, probe.chosen_expert.unsqueeze(1)
    ).squeeze(1) - probe.candidate_losses.min(dim=1).values
    torch.testing.assert_close(probe.routing_regret, expected)
    assert output.geometry_calibration_loss is not None
    assert torch.isfinite(output.geometry_calibration_loss)


def test_calibration_updates_only_geometry_not_unselected_experts() -> None:
    model = SequentialEMCModel(geometric_config(trajectory_steps=1, num_cycles=1))
    output = model(
        torch.randint(0, 16, (2, 5)),
        return_trace=True,
        counterfactual_targets=torch.randint(0, 16, (2, 5)),
        training_step=1,
    )
    output.geometry_calibration_loss.backward()
    assert model.router.competence_prototypes.grad is not None
    assert model.router.need_encoder[1].weight.grad is not None
    assert all(parameter.grad is None for expert in model.emc_modules for parameter in expert.parameters())
    before = model.router.competence_prototypes.detach().clone()
    optimizer = torch.optim.SGD(model.router.parameters(), lr=1e-3)
    optimizer.step()
    assert not torch.equal(before, model.router.competence_prototypes)


def test_selected_expert_and_acceptance_integrator_receive_task_gradients() -> None:
    model = SequentialEMCModel(
        geometric_config(trajectory_steps=1, num_cycles=1, counterfactual_calibration_enabled=False)
    )
    assert isinstance(model.integrator, SequentialAcceptanceIntegrator)
    inputs = torch.randint(0, 16, (2, 5))
    targets = torch.randint(0, 16, (2, 5))
    output = model(inputs, return_trace=True)
    next_token_loss(output.logits, targets).backward()
    selected = set(output.trace[0].selected_indices.reshape(-1).tolist())
    assert any(parameter.grad is not None for parameter in model.integrator.parameters())
    assert all(
        any(parameter.grad is not None for parameter in model.emc_modules[index].parameters())
        for index in selected
    )


def test_probe_sampling_is_seeded_bounded_and_at_most_once_per_example() -> None:
    torch.manual_seed(5)
    first = SequentialEMCModel(geometric_config(counterfactual_max_probes_per_forward=2))
    torch.manual_seed(5)
    second = SequentialEMCModel(geometric_config(counterfactual_max_probes_per_forward=2))
    inputs = torch.randint(0, 16, (4, 5))
    targets = torch.randint(0, 16, (4, 5))
    outputs = [
        model(inputs, return_trace=True, counterfactual_targets=targets, training_step=11)
        for model in (first, second)
    ]
    indices = [
        [index for trace in output.trace if trace.counterfactual for index in trace.counterfactual.sample_indices.tolist()]
        for output in outputs
    ]
    assert indices[0] == indices[1]
    assert len(indices[0]) <= 2
    assert len(indices[0]) == len(set(indices[0]))


def test_schema_defaults_to_geometric_and_migrates_v2_legacy() -> None:
    default = ExperimentConfig()
    assert default.routing.router_type == "geometric"
    assert default.routing.integrator_type == "acceptance_gate"
    assert default.routing.loss_free_balance_enabled is False
    legacy_payload = default.to_dict()
    legacy_payload["schema_version"] = 2
    legacy_payload["routing"]["router_type"] = "module_aware"
    legacy_payload["routing"]["integrator_type"] = "proposal_attention"
    migrated = ExperimentConfig.from_dict(json.loads(json.dumps(legacy_payload)))
    assert migrated.architecture == "sequential_module_aware_emc"
    model = _build_model(default, 32)
    assert isinstance(model.router, GeometricNexusRouter)


def test_geometric_report_and_console_surface_required_metrics() -> None:
    model = SequentialEMCModel(geometric_config(trajectory_steps=1, num_cycles=1))
    model(
        torch.randint(0, 16, (2, 4)),
        return_trace=True,
        counterfactual_targets=torch.randint(0, 16, (2, 4)),
        training_step=1,
    )
    report = model.geometric_routing_report()
    assert report is not None
    assert {
        "total_probes",
        "counterfactual_top1_accuracy",
        "mean_routing_regret",
        "counterfactual_win_rates",
        "actual_routing_rates",
        "basin_occupancy",
        "per_step",
        "prototype_drift",
    } <= report.keys()
    source = (Path(__file__).parents[2] / "src" / "research" / "charts" / "RoutingOverview.tsx").read_text(encoding="utf-8")
    assert "Counterfactual routing regret" in source
    assert "Selected × counterfactual-best" in source
    assert "Expert win rate vs basin traffic" in source
    assert "Counterfactual quality by trajectory step" in source


def test_counterfactual_diagnostics_group_by_labels_without_training_on_them() -> None:
    report = _aggregate_geometric_counterfactuals(
        [
            {
                "metadata": {
                    "capability": "arithmetic",
                    "operation": "addition",
                    "surface_format": "words",
                    "difficulty": 2,
                },
                "geometric_counterfactual": [
                    {
                        "routing_regret": 0.2,
                        "top1_correct": True,
                        "top2_correct": True,
                    }
                ],
            }
        ]
    )
    assert report["by_capability"]["arithmetic"]["mean_routing_regret"] == pytest.approx(0.2)
    assert report["by_operation"]["addition"]["top1_accuracy"] == 1.0
    assert report["by_surface_format"]["words"]["probe_count"] == 1
    assert report["by_difficulty"]["2"]["top2_accuracy"] == 1.0


def test_documented_geometry_warnings_are_emitted() -> None:
    warnings = _routing_warnings(
        {
            "utilization": [1.0, 0.0],
            "geometric_routing": {
                "total_probes": 60,
                "total_routing_events": 120,
                "minimum_basin_separation": 0.01,
                "mean_geometric_margin": 0.001,
                "mean_routing_regret": 0.5,
                "counterfactual_top1_accuracy": 0.1,
                "prototype_drift": {"mean": 0.0},
                "actual_routing_rates": [1.0, 0.0],
                "counterfactual_win_rates": [1.0, 0.0],
                "training_evaluation_routing_consistency": {"mismatch": True},
            },
        }
    )
    codes = {warning["code"] for warning in warnings}
    assert {"basins_collapsed", "routing_regret_high", "expert_never_selected", "training_evaluation_routing_mismatch"} <= codes
