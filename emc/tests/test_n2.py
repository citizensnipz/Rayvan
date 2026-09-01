from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rayvan_emc.architecture import (
    architecture_accounting,
    build_architectures,
    deterministic_stream_fingerprint,
)
from rayvan_emc.architecture_compare import main as compare_architectures
from rayvan_emc.chunked import ChunkedEMCModel
from rayvan_emc.data import tiny_overfit_corpus
from rayvan_emc.diagnostics import EMCDiagnostics
from rayvan_emc.experiments.common import create_n2_model
from rayvan_emc.model import EMCOutput
from rayvan_emc.n2 import (
    DeltaN1Node,
    GPTN1Node,
    N2EMCModel,
    N2ExecutionTrace,
    RecurrentN1Node,
    SSMN1Node,
)
from rayvan_emc.training import TrainingConfig, train_model


torch.set_num_threads(1)


def n2_model(population: str = "mixed", *, seed: int = 42) -> N2EMCModel:
    return create_n2_model(
        64,
        "quick",
        maximum_sequence_length=16,
        seed=seed,
        population=population,
        top_k=2,
        n1_depth=3,
    )


def traced_forward(
    model: N2EMCModel,
    *,
    forced: torch.Tensor | None = None,
    zero: torch.Tensor | None = None,
    availability: torch.Tensor | None = None,
) -> EMCOutput:
    output = model(
        torch.randint(0, model.config.vocab_size, (1, 8)),
        return_trace=True,
        diagnostic_forced_modules=forced,
        diagnostic_zero_proposal_mask=zero,
        availability_mask=availability,
    )
    assert isinstance(output, EMCOutput)
    assert isinstance(output.chunk_trace, N2ExecutionTrace)
    return output


def test_n2_nexus_can_route_among_all_four_n1_nodes() -> None:
    model = n2_model()
    latent = torch.randn(1, 8, model.config.latent_dim)
    observed: set[int] = set()
    with torch.no_grad():
        model.nexus.score_projection.weight.zero_()
        for preferred in range(4):
            bias = torch.full((4,), -10.0)
            bias[preferred] = 10.0
            bias[(preferred + 1) % 4] = 9.0
            model.nexus.score_projection.bias.copy_(bias)
            routing = model.nexus(latent, top_k=2)
            observed.update(routing.selected_indices.reshape(-1).tolist())
            assert routing.pre_top_k_probabilities.shape == (1, 4)
            torch.testing.assert_close(
                routing.pre_top_k_probabilities.sum(dim=-1), torch.ones(1)
            )
            assert routing.selected_slots.tolist() == [[0, 1]]
    assert observed == {0, 1, 2, 3}


def test_top_k_two_executes_exactly_two_n1_nodes_per_request() -> None:
    model = n2_model()
    output = traced_forward(model)
    assert output.chunk_trace.selected_node_ids.shape == (1, 2)
    assert output.chunk_trace.actual_node_executions == 2
    assert sum(node.execution_count for node in model.n1_nodes) == 2


def test_unselected_n1_nodes_skip_expensive_local_computation() -> None:
    model = n2_model()
    traced_forward(model, forced=torch.tensor([0, 1]))
    assert [node.execution_count for node in model.n1_nodes] == [1, 1, 0, 0]


def test_selected_n1_nodes_receive_gradient_and_update_signal() -> None:
    model = n2_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = [
        [parameter.detach().clone() for parameter in node.parameters()]
        for node in model.n1_nodes
    ]
    output = traced_forward(model, forced=torch.tensor([0, 1]))
    output.logits.square().mean().backward()
    assert all(any(parameter.grad is not None for parameter in model.n1_nodes[i].parameters()) for i in (0, 1))
    assert all(all(parameter.grad is None for parameter in model.n1_nodes[i].parameters()) for i in (2, 3))
    optimizer.step()
    for index, node in enumerate(model.n1_nodes):
        changed = any(
            not torch.equal(old, current.detach())
            for old, current in zip(before[index], node.parameters(), strict=True)
        )
        assert changed is (index in {0, 1})


def test_same_family_n1_nodes_have_disjoint_parameters() -> None:
    model = n2_model("gpt4")
    parameter_ids = [
        {id(parameter) for parameter in node.parameters()}
        for node in model.n1_nodes
    ]
    for left in range(4):
        for right in range(left + 1, 4):
            assert parameter_ids[left].isdisjoint(parameter_ids[right])


def test_mixed_population_has_one_node_of_each_family() -> None:
    model = n2_model("mixed")
    assert tuple(type(node) for node in model.n1_nodes) == (
        GPTN1Node,
        SSMN1Node,
        RecurrentN1Node,
        DeltaN1Node,
    )
    assert model.module_families == ("gpt", "ssm", "recurrent", "delta")


def test_gpt4_has_four_independent_gpt_n1_nodes() -> None:
    model = n2_model("gpt4")
    assert all(type(node) is GPTN1Node for node in model.n1_nodes)
    assert model.expert_names == (
        "GPT-N1-A",
        "GPT-N1-B",
        "GPT-N1-C",
        "GPT-N1-D",
    )
    assert len({id(node) for node in model.n1_nodes}) == 4


@pytest.mark.parametrize(
    ("population", "node_type", "family"),
    (
        ("ssm4", SSMN1Node, "ssm"),
        ("recurrent4", RecurrentN1Node, "recurrent"),
        ("delta4", DeltaN1Node, "delta"),
    ),
)
def test_other_homogeneous_populations_have_four_independent_nodes(
    population: str, node_type: type, family: str
) -> None:
    model = n2_model(population)
    assert len(model.n1_nodes) == 4
    assert all(type(node) is node_type for node in model.n1_nodes)
    assert model.module_families == (family,) * 4
    assert len({id(node) for node in model.n1_nodes}) == 4


def test_n2_integrator_receives_only_selected_n1_proposals() -> None:
    model = n2_model()
    captured: list[tuple[int, ...]] = []

    def observe(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(tuple(args[1].shape))

    handle = model.integrator.register_forward_pre_hook(observe)
    try:
        traced_forward(model, forced=torch.tensor([1, 3]))
    finally:
        handle.remove()
    assert captured == [(1, 8, 2, model.config.latent_dim)]


def test_n2_causal_disable_zero_and_force_target_active_paths() -> None:
    torch.manual_seed(7)
    model = n2_model().eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 8))
    baseline = model(inputs, return_trace=True)
    assert isinstance(baseline, EMCOutput)
    baseline_ids = baseline.chunk_trace.selected_node_ids[0].tolist()
    disabled_id = baseline_ids[0]
    availability = torch.ones(4, dtype=torch.bool)
    availability[disabled_id] = False
    disabled = model(inputs, return_trace=True, availability_mask=availability)
    assert disabled_id not in disabled.chunk_trace.selected_node_ids

    zero_mask = torch.zeros(4, dtype=torch.bool)
    zero_mask[disabled_id] = True
    zeroed = model(
        inputs,
        return_trace=True,
        diagnostic_forced_modules=torch.tensor(baseline_ids),
        diagnostic_zero_proposal_mask=zero_mask,
    )
    selected = zeroed.trace[0].selected_indices[0, 0].tolist()
    zero_slot = selected.index(disabled_id)
    assert float(
        zeroed.trace[0].integrator_trace.proposal_norms[0, :, zero_slot].max()
    ) == 0.0

    alternative = next(index for index in range(4) if index not in baseline_ids)
    forced = model(
        inputs,
        return_trace=True,
        diagnostic_forced_modules=torch.tensor([alternative, baseline_ids[0]]),
    )
    assert alternative in forced.chunk_trace.selected_node_ids


def test_existing_emc_diagnostics_report_individual_n2_nodes() -> None:
    model = n2_model("gpt4")
    diagnostics = EMCDiagnostics(model)
    output = traced_forward(model, forced=torch.tensor([0, 1]))
    diagnostics.observe_trace(output.trace)
    output.logits.mean().backward()
    diagnostics.observe_router_gradients(model)
    diagnostics.observe_module_gradients(model)
    report = diagnostics.report(model)
    assert report.expert_names == model.expert_names
    assert report.traffic_fraction[0] == pytest.approx(0.5)
    assert report.traffic_fraction[1] == pytest.approx(0.5)
    assert report.average_integrator_acceptance[0] > 0


def test_n2_milestone_telemetry_and_checkpoint_retention_write_files(
    tmp_path: Path,
) -> None:
    corpus = tiny_overfit_corpus()
    model = create_n2_model(
        corpus.tokenizer.vocab_size,
        "quick",
        maximum_sequence_length=4,
        seed=42,
        population="mixed",
    )
    result = train_model(
        model,
        corpus,
        TrainingConfig(
            steps=None,
            train_tokens=4,
            batch_size=1,
            sequence_length=4,
            evaluation_interval=1,
            evaluation_batches=1,
            learning_rate=1e-3,
            checkpoint_directory=str(tmp_path),
            checkpoint_prefix="",
            retain_best_checkpoint=False,
            diagnostic_milestones=(4,),
            retain_milestone_checkpoints=True,
            collect_module_diagnostics=True,
            collect_milestone_telemetry=True,
            seed=42,
        ),
        print_progress=False,
    )
    assert result.milestone_checkpoints == (str(tmp_path / "4.pt"),)
    assert (tmp_path / "4.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()
    telemetry = json.loads((tmp_path / "telemetry.json").read_text(encoding="utf-8"))
    milestone = telemetry["milestones"][0]
    assert milestone["routing_unit"] == "request"
    assert len(milestone["modules"]) == 4
    assert milestone["modules"][0]["expert_name"] == "GPT-N1-A"
    assert sum(
        row["gradient_norm"] is not None for row in milestone["modules"]
    ) == 2


def test_old_granular_n1_emc_remains_runnable_and_unmodified() -> None:
    build = build_architectures(
        ("old_emc",),
        vocab_size=64,
        model_preset="quick",
        maximum_sequence_length=8,
        seed=42,
    )
    model = build.models["old_emc"]
    assert isinstance(model, ChunkedEMCModel)
    assert model.config.architecture_stage == "n1_chunked"
    assert model.module_families == ("gpt", "ssm", "recurrent", "delta")
    output = model(torch.randint(0, 64, (1, 8)))
    assert output.shape == (1, 8, 64)


def test_comparison_models_receive_identical_deterministic_token_streams() -> None:
    corpus = tiny_overfit_corpus()
    fingerprints = {
        deterministic_stream_fingerprint(
            corpus,
            TrainingConfig(
                steps=2,
                batch_size=1,
                sequence_length=4,
                seed=42,
            ),
        )
        for _ in range(7)
    }
    assert len(fingerprints) == 1


def test_n2_parameter_and_active_compute_accounting_is_explicit() -> None:
    model = n2_model("mixed")
    accounting = architecture_accounting(model, sequence_length=8)
    assert len(accounting.node_parameters) == 4
    assert max(accounting.node_parameters) / min(accounting.node_parameters) < 1.05
    assert len(accounting.node_flops_per_invocation) == 4
    assert accounting.blocks_per_n1 == 3
    assert accounting.selected_nodes_per_event == 2
    assert accounting.approximate_active_parameters < accounting.total_parameters
    assert (
        accounting.approximate_flops_per_token
        < accounting.theoretical_all_nodes_flops_per_token
    )
    assert accounting.nexus_parameters > 0
    assert accounting.integrator_parameters > 0
    assert accounting.embedding_parameters > 0


def test_tiny_seven_way_architecture_comparison_completes_without_large_training(
    tmp_path: Path,
) -> None:
    report = compare_architectures(
        [
            "--models",
            "homogeneous_serial,old_emc,n2_mixed,n2_gpt4,n2_ssm4,n2_recurrent4,n2_delta4",
            "--dataset",
            "tiny",
            "--model-preset",
            "quick",
            "--train-tokens",
            "4",
            "--batch-size",
            "1",
            "--sequence-length",
            "4",
            "--evaluation-batches",
            "1",
            "--diagnostic-checkpoints",
            "4",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "seven-way-smoke",
            "--device",
            "cpu",
        ]
    )
    assert report is not None
    assert set(report["configuration"]["models"]) == {
        "homogeneous_serial",
        "old_emc",
        "n2_mixed",
        "n2_gpt4",
        "n2_ssm4",
        "n2_recurrent4",
        "n2_delta4",
    }
    rows = report["runs"]
    assert len(rows) == 7
    assert all(row["tokens_processed"] == 4 for row in rows)
    assert len({row["stream_fingerprint"] for row in rows}) == 1
    assert (tmp_path / "seven-way-smoke" / "comparison.json").is_file()
