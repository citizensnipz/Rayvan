from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rayvan_emc.projections import fit_projection, perplexity_projection_payload, projection_payload
from rayvan_emc.research_config import ExperimentConfig, ModelConfig, ResearchTrainingConfig, RoutingConfig, research_schema
from rayvan_emc.research_runner import _system_metrics, run_experiment


def test_research_schema_uses_the_real_capability_registry() -> None:
    schema = research_schema()
    suite = next(row for row in schema["suites"] if row["id"] == "capability_10")
    assert suite["tasks"] == [
        "language",
        "associative_recall",
        "fuzzy_recall",
        "selective_copying",
        "working_memory",
        "compression",
        "arithmetic",
        "symbolic",
        "program_execution",
        "stateful_action",
    ]
    assert {row["id"] for row in schema["expert_families"]} == {"gpt", "ssm", "recurrent", "delta"}
    labels = {row["id"]: row["label"] for row in schema["architectures"]}
    assert labels["emc"] == "Sequential EMC — Geometric"
    assert labels["sequential_module_aware_emc"] == "Sequential EMC — Legacy Module-Aware"
    assert labels["legacy_parallel_emc"] == "Legacy Parallel Top-K EMC"


def test_experiment_config_round_trips_and_validates() -> None:
    original = ExperimentConfig(name="round-trip", tags=("n1", "smoke"))
    restored = ExperimentConfig.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.expert_families == ("gpt", "ssm", "recurrent", "delta")


def test_projection_selects_supported_curve_and_marks_long_extrapolation() -> None:
    points = [(x, 4.0 * x ** -0.2) for x in (10_000, 20_000, 40_000, 80_000, 100_000)]
    fit = fit_projection(points, 10_000_000, metric="validation_loss")
    assert fit is not None
    assert fit.model_type in {
        "power_law_decay",
        "power_law_asymptote",
        "exponential_decay",
        "exponential_asymptote",
    }
    assert fit.r_squared > 0.95
    assert fit.confidence == "low"
    assert fit.warning is not None


def test_projected_perplexity_is_derived_from_non_negative_loss() -> None:
    points = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
    loss_projection = projection_payload(
        points, (100.0, 1_000.0), metric="validation_loss"
    )
    perplexity = perplexity_projection_payload(loss_projection)
    assert perplexity["derivation"] == "exp(projected_validation_loss)"
    for loss_fit, ppl_fit in zip(
        loss_projection["fits"], perplexity["fits"], strict=True
    ):
        assert loss_fit["predicted_value"] >= 0
        assert ppl_fit["predicted_value"] > 0
        assert ppl_fit["predicted_value"] == pytest.approx(
            math.exp(loss_fit["predicted_value"])
        )


def test_missing_nvml_keeps_gpu_utilization_optional(monkeypatch) -> None:
    monkeypatch.setattr("rayvan_emc.research_runner.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("rayvan_emc.research_runner.torch.cuda.memory_allocated", lambda _device: 123)
    monkeypatch.setattr("rayvan_emc.research_runner.torch.cuda.mem_get_info", lambda _device: (100, 456))

    def missing_nvml(_device):
        raise ModuleNotFoundError("nvidia-ml-py is not installed")

    monkeypatch.setattr("rayvan_emc.research_runner.torch.cuda.utilization", missing_nvml)
    metrics = _system_metrics("cuda")
    assert metrics == {
        "gpu_utilization_percent": None,
        "vram_used_bytes": 123,
        "vram_total_bytes": 456,
    }


def test_tiny_cpu_run_streams_and_persists_full_run(tmp_path) -> None:
    config = ExperimentConfig(
        name="console-e2e",
        suite="capability_10",
        architecture="emc",
        experts={"gpt": 1, "ssm": 0, "recurrent": 0, "delta": 0},
        routing=RoutingConfig(
            trajectory_steps=1,
            counterfactual_probe_preset="fixed",
            counterfactual_probe_fixed_rate=1.0,
        ),
        model=ModelConfig(
            preset="custom",
            fairness_mode="custom",
            latent_dim=8,
            context_length=4,
            attention_heads=2,
            module_hidden_dim=16,
            integrator_heads=2,
            chunk_size=4,
            shared_state_slots=2,
            tie_embeddings=False,
        ),
        training=ResearchTrainingConfig(
            tokens=4,
            batch_size=1,
            learning_rate=1e-3,
            seed=7,
            gradient_accumulation=1,
            precision="fp32",
            device="cpu",
            evaluation_interval=1,
            evaluation_batches=1,
            telemetry_interval=1,
            diagnostic_examples_per_capability=1,
        ),
    )
    summary = run_experiment(config, runs_directory=tmp_path, run_id="tiny-e2e")
    run = tmp_path / "tiny-e2e"
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert summary["status"] == "completed"
    assert {event["type"] for event in events} >= {"run_started", "training_step", "validation", "diagnostic_result", "run_completed"}
    assert (run / "config.json").is_file()
    assert (run / "summary.json").is_file()
    assert (run / "geometric-routing.json").is_file()
    assert (run / "diagnostics" / "report.json").is_file()
    report = json.loads((run / "diagnostics" / "report.json").read_text())
    consistency = report["module_diagnostics"][
        "training_evaluation_routing_consistency"
    ]
    assert consistency["module_id_family_mapping_consistent"] is True
    geometry = summary["geometric_routing"]
    assert geometry["total_probes"] == 1
    assert "diagnostic_views" in geometry


def test_console_exposes_architecture_specific_routing_controls() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "research" / "ExperimentBuilder.tsx"
    ).read_text(encoding="utf-8")
    assert 'label="Trajectory steps"' in source
    assert "Experts per step" in source
    assert 'config.architecture === "legacy_parallel_emc"' in source
    assert 'label="Top-K"' in source
    assert 'architecture === "emc" || architecture === "sequential_module_aware_emc"' in source


def test_console_renders_sequential_trajectory_telemetry() -> None:
    source = (
        Path(__file__).parents[2]
        / "src"
        / "research"
        / "charts"
        / "RoutingOverview.tsx"
    ).read_text(encoding="utf-8")
    assert "Expert selection by trajectory step" in source
    assert "Expert transition matrix" in source
    assert "Refractory effect: raw vs inhibited winner" in source


def test_tiny_sequential_run_records_real_trajectory_events(tmp_path) -> None:
    config = ExperimentConfig(
        name="sequential-trajectory-smoke",
        suite="capability_10",
        architecture="emc",
        routing=RoutingConfig(
            trajectory_steps=3,
            refractory_strength=2.0,
            refractory_decay=0.25,
        ),
        model=ModelConfig(
            preset="custom",
            fairness_mode="custom",
            latent_dim=8,
            context_length=4,
            attention_heads=2,
            module_hidden_dim=16,
            integrator_heads=2,
            chunk_size=4,
            shared_state_slots=2,
            tie_embeddings=False,
        ),
        training=ResearchTrainingConfig(
            tokens=4,
            batch_size=1,
            learning_rate=1e-3,
            seed=9,
            gradient_accumulation=1,
            precision="fp32",
            device="cpu",
            evaluation_interval=1,
            evaluation_batches=1,
            telemetry_interval=1,
            diagnostic_examples_per_capability=1,
        ),
    )
    run_experiment(config, runs_directory=tmp_path, run_id="trajectory-e2e")
    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory-e2e" / "routing.jsonl")
        .read_text()
        .splitlines()
    ]
    routing = next(event for event in events if event["type"] == "routing_metrics")
    assert len(routing["trajectory_selection_counts"]) == 3
    assert sum(routing["selection_counts"]) == 3
    assert routing["refractory_observations"] == 3
    assert sum(sum(row) for row in routing["transitions"]) == 2
