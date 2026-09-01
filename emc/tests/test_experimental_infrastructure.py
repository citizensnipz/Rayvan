from __future__ import annotations

import json
from pathlib import Path

import torch

from rayvan_emc.architecture import (
    architecture_accounting,
    build_architectures,
    deterministic_stream_fingerprint,
)
from rayvan_emc.architecture_compare import TOKEN_PRESETS, main as compare_main
from rayvan_emc.baseline import TransformerConfig, TransformerLanguageModel
from rayvan_emc.checkpoint import load_model_checkpoint
from rayvan_emc.chunked import ChunkedEMCModel
from rayvan_emc.model import EMCConfig
from rayvan_emc.serial import HeterogeneousSerialModel
from rayvan_emc.training import (
    TrainingConfig,
    _module_gradient_norms,
    evaluate_model_metrics,
    train_model,
)
from rayvan_emc.data import tiny_overfit_corpus


torch.set_num_threads(1)


def chunk_config(**overrides: object) -> EMCConfig:
    corpus = tiny_overfit_corpus()
    values: dict[str, object] = {
        "latent_dim": 8,
        "num_modules": 4,
        "modules_per_cycle": 2,
        "num_cycles": 1,
        "vocab_size": corpus.tokenizer.vocab_size,
        "max_sequence_length": 8,
        "module_hidden_dim": 16,
        "attention_heads": 2,
        "tie_embeddings": False,
        "module_families": ("gpt", "ssm", "recurrent", "delta"),
        "state_space_dim": 8,
        "recurrent_dim": 8,
        "router_type": "module_aware",
        "router_descriptor_dim": 4,
        "integrator_type": "proposal_attention",
        "integrator_heads": 2,
        "architecture_stage": "n1_chunked",
        "chunk_size": 4,
        "shared_state_slots": 2,
        "request_pool_size": 4,
        "active_top_k": 2,
        "recurrent_precision": "model",
        "delta_internal_dim": 8,
        "delta_heads": 2,
        "delta_ffn_dim": 16,
        "loss_free_balance_enabled": False,
    }
    values.update(overrides)
    return EMCConfig(**values)


def tiny_training_config(directory: Path, **overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "steps": 2,
        "batch_size": 1,
        "sequence_length": 4,
        "evaluation_interval": 2,
        "evaluation_batches": 1,
        "checkpoint_directory": str(directory),
        "checkpoint_prefix": "",
        "diagnostic_milestones": (4, 8),
        "collect_module_diagnostics": True,
        "collect_milestone_telemetry": True,
        "seed": 13,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_milestone_checkpoints_latest_and_module_telemetry(tmp_path: Path) -> None:
    model = ChunkedEMCModel(chunk_config())
    result = train_model(
        model,
        tiny_overfit_corpus(),
        tiny_training_config(tmp_path),
        print_progress=False,
    )

    assert (tmp_path / "4.pt").exists()
    assert (tmp_path / "8.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    assert result.latest_checkpoint == str(tmp_path / "latest.pt")
    assert load_model_checkpoint(tmp_path / "latest.pt").progress.step == 2
    telemetry = json.loads((tmp_path / "telemetry.json").read_text())
    assert [row["milestone_tokens"] for row in telemetry["milestones"]] == [4, 8]
    assert len(telemetry["milestones"][0]["modules"]) == 4
    for module in telemetry["milestones"][0]["modules"]:
        assert "selection_frequency" in module
        assert "request_selection_fraction" in module
        assert "chunk_selection_fraction" in module
        assert "selection_slot_distribution" in module
        assert "mean_router_probability_before_top_k" in module
        assert "mean_normalized_selected_weight" in module
        assert "gradient_norm" in module
        assert "parameter_norm" in module
        assert "update_norm" in module
        assert "active_step_fraction_since_previous_milestone" in module
    assert (tmp_path / "telemetry.csv").exists()
    assert (tmp_path / "developmental-report.md").exists()
    assert (tmp_path / "plots" / "selection_frequency.svg").exists()


def test_milestone_retention_can_be_disabled(tmp_path: Path) -> None:
    train_model(
        ChunkedEMCModel(chunk_config()),
        tiny_overfit_corpus(),
        tiny_training_config(
            tmp_path,
            steps=1,
            diagnostic_milestones=(4,),
            retain_milestone_checkpoints=False,
        ),
        print_progress=False,
    )

    assert not (tmp_path / "4.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "telemetry.json").exists()


def test_unavailable_module_gradients_are_recorded_as_none() -> None:
    model = ChunkedEMCModel(chunk_config())
    assert _module_gradient_norms(model) == [None, None, None, None]


def test_milestone_causal_evaluation_uses_retained_checkpoint(
    tmp_path: Path, monkeypatch: object
) -> None:
    observed: list[Path] = []

    def fake_evaluate(checkpoint: Path, output: Path, config: object) -> dict[str, object]:
        del output, config
        observed.append(Path(checkpoint))
        loaded = load_model_checkpoint(checkpoint)
        assert loaded.progress.tokens_processed == 4
        return {
            "causal_ablations": {
                "performance_drop_when_family_removed": {
                    "working_memory": {
                        "ssm": {
                            "mean_loss_increase": -0.1,
                            "mean_token_accuracy_degradation": -0.2,
                            "fraction_measurably_worsened": 0.0,
                            "fraction_measurably_improved": 1.0,
                        }
                    }
                },
                "matrices": {"disable_module": {}},
                "specialization_status": "descriptive",
                "active_interventions_evaluated": 1,
            },
            "nexus_analysis": {
                "routing_frequency_by_capability": {
                    "working_memory": {"ssm": 0.75}
                },
                "module_frequency_by_capability": {},
                "router_probability_by_capability": {},
            },
            "router_collapse": {"status": "partial"},
        }

    monkeypatch.setattr("rayvan_emc.evaluate.evaluate_checkpoint", fake_evaluate)
    train_model(
        ChunkedEMCModel(chunk_config()),
        tiny_overfit_corpus(),
        tiny_training_config(
            tmp_path,
            steps=1,
            diagnostic_milestones=(4,),
            milestone_causal_diagnostics=True,
            milestone_causal_sample_size=1,
        ),
        print_progress=False,
    )

    assert observed == [tmp_path / "4.pt"]
    telemetry = json.loads((tmp_path / "telemetry.json").read_text())
    assert telemetry["milestones"][0]["causal"]["router_collapse_status"] == "partial"
    assert (tmp_path / "plots" / "causal_loss_impact.svg").exists()


def test_top_k_schedule_executes_three_then_returns_to_two(tmp_path: Path) -> None:
    model = ChunkedEMCModel(chunk_config())
    result = train_model(
        model,
        tiny_overfit_corpus(),
        tiny_training_config(
            tmp_path,
            steps=3,
            evaluation_interval=3,
            diagnostic_milestones=(4, 8, 12),
            top_k_schedule=((0, 3), (8, 2)),
        ),
        print_progress=False,
    )

    first = load_model_checkpoint(tmp_path / "4.pt")
    boundary = load_model_checkpoint(tmp_path / "8.pt")
    final = load_model_checkpoint(tmp_path / "12.pt")
    assert isinstance(first.model, ChunkedEMCModel)
    assert first.model.active_top_k == 3
    assert boundary.model.active_top_k == 3
    assert final.model.active_top_k == 2
    assert result.tokens_processed == 12

    inputs = torch.randint(0, first.model.config.vocab_size, (1, 4))
    output = first.model(inputs, return_trace=True)
    assert output.chunk_trace.chunks[0].active_modules.shape[-1] == 3
    assert output.chunk_trace.chunks[0].computed_chunk_module_pairs == 3
    final_output = final.model(inputs, return_trace=True)
    assert final_output.chunk_trace.chunks[0].active_modules.shape[-1] == 2


def test_top_k_transition_preserves_resume_and_optimizer_state(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    train_model(
        ChunkedEMCModel(chunk_config()),
        tiny_overfit_corpus(),
        tiny_training_config(
            first_directory,
            steps=2,
            diagnostic_milestones=(8,),
            top_k_schedule=((0, 3), (8, 2)),
        ),
        print_progress=False,
    )
    resumed_directory = tmp_path / "resumed"
    resumed = train_model(
        ChunkedEMCModel(chunk_config()),
        tiny_overfit_corpus(),
        tiny_training_config(
            resumed_directory,
            steps=3,
            diagnostic_milestones=(12,),
            top_k_schedule=((0, 3), (8, 2)),
            resume_from=str(first_directory / "8.pt"),
        ),
        print_progress=False,
    )
    assert resumed.steps_completed == 3
    assert load_model_checkpoint(resumed_directory / "12.pt").model.active_top_k == 2


def test_serial_models_train_evaluate_and_execute_configured_order(tmp_path: Path) -> None:
    corpus = tiny_overfit_corpus()
    homogeneous = TransformerLanguageModel(
        TransformerConfig(
            vocab_size=corpus.tokenizer.vocab_size,
            latent_dim=8,
            num_layers=1,
            attention_heads=2,
            feed_forward_dim=16,
            max_sequence_length=4,
        )
    )
    heterogeneous = HeterogeneousSerialModel(chunk_config(max_sequence_length=4))
    config = TrainingConfig(
        steps=1,
        batch_size=1,
        sequence_length=4,
        evaluation_interval=1,
        evaluation_batches=1,
        checkpoint_directory=str(tmp_path),
        diagnostic_milestones=(),
        seed=7,
    )

    homogeneous_result = train_model(
        homogeneous, corpus, config, print_progress=False
    )
    heterogeneous_result = train_model(
        heterogeneous, corpus, config, print_progress=False
    )
    loss, perplexity, accuracy = evaluate_model_metrics(
        heterogeneous, corpus, config
    )
    assert homogeneous_result.final_validation_loss > 0
    assert heterogeneous_result.final_validation_loss > 0
    assert loss > 0 and perplexity > 1 and 0 <= accuracy <= 1
    assert heterogeneous.last_execution_order == (
        "gpt",
        "ssm",
        "recurrent",
        "delta",
    )


def test_architecture_builder_uses_real_emc_and_reports_fairness() -> None:
    corpus = tiny_overfit_corpus()
    build = build_architectures(
        ("homogeneous_serial", "heterogeneous_serial", "emc"),
        vocab_size=corpus.tokenizer.vocab_size,
        model_preset="quick",
        maximum_sequence_length=8,
        seed=5,
        fairness_mode="capacity",
        tie_embeddings=False,
    )
    assert isinstance(build.models["emc"], ChunkedEMCModel)
    assert type(build.models["emc"].router).__name__ == "ChunkNexus"
    assert type(build.models["emc"].integrator).__name__ == "ChunkIntegrator"
    assert isinstance(build.models["heterogeneous_serial"], HeterogeneousSerialModel)
    for model in build.models.values():
        accounting = architecture_accounting(model, sequence_length=8)
        assert accounting.total_parameters > 0
        assert accounting.approximate_active_parameters > 0
        assert accounting.approximate_flops_per_token > 0
        assert accounting.module_computations_per_forward > 0
        assert accounting.method
        assert accounting.limitations


def test_identical_training_stream_fingerprint_for_every_architecture() -> None:
    corpus = tiny_overfit_corpus()
    config = TrainingConfig(
        steps=3,
        batch_size=2,
        sequence_length=4,
        evaluation_interval=1,
        evaluation_batches=1,
        seed=99,
        diagnostic_milestones=(),
    )
    fingerprints = [
        deterministic_stream_fingerprint(corpus, config) for _ in range(3)
    ]
    assert len(set(fingerprints)) == 1


def test_three_way_comparison_cli_completes_tiny_smoke(tmp_path: Path) -> None:
    report = compare_main(
        [
            "--dataset",
            "tiny",
            "--model-preset",
            "quick",
            "--train-tokens",
            "8",
            "--batch-size",
            "1",
            "--sequence-length",
            "8",
            "--evaluation-batches",
            "1",
            "--diagnostic-checkpoints",
            "8",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "smoke",
            "--device",
            "cpu",
            "--precision",
            "fp32",
        ]
    )

    assert report is not None
    assert set(report["aggregate"]) == {
        "homogeneous_serial",
        "heterogeneous_serial",
        "emc",
    }
    fingerprints = {row["stream_fingerprint"] for row in report["runs"]}
    assert len(fingerprints) == 1
    assert (tmp_path / "smoke" / "comparison.json").exists()
    assert (tmp_path / "smoke" / "comparison.csv").exists()
    assert (tmp_path / "smoke" / "comparison.md").exists()
    assert TOKEN_PRESETS["smoke"] == 25_000
    assert TOKEN_PRESETS["standard"] == 1_000_000
    assert TOKEN_PRESETS["follow-up"] == 5_000_000
