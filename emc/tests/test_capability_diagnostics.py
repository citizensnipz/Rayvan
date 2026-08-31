from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rayvan_emc.capability_tasks import (
    CAPABILITIES,
    DEFAULT_MIXTURE_WEIGHTS,
    CapabilitySuiteConfig,
    CapabilityTaskSuite,
    apply_state_updates,
    diagnostic_tokenizer,
    evaluate_arithmetic_expression,
    evaluate_arithmetic_comparison,
    execute_restricted_program,
)
from rayvan_emc.checkpoint import save_training_checkpoint
from rayvan_emc.chunked import ChunkedEMCModel
from rayvan_emc.evaluate import (
    DiagnosticEvaluationConfig,
    _aggregate_capability_results,
    evaluate_suite,
    compare_checkpoints,
    write_report,
)
from rayvan_emc.model import EMCConfig, EMCModel, EMCOutput


torch.set_num_threads(1)


def token_config(**overrides: object) -> EMCConfig:
    values: dict[str, object] = {
        "latent_dim": 8,
        "num_modules": 4,
        "modules_per_cycle": 2,
        "num_cycles": 1,
        "vocab_size": diagnostic_tokenizer().vocab_size,
        "max_sequence_length": 1024,
        "module_hidden_dim": 16,
        "attention_heads": 2,
        "module_families": ("gpt", "ssm", "recurrent", "gpt"),
        "state_space_dim": 8,
        "recurrent_dim": 8,
        "router_type": "module_aware",
        "router_descriptor_dim": 4,
        "integrator_type": "proposal_attention",
        "integrator_heads": 2,
    }
    values.update(overrides)
    return EMCConfig(**values)


def chunk_config(**overrides: object) -> EMCConfig:
    values = dict(token_config().__dict__)
    values.update(
        {
            "architecture_stage": "n1_chunked",
            "module_families": ("gpt", "ssm", "recurrent", "delta"),
            "chunk_size": 8,
            "shared_state_slots": 2,
            "request_pool_size": 4,
            "active_top_k": 2,
            "delta_internal_dim": 8,
            "delta_heads": 2,
            "delta_ffn_dim": 16,
            "recurrent_precision": "model",
        }
    )
    values.update(overrides)
    return EMCConfig(**values)


def test_every_lane_is_reproducible_and_metadata_stays_outside_model_text() -> None:
    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=17))
    examples = {
        capability: suite.generate(capability, split="evaluation", index=index)
        for index, capability in enumerate(CAPABILITIES)
    }

    assert set(examples) == set(CAPABILITIES)
    for index, (capability, example) in enumerate(examples.items()):
        repeated = suite.generate(capability, split="evaluation", index=index)
        assert repeated == example
        assert example.model_text == example.prompt + example.target + "\n"
        assert example.diagnostic_metadata.capability == capability
        assert example.diagnostic_metadata.operation not in example.model_text
        training = suite.generate(capability, split="train", index=index)
        assert training.diagnostic_metadata.generator_seed != example.diagnostic_metadata.generator_seed


def test_cross_surface_tasks_keep_computation_and_target_constant() -> None:
    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=9))
    for capability in ("associative_recall", "working_memory"):
        examples = [
            suite.generate(
                capability,
                split="evaluation",
                index=5,
                surface=surface,
                difficulty=2,
            )
            for surface in ("english", "structured", "json", "code", "symbolic")
        ]
        assert len({example.target for example in examples}) == 1
        assert len({example.diagnostic_metadata.operation for example in examples}) == 1
        assert len({example.prompt for example in examples}) == 5


def test_held_out_combinations_are_excluded_only_from_training() -> None:
    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=3))
    with pytest.raises(ValueError, match="excluded"):
        suite.generate(
            "associative_recall", split="train", index=0, surface="json"
        )
    held_out = suite.generate(
        "associative_recall", split="evaluation", index=0, surface="json"
    )
    assert held_out.diagnostic_metadata.held_out_combination
    assert all(
        not example.diagnostic_metadata.held_out_combination
        for example in suite.mixed_examples(2_000, split="train")
    )


def test_default_mixer_preserves_computational_diversity() -> None:
    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=5))
    examples = suite.mixed_examples(10_000, split="train")
    counts = {capability: 0 for capability in CAPABILITIES}
    for example in examples:
        counts[example.diagnostic_metadata.capability] += 1
    language_token_fraction = sum(
        len(example.model_text)
        for example in examples
        if example.diagnostic_metadata.capability == "language"
    ) / sum(len(example.model_text) for example in examples)
    language_fraction = counts["language"] / len(examples)
    assert 0.15 <= language_fraction <= 0.25
    assert all(counts[capability] > 0 for capability in DEFAULT_MIXTURE_WEIGHTS)
    assert sum(DEFAULT_MIXTURE_WEIGHTS.values()) == pytest.approx(1.0)
    assert 0.15 <= language_token_fraction <= 0.25


def test_arithmetic_state_and_program_ground_truth_are_deterministic() -> None:
    assert evaluate_arithmetic_expression("(((3+4)*2)-5)") == 9
    assert apply_state_updates(3, [("add", 4), ("subtract", 2), ("set", 8)]) == (
        8,
        (7, 5, 8),
    )
    program = "x=3\nrepeat 2:\n x=x+4\nif x>10: x=x-1\nprint x"
    assert execute_restricted_program(program) == (10, (7, 11, 10))

    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=11))
    arithmetic = suite.generate(
        "arithmetic", split="evaluation", index=2, surface="symbolic", difficulty=4
    )
    expression = arithmetic.prompt.removesuffix("=")
    if arithmetic.diagnostic_metadata.operation == "arithmetic_comparison":
        expected = "true" if evaluate_arithmetic_comparison(expression) else "false"
        assert arithmetic.target == expected
    else:
        assert evaluate_arithmetic_expression(expression) == int(arithmetic.target)
    generated_program = suite.generate(
        "program_execution", split="evaluation", index=7, difficulty=3
    )
    program_text = generated_program.prompt.rsplit("\noutput=", 1)[0]
    assert execute_restricted_program(program_text)[0] == int(generated_program.target)


def test_difficulty_tiers_scale_computational_work() -> None:
    suite = CapabilityTaskSuite(CapabilitySuiteConfig(seed=29))
    for difficulty, expected in enumerate((2, 4, 8, 16), start=1):
        working_memory = suite.generate(
            "working_memory",
            split="evaluation",
            index=difficulty,
            surface="structured",
            difficulty=difficulty,
        )
        assert len(working_memory.intermediate_targets) == expected
    for difficulty, repeat_count in enumerate((1, 2, 4, 8), start=1):
        program = suite.generate(
            "program_execution",
            split="evaluation",
            index=difficulty,
            difficulty=difficulty,
        )
        assert len(program.intermediate_targets) == repeat_count + 1
    abstract = suite.generate(
        "symbolic",
        split="evaluation",
        index=4,
        surface="symbolic",
        difficulty=4,
    )
    assert abstract.diagnostic_metadata.operation == "abstract_symbol_rewrite"
    assert "→" in abstract.prompt


def test_recall_target_matches_generated_association() -> None:
    example = CapabilityTaskSuite(CapabilitySuiteConfig(seed=19)).generate(
        "associative_recall",
        split="evaluation",
        index=4,
        surface="structured",
        difficulty=2,
    )
    rows = example.prompt.splitlines()
    query = rows[-1].removeprefix("query ").removesuffix("=")
    mapping = dict(row.split(":", 1) for row in rows[:-1])
    assert example.target == mapping[query]
    keys = [row.split(":", 1)[0] for row in rows[:-1]]
    assert len(keys) > len(set(keys))
    targets = {
        CapabilityTaskSuite(CapabilitySuiteConfig(seed=19))
        .generate(
            "associative_recall",
            split="evaluation",
            index=index,
            surface="structured",
            difficulty=3,
        )
        .target
        for index in range(12)
    }
    assert len({len(target) for target in targets}) > 1


def test_token_interventions_disable_zero_and_force_intended_modules() -> None:
    model = EMCModel(token_config()).eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 12))
    availability = torch.tensor([False, True, True, True])
    disabled = model(inputs, return_trace=True, availability_mask=availability)
    assert isinstance(disabled, EMCOutput)
    assert 0 not in disabled.trace[0].selected_indices

    forced = model(
        inputs,
        return_trace=True,
        diagnostic_forced_modules=torch.tensor([0, 1]),
        diagnostic_zero_proposal_mask=torch.tensor([True, False, False, False]),
    )
    assert isinstance(forced, EMCOutput)
    assert torch.equal(
        forced.trace[0].selected_indices,
        torch.tensor([0, 1]).reshape(1, 1, 2).expand(1, 12, 2),
    )
    assert torch.count_nonzero(
        forced.trace[0].integrator_trace.proposal_norms[..., 0]
    ) == 0


def test_chunk_interventions_and_state_diagnostics_are_effective() -> None:
    model = ChunkedEMCModel(chunk_config()).eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 16))
    disabled = model(
        inputs,
        return_trace=True,
        availability_mask=torch.tensor([False, True, True, True]),
    )
    assert isinstance(disabled, EMCOutput)
    assert disabled.chunk_trace.request_pool.module_indices.size(-1) == 3
    assert all(
        0 not in chunk.active_modules for chunk in disabled.chunk_trace.chunks
    )

    forced = model(
        inputs,
        return_trace=True,
        diagnostic_forced_modules=torch.tensor([2, 3]),
        diagnostic_zero_proposal_mask=torch.tensor([False, False, True, False]),
    )
    assert isinstance(forced, EMCOutput)
    for chunk in forced.chunk_trace.chunks:
        assert torch.equal(chunk.active_modules, torch.tensor([[2, 3]]))
        assert torch.count_nonzero(chunk.token_integrator_trace.proposal_norms[..., 0]) == 0
        assert torch.count_nonzero(chunk.state_integrator_trace.proposal_norms[..., 0]) == 0
        assert chunk.lease_state_norm >= 0
        assert chunk.state_reset_count >= 0


def test_capability_metrics_use_exact_observable_values() -> None:
    records = [
        {
            "metadata": {"capability": "arithmetic"},
            "skipped": False,
            "exact_match": 1.0,
            "token_accuracy": 1.0,
            "loss": 0.2,
        },
        {
            "metadata": {"capability": "arithmetic"},
            "skipped": False,
            "exact_match": 0.0,
            "token_accuracy": 0.5,
            "loss": 0.6,
        },
    ]
    metrics = _aggregate_capability_results(records)["arithmetic"]
    assert metrics["exact_accuracy"] == 0.5
    assert metrics["token_accuracy"] == 0.75
    assert metrics["cross_entropy"] == pytest.approx(0.4)


def test_smoke_evaluation_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    tokenizer = diagnostic_tokenizer()
    model = EMCModel(
        token_config(
            num_modules=2,
            modules_per_cycle=1,
            module_families=("gpt", "recurrent"),
        )
    )
    report = evaluate_suite(
        model,
        tokenizer,
        DiagnosticEvaluationConfig(
            seed=23,
            examples_per_capability=1,
            ablation_examples_per_capability=0,
            run_module_ablations=False,
            run_family_ablations=False,
            run_zero_proposal=False,
            run_forced_alternatives=False,
            device="cpu",
            smoke=True,
            max_notable_examples=2,
        ),
    )
    write_report(report, tmp_path)

    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "report.md").is_file()
    loaded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert set(loaded["capability_results"]) == set(CAPABILITIES)
    assert "nexus_analysis" in loaded
    assert "causal_ablations" in loaded
    assert "overall_metrics" in loaded
    assert set(loaded["difficulty_curves"]) == set(CAPABILITIES)
    assert "routing_entropy" in loaded["difficulty_curves"]["arithmetic"]["1"]


def test_checkpoint_comparison_writes_side_by_side_metrics(tmp_path: Path) -> None:
    tokenizer = diagnostic_tokenizer()
    model = EMCModel(
        token_config(
            num_modules=2,
            modules_per_cycle=1,
            module_families=("gpt", "recurrent"),
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoints = []
    for index in range(2):
        checkpoint = tmp_path / f"model-{index}.pt"
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            step=index + 1,
            tokens_processed=10,
            validation_loss=1.0,
            best_validation_loss=1.0,
            training_config={"seed": 7},
        )
        checkpoints.append(checkpoint)
    comparison = compare_checkpoints(
        checkpoints,
        tmp_path / "comparison",
        DiagnosticEvaluationConfig(
            examples_per_capability=1,
            ablation_examples_per_capability=0,
            run_module_ablations=False,
            run_family_ablations=False,
            run_zero_proposal=False,
            run_forced_alternatives=False,
            device="cpu",
            smoke=True,
            max_notable_examples=0,
        ),
    )
    assert len(comparison["checkpoints"]) == 2
    assert set(comparison["capability_exact_accuracy"]) == set(CAPABILITIES)
    assert (tmp_path / "comparison" / "comparison.json").is_file()
    assert (tmp_path / "comparison" / "comparison.csv").is_file()
    assert (tmp_path / "comparison" / "comparison.md").is_file()
