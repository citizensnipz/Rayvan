from __future__ import annotations

import json

import pytest
import torch

from rayvan_emc.checkpoint import load_model_checkpoint, save_training_checkpoint
from rayvan_emc.data import CharacterTokenizer
from rayvan_emc.model import EMCConfig, SequentialEMCModel
from rayvan_emc.research_config import ExperimentConfig, RoutingConfig
from rayvan_emc.research_runner import _build_model


def config(**overrides: object) -> EMCConfig:
    values: dict[str, object] = {
        "latent_dim": 8,
        "num_modules": 2,
        "modules_per_cycle": 1,
        "num_cycles": 3,
        "trajectory_steps": 3,
        "vocab_size": 12,
        "max_sequence_length": 8,
        "module_hidden_dim": 16,
        "attention_heads": 2,
        "integrator_heads": 2,
        "module_families": ("gpt", "recurrent"),
        "router_type": "module_aware",
        "integrator_type": "proposal_attention",
        "architecture_stage": "n1_sequential",
        "shared_state_slots": 2,
        "refractory_strength": 2.0,
        "refractory_decay": 0.25,
        "switch_cost": 0.0,
        "persistence_bonus": 0.0,
    }
    values.update(overrides)
    return EMCConfig(**values)


def test_sequential_emc_routes_once_per_updated_state(monkeypatch) -> None:
    model = SequentialEMCModel(config())
    seen_latents: list[torch.Tensor] = []
    original = model.router.route_one

    def observe(latent: torch.Tensor, **kwargs):
        seen_latents.append(latent.detach().clone())
        return original(latent, **kwargs)

    monkeypatch.setattr(model.router, "route_one", observe)
    output = model(torch.randint(0, 12, (2, 5)), return_trace=True)

    assert len(output.trace) == 3
    assert all(trace.selected_indices.shape == (2, 1, 1) for trace in output.trace)
    assert len(seen_latents) == 3
    assert not torch.equal(seen_latents[0], seen_latents[1])
    assert not torch.equal(seen_latents[1], seen_latents[2])


def test_sequential_path_never_calls_torch_topk(monkeypatch) -> None:
    model = SequentialEMCModel(config())
    monkeypatch.setattr(torch, "topk", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Top-K called")))
    output = model(torch.randint(0, 12, (1, 4)), return_trace=True)
    assert len(output.trace) == 3


def test_sequential_sparse_dispatch_accepts_bfloat16_expert_outputs() -> None:
    model = SequentialEMCModel(config())
    token_ids = torch.randint(0, 12, (2, 4))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(token_ids, return_trace=True)
    assert output.logits.shape == (2, 4, 12)
    assert torch.isfinite(output.logits.float()).all()


def test_refractory_penalty_decays_and_can_change_the_winner() -> None:
    model = SequentialEMCModel(config(router_type="fixed_index"))
    with torch.no_grad():
        model.router.score_projection.weight.zero_()
        model.router.score_projection.bias.zero_()
    output = model(torch.randint(0, 12, (1, 3)), return_trace=True)
    selected = [int(trace.selected_indices[0, 0, 0]) for trace in output.trace]
    assert selected == [0, 1, 0]
    assert output.trace[1].refractory_penalty[0, 0, 0].item() == pytest.approx(2.0)
    assert output.trace[2].refractory_penalty[0, 0, 0].item() == pytest.approx(0.5)


def test_repeat_selection_remains_possible_and_refractory_resets() -> None:
    model = SequentialEMCModel(config(num_modules=1, module_families=("gpt",)))
    first = model(torch.randint(0, 12, (1, 3)), return_trace=True)
    second = model(torch.randint(0, 12, (1, 3)), return_trace=True)
    assert all(int(trace.selected_indices[0, 0, 0]) == 0 for trace in first.trace)
    assert torch.count_nonzero(first.trace[0].refractory_penalty) == 0
    assert torch.count_nonzero(second.trace[0].refractory_penalty) == 0


def test_primary_and_legacy_architectures_remain_distinct() -> None:
    sequential = _build_model(ExperimentConfig(), 100)
    legacy = _build_model(
        ExperimentConfig(
            architecture="legacy_parallel_emc",
            routing=RoutingConfig(top_k=2),
        ),
        100,
    )
    assert isinstance(sequential, SequentialEMCModel)
    assert sequential.config.modules_per_cycle == 1
    assert legacy.config.architecture_stage == "n1_chunked"
    assert legacy.active_top_k == 2


def test_v1_config_is_not_silently_reinterpreted() -> None:
    payload = ExperimentConfig().to_dict()
    payload["schema_version"] = 1
    payload["routing"]["top_k"] = 2
    with pytest.raises(ValueError, match="parallel Top-K"):
        ExperimentConfig.from_dict(json.loads(json.dumps(payload)))


def test_checkpoint_preserves_sequential_type_and_balance_bias(tmp_path) -> None:
    model = SequentialEMCModel(config())
    with torch.no_grad():
        model.router.balance_bias.copy_(torch.tensor([0.125, -0.125]))
    optimizer = torch.optim.AdamW(model.parameters())
    tokenizer = CharacterTokenizer("\nabcdefghij")
    path = tmp_path / "sequential.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        step=1,
        tokens_processed=4,
        validation_loss=1.0,
        best_validation_loss=1.0,
        training_config={},
    )
    loaded = load_model_checkpoint(path)
    assert isinstance(loaded.model, SequentialEMCModel)
    torch.testing.assert_close(
        loaded.model.router.balance_bias, model.router.balance_bias
    )
