from __future__ import annotations


from pathlib import Path
import torch
from torch import Tensor, nn

from rayvan_emc import (
    ChunkGPTModule,
    ChunkEMCModuleBase,
    ChunkGatedDeltaNetModule,
    ChunkIntegrator,
    ChunkMetadata,
    ChunkNexus,
    ChunkRecurrentModule,
    ChunkStateSpaceModule,
    ChunkedEMCModel,
    EMCConfig,
    EMCModel,
    EMCOutput,
    ModuleCapabilities,
    ModuleInput,
    ModuleLeaseState,
    load_model_checkpoint,
    save_training_checkpoint,
    tiny_overfit_corpus,
    ModuleOutput,
)
from rayvan_emc.chunk_modules import (
    _parallel_affine_scan,
    _parallel_diagonal_scan,
)
from rayvan_emc.chunked import ChunkRoutingDecision
from rayvan_emc.experiments.common import create_emc_model


torch.set_num_threads(1)


def chunk_config(**overrides: object) -> EMCConfig:
    values: dict[str, object] = {
        "latent_dim": 8,
        "num_modules": 4,
        "modules_per_cycle": 2,
        "num_cycles": 2,
        "vocab_size": 31,
        "max_sequence_length": 16,
        "module_hidden_dim": 16,
        "attention_heads": 2,
        "tie_embeddings": False,
        "module_families": ("gpt", "ssm", "recurrent", "delta"),
        "state_space_dim": 8,
        "state_space_kernel_size": 3,
        "recurrent_dim": 8,
        "router_descriptor_dim": 4,
        "integrator_heads": 2,
        "architecture_stage": "n1_chunked",
        "chunk_size": 4,
        "shared_state_slots": 2,
        "request_pool_size": 4,
        "switch_cost": 0.05,
        "persistence_bonus": 0.1,
        "loss_free_balance_enabled": True,
        "balance_bias_lr": 0.05,
        "balance_bias_limit": 0.25,
        "shared_core_hidden_dim": 8,
        "recurrent_precision": "model",
        "delta_internal_dim": 8,
        "delta_heads": 2,
        "delta_ffn_dim": 16,
    }
    values.update(overrides)
    return EMCConfig(**values)


class CountingChunkModule(ChunkEMCModuleBase):
    def __init__(self, config: EMCConfig, family: str, value: float) -> None:
        super().__init__(config)
        self.family = family
        self.value = value
        self.forward_calls = 0
        self.begin_calls = 0
        self.end_calls = 0
        self.input_counters: list[Tensor] = []
        self.shared_inputs: list[Tensor] = []

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(self.family, 1, 1, "model", "counting")

    def begin_lease(self, shared_state: Tensor) -> ModuleLeaseState:
        self.begin_calls += 1
        return ModuleLeaseState(
            {"counter": shared_state.new_zeros(shared_state.size(0), 1)}
        )

    def end_lease(self, _lease_state: ModuleLeaseState) -> None:
        self.end_calls += 1

    def forward_chunk(self, module_input: ModuleInput) -> ModuleOutput:
        self.forward_calls += 1
        counter = module_input.lease_state.tensors["counter"]
        self.input_counters.append(counter.detach().clone())
        self.shared_inputs.append(module_input.shared_state.detach().clone())
        token = torch.ones_like(module_input.chunk_latent) * self.value
        state = torch.ones_like(module_input.shared_state) * self.value
        return ModuleOutput(
            token_proposal=token,
            state_proposal=state,
            new_lease_state=ModuleLeaseState({"counter": counter + 1}),
        )


class ScheduledNexus(ChunkNexus):
    def __init__(
        self,
        config: EMCConfig,
        schedule: list[list[int]],
        pool: list[int],
    ) -> None:
        super().__init__(config)
        self.schedule = schedule
        self.pool = pool
        self.call_index = 0

    def select_request_pool(
        self,
        first_token: Tensor,
        shared_state: Tensor,
        availability_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del shared_state, availability_mask
        batch = first_token.size(0)
        scores = first_token.new_zeros(batch, self.config.num_modules)
        pool = torch.tensor(self.pool, device=first_token.device).expand(batch, -1)
        return scores, pool

    def route_chunk(
        self,
        first_token: Tensor,
        shared_state: Tensor,
        pool_mask: Tensor,
        request_scores: Tensor,
        previous_active: Tensor,
        lease_ages: Tensor,
        availability_mask: Tensor | None = None,
    ) -> ChunkRoutingDecision:
        del shared_state, pool_mask, request_scores, availability_mask
        batch = first_token.size(0)
        selected = torch.tensor(
            self.schedule[self.call_index], device=first_token.device
        ).expand(batch, -1)
        self.call_index += 1
        scores = first_token.new_zeros(batch, self.config.num_modules)
        weights = first_token.new_full(
            (batch, self.config.modules_per_cycle),
            1.0 / self.config.modules_per_cycle,
        )
        persistence = previous_active.to(first_token.dtype) * self.config.persistence_bonus
        switching = (~previous_active).to(first_token.dtype) * (-self.config.switch_cost)
        return ChunkRoutingDecision(
            scores,
            selected,
            weights,
            persistence,
            switching,
        )


class CanonicalStateIntegrator(nn.Module):
    def forward(
        self,
        chunk_latent: Tensor,
        shared_state: Tensor,
        token_proposals: Tensor,
        state_proposals: Tensor,
        routing_weights: Tensor,
        *,
        return_diagnostics: bool,
    ):
        del routing_weights, return_diagnostics
        return (
            chunk_latent + token_proposals.sum(dim=1),
            shared_state + state_proposals.sum(dim=1),
            None,
            None,
        )


def module_input(config: EMCConfig, batch: int = 2, length: int = 4) -> ModuleInput:
    shared = torch.randn(batch, config.shared_state_slots, config.latent_dim)
    return ModuleInput(
        chunk_latent=torch.randn(batch, length, config.latent_dim),
        shared_state=shared,
        lease_state=ModuleLeaseState(),
        metadata=ChunkMetadata(
            request_indices=torch.arange(batch),
            chunk_index=0,
            lease_ages=torch.ones(batch, dtype=torch.long),
            module_index=0,
            lease_ids=torch.stack(
                (
                    torch.arange(batch),
                    torch.zeros(batch, dtype=torch.long),
                    torch.ones(batch, dtype=torch.long),
                ),
                dim=-1,
            ),
            continuing_lease=torch.zeros(batch, dtype=torch.bool),
        ),
    )


def test_chunked_model_is_causal_across_and_within_chunks() -> None:
    torch.manual_seed(2)
    model = ChunkedEMCModel(chunk_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = torch.tensor([[1, 2, 9, 10, 11, 12, 13, 14]])

    first_logits = model(first)
    second_logits = model(second)

    assert isinstance(first_logits, Tensor)
    assert isinstance(second_logits, Tensor)
    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])


def test_chunked_model_dispatches_bfloat16_proposals_into_canonical_buffers() -> None:
    model = ChunkedEMCModel(chunk_config()).train()
    inputs = torch.randint(0, model.config.vocab_size, (1, 8))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(inputs)
        assert isinstance(logits, Tensor)
        loss = logits.float().square().mean()
    loss.backward()

    assert torch.isfinite(logits).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_only_selected_modules_execute_and_sparse_pairs_match() -> None:
    config = chunk_config(
        modules_per_cycle=2,
        request_pool_size=2,
        module_families=("gpt", "ssm", "recurrent", "delta"),
    )
    modules = [
        CountingChunkModule(config, family, index + 1.0)
        for index, family in enumerate(config.resolved_module_families)
    ]
    nexus = ScheduledNexus(config, [[0, 1], [0, 1]], [0, 1])
    model = ChunkedEMCModel(config, modules=modules, nexus=nexus)

    output = model(torch.randint(0, config.vocab_size, (1, 8)), return_trace=True)

    assert isinstance(output, EMCOutput)
    assert modules[0].forward_calls == 2
    assert modules[1].forward_calls == 2
    assert modules[2].forward_calls == 0
    assert modules[3].forward_calls == 0
    assert output.chunk_trace is not None
    for chunk in output.chunk_trace.chunks:
        assert chunk.computed_chunk_module_pairs == 2
        assert chunk.retained_chunk_module_pairs == 2
        assert set(chunk.executed_modules) == {0, 1}


def test_request_pool_and_active_top_k_are_honored() -> None:
    config = chunk_config(request_pool_size=2, modules_per_cycle=1)
    nexus = ScheduledNexus(config, [[1], [1]], [1, 3])
    model = ChunkedEMCModel(config, nexus=nexus)

    output = model(torch.randint(0, config.vocab_size, (1, 8)), return_trace=True)

    assert isinstance(output, EMCOutput)
    trace = output.chunk_trace
    assert trace is not None
    assert trace.request_pool.module_indices.shape == (1, 2)
    assert set(trace.request_pool.module_indices[0].tolist()) == {1, 3}
    assert all(chunk.active_modules.shape[-1] == 1 for chunk in trace.chunks)
    assert all(set(chunk.active_modules.flatten().tolist()) <= {1, 3} for chunk in trace.chunks)


def test_lease_persists_across_contiguous_chunks() -> None:
    config = chunk_config(num_modules=2, modules_per_cycle=1, request_pool_size=2, module_families=("gpt", "ssm"))
    modules = [
        CountingChunkModule(config, family, index + 1.0)
        for index, family in enumerate(config.resolved_module_families)
    ]
    nexus = ScheduledNexus(config, [[0], [0], [0]], [0, 1])
    model = ChunkedEMCModel(config, modules=modules, nexus=nexus)

    output = model(torch.randint(0, config.vocab_size, (1, 12)), return_trace=True)

    assert isinstance(output, EMCOutput)
    assert modules[0].begin_calls == 1
    assert modules[0].forward_calls == 3
    assert [counter.item() for counter in modules[0].input_counters] == [0, 1, 2]
    assert output.chunk_trace is not None
    assert [chunk.lease_ages[0, 0].item() for chunk in output.chunk_trace.chunks] == [1, 2, 3]


def test_inactive_gap_discards_and_reinitializes_private_state() -> None:
    config = chunk_config(num_modules=2, modules_per_cycle=1, request_pool_size=2, module_families=("gpt", "ssm"))
    modules = [
        CountingChunkModule(config, family, index + 1.0)
        for index, family in enumerate(config.resolved_module_families)
    ]
    nexus = ScheduledNexus(config, [[0], [1], [0]], [0, 1])
    model = ChunkedEMCModel(config, modules=modules, nexus=nexus)

    output = model(
        torch.randint(0, config.vocab_size, (1, 12)),
        return_trace=True,
    )

    assert modules[0].begin_calls == 2
    assert modules[0].end_calls == 1
    assert [counter.item() for counter in modules[0].input_counters] == [0, 0]
    assert isinstance(output, EMCOutput)
    assert output.chunk_trace is not None
    assert output.chunk_trace.chunks[0].lease_generations[0, 0].item() == 1
    assert output.chunk_trace.chunks[2].lease_generations[0, 0].item() == 2


def test_canonical_shared_state_survives_module_switching() -> None:
    config = chunk_config(num_modules=2, modules_per_cycle=1, request_pool_size=2, module_families=("gpt", "ssm"))
    modules = [
        CountingChunkModule(config, family, index + 1.0)
        for index, family in enumerate(config.resolved_module_families)
    ]
    nexus = ScheduledNexus(config, [[0], [1], [0]], [0, 1])
    model = ChunkedEMCModel(config, modules=modules, nexus=nexus)
    model.integrator = CanonicalStateIntegrator()

    model(torch.randint(0, config.vocab_size, (1, 12)))

    assert len(modules[0].shared_inputs) == 2
    first_shared, resumed_shared = modules[0].shared_inputs
    assert not torch.equal(first_shared, resumed_shared)
    torch.testing.assert_close(resumed_shared, first_shared + 3.0)


def test_all_chunk_module_families_follow_common_contract() -> None:
    config = chunk_config()
    for module_class in (
        ChunkGPTModule,
        ChunkStateSpaceModule,
        ChunkRecurrentModule,
        ChunkGatedDeltaNetModule,
    ):
        module = module_class(config)
        input_value = module_input(config)
        input_value.lease_state.tensors.update(
            module.begin_lease(input_value.shared_state).tensors
        )
        output = module.forward_chunk(input_value)
        assert output.token_proposal.shape == input_value.chunk_latent.shape
        assert output.state_proposal.shape == input_value.shared_state.shape
        assert isinstance(output.new_lease_state, ModuleLeaseState)


def test_chunk_integrator_receives_individual_token_and_state_proposals() -> None:
    config = chunk_config()
    integrator = ChunkIntegrator(config)
    chunk = torch.randn(1, 4, config.latent_dim)
    shared = torch.randn(1, config.shared_state_slots, config.latent_dim)
    token_proposals = torch.stack(
        (torch.ones_like(chunk), -torch.ones_like(chunk)), dim=1
    ).requires_grad_()
    state_proposals = torch.stack(
        (torch.ones_like(shared), -torch.ones_like(shared) * 2), dim=1
    ).requires_grad_()
    weights = torch.full((1, 2), 0.5)

    result = integrator(
        chunk,
        shared,
        token_proposals,
        state_proposals,
        weights,
        return_diagnostics=True,
    )
    updated_chunk, updated_state, token_trace, state_trace = result
    (updated_chunk.square().mean() + updated_state.square().mean()).backward()

    assert token_trace is not None and state_trace is not None
    assert token_trace.proposal_acceptance.shape[-1] == 2
    assert state_trace.proposal_acceptance.shape[-1] == 2
    assert token_proposals.grad is not None
    assert state_proposals.grad is not None
    assert torch.count_nonzero(token_proposals.grad[:, 0]).item() > 0
    assert torch.count_nonzero(token_proposals.grad[:, 1]).item() > 0
    assert torch.count_nonzero(state_proposals.grad[:, 0]).item() > 0
    assert torch.count_nonzero(state_proposals.grad[:, 1]).item() > 0


def _assert_module_gradients(
    module: ChunkEMCModuleBase, config: EMCConfig
) -> None:
    value = module_input(config, batch=1, length=4)
    value.lease_state.tensors.update(
        module.begin_lease(value.shared_state).tensors
    )
    output = module.forward_chunk(value)
    loss = (
        output.token_proposal.square().mean()
        + output.state_proposal.square().mean()
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_gated_deltanet_produces_valid_gradients() -> None:
    _assert_module_gradients(ChunkGatedDeltaNetModule(chunk_config()), chunk_config())


def test_chunkwise_ssm_produces_valid_gradients() -> None:
    _assert_module_gradients(ChunkStateSpaceModule(chunk_config()), chunk_config())


def test_chunk_recurrent_module_produces_valid_gradients() -> None:
    _assert_module_gradients(ChunkRecurrentModule(chunk_config()), chunk_config())


def test_chunked_checkpoint_round_trip(tmp_path: Path) -> None:
    corpus = tiny_overfit_corpus()
    config = chunk_config(
        vocab_size=corpus.tokenizer.vocab_size,
        max_sequence_length=8,
    )
    model = ChunkedEMCModel(config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    expected = model(tokens)
    checkpoint = tmp_path / "chunked.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        tokenizer=corpus.tokenizer,
        step=1,
        tokens_processed=8,
        validation_loss=2.0,
        best_validation_loss=2.0,
        training_config={},
    )

    loaded = load_model_checkpoint(checkpoint)
    loaded.model.eval()
    actual = loaded.model(tokens)

    assert isinstance(loaded.model, ChunkedEMCModel)
    torch.testing.assert_close(expected, actual)


def test_parallel_ssm_scan_matches_sequential_recurrence() -> None:
    torch.manual_seed(3)
    log_decay = -torch.rand(2, 5, 4)
    candidate = torch.randn(2, 5, 4)
    initial = torch.randn(2, 4)
    parallel = _parallel_diagonal_scan(log_decay, candidate, initial)
    state = initial
    expected: list[Tensor] = []
    for token in range(log_decay.size(1)):
        decay = torch.exp(log_decay[:, token])
        state = decay * state + (1.0 - decay) * candidate[:, token]
        expected.append(state)

    torch.testing.assert_close(parallel, torch.stack(expected, dim=1))


def test_parallel_ssm_scan_masks_noncausal_terms_before_exponentiation() -> None:
    log_decay = torch.full((1, 64, 4), -100.0, requires_grad=True)
    candidate = torch.randn(1, 64, 4, requires_grad=True)
    initial = torch.randn(1, 4, requires_grad=True)

    states = _parallel_diagonal_scan(log_decay, candidate, initial)
    states.square().mean().backward()

    assert torch.isfinite(states).all()
    assert torch.isfinite(log_decay.grad).all()
    assert torch.isfinite(candidate.grad).all()
    assert torch.isfinite(initial.grad).all()


def test_parallel_delta_affine_scan_matches_sequential_composition() -> None:
    torch.manual_seed(4)
    transition = torch.randn(1, 5, 2, 3, 3) * 0.05
    transition = transition + torch.eye(3).reshape(1, 1, 1, 3, 3)
    write = torch.randn(1, 5, 2, 3, 3) * 0.05
    parallel_transition, parallel_write = _parallel_affine_scan(
        transition, write
    )
    expected_transition = transition[:, 0]
    expected_write = write[:, 0]
    transition_prefixes = [expected_transition]
    write_prefixes = [expected_write]
    for token in range(1, transition.size(1)):
        expected_write = (
            torch.matmul(expected_write, transition[:, token])
            + write[:, token]
        )
        expected_transition = torch.matmul(
            expected_transition, transition[:, token]
        )
        transition_prefixes.append(expected_transition)
        write_prefixes.append(expected_write)

    torch.testing.assert_close(
        parallel_transition, torch.stack(transition_prefixes, dim=1)
    )
    torch.testing.assert_close(
        parallel_write, torch.stack(write_prefixes, dim=1)
    )


def test_chunk_nexus_descriptors_receive_routing_gradients() -> None:
    config = chunk_config()
    nexus = ChunkNexus(config)
    first = torch.randn(2, config.latent_dim)
    shared = torch.randn(2, config.shared_state_slots, config.latent_dim)
    _, pool = nexus.select_request_pool(first, shared)
    pool_mask = torch.zeros(2, config.num_modules, dtype=torch.bool)
    pool_mask.scatter_(1, pool, True)
    decision = nexus.route_chunk(
        first,
        shared,
        pool_mask,
        torch.zeros_like(pool_mask, dtype=first.dtype),
        torch.zeros_like(pool_mask),
        torch.zeros_like(pool_mask, dtype=torch.long),
    )

    decision.scores[torch.isfinite(decision.scores)].sum().backward()

    assert nexus.module_descriptors.grad is not None
    assert torch.count_nonzero(nexus.module_descriptors.grad).item() > 0


def test_chunk_router_persistence_bonus_retains_active_module() -> None:
    config = chunk_config(
        modules_per_cycle=1,
        active_top_k=1,
        persistence_bonus=1.0,
        switch_cost=0.1,
    )
    nexus = ChunkNexus(config)
    with torch.no_grad():
        nexus.chunk_query.weight.zero_()
        nexus.chunk_query.bias.zero_()
        nexus.module_descriptors.zero_()
    first = torch.randn(1, config.latent_dim)
    shared = torch.randn(1, config.shared_state_slots, config.latent_dim)
    pool_mask = torch.ones(1, config.num_modules, dtype=torch.bool)
    previous = torch.zeros_like(pool_mask)
    previous[:, 2] = True
    decision = nexus.route_chunk(
        first,
        shared,
        pool_mask,
        torch.zeros(1, config.num_modules),
        previous,
        torch.ones_like(previous, dtype=torch.long),
    )

    assert decision.selected_indices.item() == 2
    assert decision.persistence_contribution[0, 2].item() == 1.0


def test_loss_free_balance_bias_is_not_in_model_gradient() -> None:
    config = chunk_config()
    nexus = ChunkNexus(config)
    with torch.no_grad():
        nexus.chunk_query.weight.zero_()
        nexus.chunk_query.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        nexus.module_descriptors.zero_()
        nexus.module_descriptors[:, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    first = torch.randn(2, config.latent_dim)
    shared = torch.randn(2, config.shared_state_slots, config.latent_dim)
    request_scores, pool = nexus.select_request_pool(first, shared)
    pool_mask = torch.zeros(2, config.num_modules, dtype=torch.bool)
    pool_mask.scatter_(1, pool, True)
    before = nexus.balance_bias.clone()
    decision = nexus.route_chunk(
        first,
        shared,
        pool_mask,
        request_scores,
        torch.zeros_like(pool_mask),
        torch.zeros_like(pool_mask, dtype=torch.long),
    )
    decision.scores[torch.isfinite(decision.scores)].sum().backward()

    assert not torch.equal(before, nexus.balance_bias)
    assert nexus.balance_bias.requires_grad is False
    assert nexus.balance_bias.grad is None
    assert "balance_bias" not in dict(nexus.named_parameters())


def test_request_pool_query_receives_end_to_end_gradients() -> None:
    model = ChunkedEMCModel(chunk_config())
    output = model(
        torch.randint(0, model.config.vocab_size, (2, 8)),
        return_trace=True,
    )
    assert isinstance(output, EMCOutput)
    output.logits.square().mean().backward()

    assert model.router.request_query.weight.grad is not None
    assert torch.count_nonzero(model.router.request_query.weight.grad).item() > 0
    assert model.router.chunk_query.weight.grad is not None
    assert torch.count_nonzero(model.router.chunk_query.weight.grad).item() > 0


def test_legacy_stage_remains_reproducible() -> None:
    torch.manual_seed(12)
    first = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=8,
        seed=12,
        n1_stage="n1",
        module_population="mixed",
    ).eval()
    second = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=8,
        seed=12,
        n1_stage="n1",
        module_population="mixed",
    ).eval()
    chunked = create_emc_model(
        31,
        "quick",
        maximum_sequence_length=8,
        seed=12,
        n1_stage="n1_chunked",
        module_population="mixed",
    )
    tokens = torch.randint(0, 31, (1, 8))

    assert isinstance(first, EMCModel)
    assert isinstance(second, EMCModel)
    assert isinstance(chunked, ChunkedEMCModel)
    torch.testing.assert_close(first(tokens), second(tokens), rtol=0, atol=0)
