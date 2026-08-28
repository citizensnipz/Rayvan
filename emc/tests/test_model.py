import torch
from torch import nn

from rayvan_emc import (
    EMCConfig,
    EMCModel,
    EMCModule,
    EMCOutput,
    Integrator,
    NexusRouter,
)


torch.set_num_threads(1)


def tiny_config(**overrides: int) -> EMCConfig:
    values = {
        "latent_dim": 8,
        "num_modules": 4,
        "modules_per_cycle": 2,
        "num_cycles": 3,
        "vocab_size": 19,
        "module_hidden_dim": 16,
        "attention_heads": 2,
    }
    values.update(overrides)
    return EMCConfig(**values)


def test_model_returns_language_model_shaped_logits() -> None:
    config = tiny_config()
    model = EMCModel(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(token_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 5, config.vocab_size)


def test_router_selects_exactly_configured_top_k_modules() -> None:
    config = tiny_config(modules_per_cycle=3)
    router = NexusRouter(config)
    latent = torch.randn(2, 5, config.latent_dim)

    routing = router(latent)
    expected_indices = torch.topk(routing.scores, k=3, dim=-1).indices

    assert routing.selected_indices.shape == (2, 5, 3)
    assert all(
        indices.unique().numel() == 3
        for indices in routing.selected_indices.reshape(-1, 3)
    )
    assert torch.equal(routing.selected_indices, expected_indices)
    torch.testing.assert_close(
        routing.selected_weights.sum(dim=-1), torch.ones(2, 5)
    )


def test_different_inputs_can_produce_different_routing_decisions() -> None:
    config = tiny_config(
        latent_dim=4,
        num_modules=4,
        modules_per_cycle=1,
        attention_heads=1,
    )
    router = NexusRouter(config)
    with torch.no_grad():
        router.score_projection.weight.copy_(torch.eye(4))
        router.score_projection.bias.zero_()

    first_latent = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    second_latent = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    first_choice = router(first_latent).selected_indices.item()
    second_choice = router(second_latent).selected_indices.item()

    assert first_choice == 0
    assert second_choice == 1


def test_selected_modules_have_independent_weights_and_shaped_updates() -> None:
    config = tiny_config()
    model = EMCModel(config)
    latent = torch.randn(2, 5, config.latent_dim)

    selected_indices = torch.tensor([0, 2]).expand(2, 5, 2)
    updates = model.execute_selected_modules(latent, selected_indices)
    first_parameters = {
        id(parameter) for parameter in model.emc_modules[0].parameters()
    }
    second_parameters = {
        id(parameter) for parameter in model.emc_modules[2].parameters()
    }

    assert updates.shape == (2, 5, 2, config.latent_dim)
    assert first_parameters.isdisjoint(second_parameters)


def test_integrator_preserves_shared_latent_shape() -> None:
    config = tiny_config()
    integrator = Integrator(config)
    latent = torch.randn(2, 5, config.latent_dim)
    module_updates = torch.randn(
        2, 5, config.modules_per_cycle, config.latent_dim
    )
    routing_weights = torch.softmax(
        torch.randn(2, 5, config.modules_per_cycle), dim=-1
    )

    integrated = integrator(latent, module_updates, routing_weights)

    assert integrated.shape == latent.shape


def test_shared_latent_circulates_through_every_cycle() -> None:
    config = tiny_config(num_cycles=4)
    model = EMCModel(config)
    router_inputs: list[torch.Tensor] = []
    integrated_states: list[torch.Tensor] = []

    router_hook = model.router.register_forward_pre_hook(
        lambda _module, args: router_inputs.append(args[0].detach().clone())
    )
    integrator_hook = model.integrator.register_forward_hook(
        lambda _module, _args, output: integrated_states.append(output.detach().clone())
    )
    result = model(
        torch.randint(0, config.vocab_size, (2, 5)), return_trace=True
    )
    router_hook.remove()
    integrator_hook.remove()

    assert isinstance(result, EMCOutput)
    assert len(result.trace) == config.num_cycles
    assert len(router_inputs) == config.num_cycles
    assert len(integrated_states) == config.num_cycles
    for cycle in range(1, config.num_cycles):
        torch.testing.assert_close(router_inputs[cycle], integrated_states[cycle - 1])
    for cycle_trace in result.trace:
        assert config.modules_per_cycle <= len(cycle_trace.selected_modules)
        assert len(cycle_trace.selected_modules) <= config.num_modules
        assert cycle_trace.router_scores.shape == (2, 5, config.num_modules)
        assert cycle_trace.router_weights.shape == (
            2,
            5,
            config.modules_per_cycle,
        )
        assert cycle_trace.selected_indices is not None
        assert cycle_trace.selected_indices.shape == (
            2,
            5,
            config.modules_per_cycle,
        )
        assert cycle_trace.latent_shape == (2, 5, config.latent_dim)


def test_gradients_propagate_through_end_to_end_model() -> None:
    config = tiny_config()
    model = EMCModel(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(token_ids)
    assert isinstance(logits, torch.Tensor)
    logits.square().mean().backward()

    def has_nonzero_finite_gradient(parameter: nn.Parameter) -> bool:
        gradient = parameter.grad
        return (
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and torch.count_nonzero(gradient).item() > 0
        )

    assert has_nonzero_finite_gradient(model.token_embedding.weight)
    assert has_nonzero_finite_gradient(model.router.score_projection.weight)
    assert any(
        has_nonzero_finite_gradient(parameter)
        for module in model.emc_modules
        for parameter in module.parameters()
    )
    assert has_nonzero_finite_gradient(model.integrator.update_projection.weight)
    assert has_nonzero_finite_gradient(model.output_projection.weight)


def test_seeded_models_produce_reproducible_output() -> None:
    config = tiny_config()
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    torch.manual_seed(2026)
    first_model = EMCModel(config).eval()
    torch.manual_seed(2026)
    second_model = EMCModel(config).eval()

    first_output = first_model(token_ids)
    second_output = second_model(token_ids)

    assert isinstance(first_output, torch.Tensor)
    assert isinstance(second_output, torch.Tensor)
    torch.testing.assert_close(first_output, second_output, rtol=0, atol=0)


def test_modules_only_receive_shared_latent_and_never_call_each_other() -> None:
    config = tiny_config()
    model = EMCModel(config)
    latent = torch.randn(2, 5, config.latent_dim)
    calls: list[tuple[int, int]] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for module_index, module in enumerate(model.emc_modules):
        hooks.append(
            module.register_forward_pre_hook(
                lambda _module, args, index=module_index: calls.append(
                    (index, id(args[0]))
                )
            )
        )

    selected_indices = torch.tensor([1, 3]).expand(2, 5, 2)
    updates = model.execute_selected_modules(latent, selected_indices)
    for hook in hooks:
        hook.remove()

    assert updates.shape == (2, 5, 2, config.latent_dim)
    assert calls == [(1, id(latent)), (3, id(latent))]
    assert all(isinstance(module, EMCModule) for module in model.emc_modules)
    assert all(isinstance(module, nn.Module) for module in model.emc_modules)
