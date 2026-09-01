from __future__ import annotations

import math
from contextlib import nullcontext

import pytest
import torch
from torch import Tensor

from rayvan_emc.experiments.common import create_n2_model
from rayvan_emc.n2 import N1Input, N2EMCModel, _force_n2_routing
from rayvan_emc.training import next_token_loss


def _precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _reference_dispatch(
    model: N2EMCModel, latent: Tensor, selected_indices: Tensor
) -> Tensor:
    batch, sequence, latent_dim = latent.shape
    selected = selected_indices.size(1)
    flattened = latent.new_zeros(batch * selected, sequence, latent_dim)
    for node_id, node in enumerate(model.n1_nodes):
        locations = torch.nonzero(selected_indices == node_id, as_tuple=False)
        if locations.numel() == 0:
            continue
        request_rows = locations[:, 0]
        node_output = node(
            N1Input(
                shared_latent=latent.index_select(0, request_rows),
                request_indices=request_rows,
            )
        )
        flat_locations = request_rows * selected + locations[:, 1]
        node_buffer = latent.new_zeros(batch * selected, sequence, latent_dim)
        flattened = flattened + node_buffer.index_copy(
            0, flat_locations, node_output.proposal
        )
    return flattened.reshape(
        batch, selected, sequence, latent_dim
    ).permute(0, 2, 1, 3)


def _reference_integrator(
    model: N2EMCModel,
    latent: Tensor,
    proposals: Tensor,
    selected_weights: Tensor,
) -> Tensor:
    integrator = model.integrator.proposal_integrator
    batch, sequence, selected, latent_dim = proposals.shape
    routing_weights = selected_weights.unsqueeze(1).expand(-1, sequence, -1)
    normalized_proposals = integrator.proposal_norm(proposals)
    query = integrator.query_projection(integrator.latent_norm(latent)).reshape(
        batch, sequence, integrator.num_heads, integrator.head_dim
    )
    keys = integrator.key_projection(normalized_proposals).reshape(
        batch, sequence, selected, integrator.num_heads, integrator.head_dim
    ).permute(0, 1, 3, 2, 4)
    values = integrator.value_projection(normalized_proposals).reshape(
        batch, sequence, selected, integrator.num_heads, integrator.head_dim
    ).permute(0, 1, 3, 2, 4)
    attention_scores = torch.einsum("bshd,bshkd->bshk", query, keys)
    attention_scores = attention_scores / math.sqrt(integrator.head_dim)
    attention_scores = attention_scores + integrator.routing_prior_scale * (
        routing_weights.clamp_min(1e-9).log().unsqueeze(2)
    )
    acceptance = torch.softmax(attention_scores, dim=-1)
    attended_heads = torch.einsum("bshk,bshkd->bshd", acceptance, values)
    attended = integrator.attention_output(
        attended_heads.reshape(batch, sequence, latent_dim)
    )
    integration_input = torch.cat(
        (
            integrator.latent_norm(latent),
            attended,
            proposals.mean(dim=2),
            proposals.var(dim=2, unbiased=False),
        ),
        dim=-1,
    )
    candidate = integrator.update_projection(integration_input)
    gate = torch.sigmoid(integrator.gate_projection(integration_input))
    return latent + gate * candidate


def _staged_forward(
    model: N2EMCModel,
    token_ids: Tensor,
    forced: Tensor,
    *,
    optimized: bool,
) -> dict[str, Tensor]:
    positions = torch.arange(token_ids.size(1), device=token_ids.device)
    latent = model.token_embedding(token_ids) + model.position_embedding(positions)
    natural_routing = model.router(latent, top_k=model.active_top_k)
    routing = _force_n2_routing(
        natural_routing,
        forced,
        batch=token_ids.size(0),
        top_k=model.active_top_k,
        num_nodes=model.config.num_modules,
    )
    if optimized:
        proposals, _, _, _, _ = model._execute_selected_nodes(
            latent, routing.selected_indices, n2_state=None
        )
        integrated = model.integrator(
            latent, proposals, routing.selected_weights
        )
        assert isinstance(integrated, Tensor)
    else:
        proposals = _reference_dispatch(model, latent, routing.selected_indices)
        integrated = _reference_integrator(
            model, latent, proposals, routing.selected_weights
        )
    logits = model.output_projection(model.output_norm(integrated))
    return {
        "natural_scores": natural_routing.scores,
        "natural_probabilities": natural_routing.pre_top_k_probabilities,
        "natural_indices": natural_routing.selected_indices,
        "scores": routing.scores,
        "probabilities": routing.pre_top_k_probabilities,
        "selected_indices": routing.selected_indices,
        "selected_weights": routing.selected_weights,
        "proposals": proposals,
        "integrated": integrated,
        "logits": logits,
    }


def _assert_equivalent(device: torch.device) -> None:
    optimized = create_n2_model(
        64,
        "quick",
        maximum_sequence_length=16,
        seed=17,
        population="mixed",
        top_k=2,
        n1_depth=3,
    ).to(device)
    reference = create_n2_model(
        64,
        "quick",
        maximum_sequence_length=16,
        seed=99,
        population="mixed",
        top_k=2,
        n1_depth=3,
    ).to(device)
    reference.load_state_dict(optimized.state_dict())
    optimized.train()
    reference.train()
    token_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        device=device,
    )
    targets = torch.roll(token_ids, shifts=-1, dims=1)
    forced = torch.tensor([[0, 1], [2, 3]], device=device)
    with _precision_context(device):
        actual = _staged_forward(optimized, token_ids, forced, optimized=True)
        expected = _staged_forward(reference, token_ids, forced, optimized=False)
        actual_loss = next_token_loss(actual["logits"], targets)
        expected_loss = next_token_loss(expected["logits"], targets)
    atol = 3e-2 if device.type == "cuda" else 2e-5
    rtol = 3e-2 if device.type == "cuda" else 2e-5
    for name in (
        "natural_scores",
        "natural_probabilities",
        "scores",
        "probabilities",
        "selected_weights",
        "proposals",
        "integrated",
        "logits",
    ):
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol)
    assert torch.equal(actual["natural_indices"], expected["natural_indices"])
    assert torch.equal(actual["selected_indices"], expected["selected_indices"])
    torch.testing.assert_close(actual_loss, expected_loss, atol=atol, rtol=rtol)

    actual_loss.backward()
    expected_loss.backward()
    actual_parameters = dict(optimized.named_parameters())
    expected_parameters = dict(reference.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    gradient_atol = 6e-2 if device.type == "cuda" else 3e-5
    gradient_rtol = 6e-2 if device.type == "cuda" else 3e-5
    for name in actual_parameters:
        actual_gradient = actual_parameters[name].grad
        expected_gradient = expected_parameters[name].grad
        assert (actual_gradient is None) is (expected_gradient is None), name
        if actual_gradient is not None and expected_gradient is not None:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=gradient_atol,
                rtol=gradient_rtol,
                msg=lambda message, name=name: f"{name}: {message}",
            )


def test_optimized_n2_matches_reference_fp32() -> None:
    _assert_equivalent(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optimized_n2_matches_reference_bf16_cuda() -> None:
    _assert_equivalent(torch.device("cuda"))
