"""Generate deterministic Python reference fixtures for the native EMC tests.

This utility is test/conversion tooling only. The C++ runtime never imports Python.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rayvan_emc.n2 import N1Input, N2Config, N2EMCModel


class TensorBundle(nn.Module):
    def __init__(self, tensors: dict[str, Tensor]) -> None:
        super().__init__()
        self.register_buffer("_anchor", torch.zeros(1))
        for name, tensor in tensors.items():
            self.register_buffer(name.replace(".", "__DOT__"), tensor.detach().cpu().contiguous())

    def forward(self) -> Tensor:
        return self._anchor


def save_bundle(path: Path, tensors: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(TensorBundle(tensors)).save(str(path))


def config() -> N2Config:
    return N2Config(
        latent_dim=16,
        vocab_size=67,
        max_sequence_length=8,
        attention_heads=4,
        integrator_heads=4,
        module_hidden_dim=32,
        state_space_dim=24,
        state_space_kernel_size=3,
        recurrent_dim=20,
        chunk_size=4,
        shared_state_slots=2,
        num_modules=3,
        modules_per_cycle=2,
        active_top_k=2,
        n1_depth=2,
        module_families=("gpt", "ssm", "recurrent"),
        n2_population="supported",
        tie_embeddings=True,
        recurrent_precision="model",
    )


def model_config_text(cfg: N2Config) -> str:
    return "\n".join(
        (
            "format=rayvan-emc-config-v1",
            f"latent_dim={cfg.latent_dim}",
            f"vocab_size={cfg.vocab_size}",
            f"max_sequence_length={cfg.max_sequence_length}",
            f"attention_heads={cfg.attention_heads}",
            f"integrator_heads={cfg.integrator_heads}",
            f"module_hidden_dim={cfg.module_hidden_dim or 0}",
            f"state_space_dim={cfg.state_space_dim or 0}",
            f"state_space_kernel_size={cfg.state_space_kernel_size}",
            f"recurrent_dim={cfg.recurrent_dim or 0}",
            f"chunk_size={cfg.chunk_size}",
            f"shared_state_slots={cfg.shared_state_slots}",
            f"n1_depth={cfg.n1_depth}",
            f"top_k={cfg.resolved_active_top_k}",
            f"tie_embeddings={int(cfg.tie_embeddings)}",
            "population=gpt,ssm,recurrent",
            "",
        )
    )


def reference_forward(model: N2EMCModel, tokens: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    positions = torch.arange(tokens.size(1), device=tokens.device)
    embeddings = model.token_embedding(tokens) + model.position_embedding(positions)
    routing = model.nexus(embeddings, top_k=model.active_top_k)
    proposals, _, _, _, dispatch = model._execute_selected_nodes(  # noqa: SLF001
        embeddings, routing.selected_indices, n2_state=None
    )
    integrated, trace = model.integrator(
        embeddings, proposals, routing.selected_weights, return_diagnostics=True
    )
    logits = model.output_projection(model.output_norm(integrated))
    targets = torch.roll(tokens, shifts=-1, dims=1)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    full_proposals: dict[str, Tensor] = {}
    request_indices = torch.arange(tokens.size(0))
    for index, node in enumerate(model.n1_nodes):
        full_proposals[f"n1_proposal_{index}"] = node(
            N1Input(shared_latent=embeddings, request_indices=request_indices)
        ).proposal

    tensors = {
        "tokens": tokens,
        "targets": targets,
        "embeddings": embeddings,
        "shared_state": embeddings,
        "router_scores": routing.scores,
        "router_probabilities": routing.pre_top_k_probabilities,
        "top_k_indices": routing.selected_indices,
        "selected_weights": routing.selected_weights,
        "proposals": proposals,
        "dispatch_permutation": dispatch.permutation,
        "dispatch_inverse_permutation": dispatch.inverse_permutation,
        "integrator_acceptance": trace.proposal_acceptance,
        "integrated_state": integrated,
        "logits": logits,
        "loss": loss.reshape(1),
        **full_proposals,
    }
    return tensors, loss


def export(destination: Path) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260902)
    cfg = config()
    model = N2EMCModel(cfg).eval()
    tokens = torch.tensor(
        [[1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30]],
        dtype=torch.long,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model.rvcfg").write_text(model_config_text(cfg), encoding="utf-8")
    save_bundle(destination / "weights.pt", dict(model.state_dict()))

    tensors, _ = reference_forward(model, tokens)
    forced_nodes = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    forced_output = model(
        tokens,
        return_trace=True,
        diagnostic_forced_modules=forced_nodes,
    )
    gradient_loss = F.cross_entropy(
        forced_output.logits.reshape(-1, forced_output.logits.size(-1)),
        torch.roll(tokens, shifts=-1, dims=1).reshape(-1),
    )
    model.zero_grad(set_to_none=True)
    gradient_loss.backward()
    gradient_names = (
        "token_embedding.weight",
        "router.score_projection.weight",
        "n1_nodes.0.blocks.0.attention.in_proj_weight",
        "n1_nodes.1.blocks.0.log_decay",
        "n1_nodes.2.blocks.0.recurrent.weight_ih_l0",
        "integrator.proposal_integrator.query_projection.weight",
        "output_norm.weight",
    )
    gradients = {
        "forced_nodes": forced_nodes,
        "loss": gradient_loss.reshape(1),
        **{
            f"gradient.{name}": dict(model.named_parameters())[name].grad
            for name in gradient_names
        },
    }
    save_bundle(destination / "forward.pt", tensors)
    save_bundle(destination / "gradients.pt", gradients)

    # Tiny optimizer trajectory starts from the same exported initialization.
    torch.manual_seed(20260902)
    model = N2EMCModel(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    losses: list[Tensor] = []
    routes: list[Tensor] = []
    gradient_norms: list[Tensor] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(tokens, return_trace=True)
        step_loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)),
            torch.roll(tokens, shifts=-1, dims=1).reshape(-1),
        )
        step_loss.backward()
        gradient_norms.append(
            torch.sqrt(sum(parameter.grad.float().square().sum() for parameter in model.parameters() if parameter.grad is not None))
        )
        optimizer.step()
        losses.append(step_loss.detach())
        routes.append(output.chunk_trace.selected_node_ids.detach())
    save_bundle(
        destination / "training.pt",
        {
            "losses": torch.stack(losses),
            "routes": torch.stack(routes),
            "gradient_norms": torch.stack(gradient_norms),
            **{f"final.{name}": parameter for name, parameter in model.named_parameters() if name in gradient_names},
        },
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "seed": 20260902,
                "fp32": {"atol": 2e-5, "rtol": 2e-4},
                "bf16": {"atol": 3e-2, "rtol": 5e-2},
                "top_k": "exact except diagnosed score ties",
                "architecture": ["gpt", "ssm", "recurrent"],
                "delta": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.destination)


if __name__ == "__main__":
    main()
