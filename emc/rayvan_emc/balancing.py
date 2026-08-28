from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RouterBalanceMetrics:
    loss: Tensor
    utilization: Tensor
    normalized_entropy: Tensor


def router_balance_metrics(
    router_scores: Tensor,
    selected_indices: Tensor,
    *,
    entropy_floor: float = 0.75,
) -> RouterBalanceMetrics:
    """Penalize only routing whose normalized utilization entropy is too low.

    Forward values use actual top-K assignment traffic. Gradients use the full soft
    router probabilities, giving every module a path to move into the selected set.
    Squared ReLU creates a zero-penalty region above ``entropy_floor`` and a smooth
    increase below it.
    """
    if not 0.0 <= entropy_floor <= 1.0:
        raise ValueError("entropy_floor must be between zero and one")
    if router_scores.shape[:-1] != selected_indices.shape[:-1]:
        raise ValueError("router score and selection dimensions do not match")

    module_count = router_scores.size(-1)
    if module_count <= 1:
        one = router_scores.new_ones(())
        return RouterBalanceMetrics(router_scores.new_zeros(()), one.unsqueeze(0), one)

    soft_utilization = torch.softmax(router_scores, dim=-1).reshape(
        -1, module_count
    ).mean(dim=0)
    hard_utilization = F.one_hot(
        selected_indices, num_classes=module_count
    ).to(router_scores.dtype).reshape(-1, module_count).mean(dim=0)
    utilization = soft_utilization + (hard_utilization - soft_utilization).detach()

    entropy = -(
        utilization * utilization.clamp_min(torch.finfo(utilization.dtype).tiny).log()
    ).sum()
    normalized_entropy = entropy / math.log(module_count)
    entropy_shortfall = F.relu(
        normalized_entropy.new_tensor(entropy_floor) - normalized_entropy
    )
    return RouterBalanceMetrics(
        loss=entropy_shortfall.square(),
        utilization=utilization,
        normalized_entropy=normalized_entropy,
    )


def router_balance_loss(
    router_scores: Tensor,
    selected_indices: Tensor,
    *,
    entropy_floor: float = 0.75,
) -> Tensor:
    return router_balance_metrics(
        router_scores,
        selected_indices,
        entropy_floor=entropy_floor,
    ).loss
