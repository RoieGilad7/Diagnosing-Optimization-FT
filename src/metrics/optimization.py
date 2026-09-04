from __future__ import annotations

import torch
from torch import Tensor


def global_norm(tensors: list[Tensor]) -> float:
    """L2 norm of the concatenation of all tensors (the standard 'global grad norm')."""
    if not tensors:
        return 0.0
    sq = torch.stack([t.detach().float().pow(2).sum() for t in tensors]).sum()
    return float(sq.sqrt())


def grad_global_norm(params: list[torch.nn.Parameter]) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    return global_norm(grads)


def param_global_norm(params: list[torch.nn.Parameter]) -> float:
    return global_norm([p.data for p in params])


def update_global_norm(pre_step: list[Tensor], params: list[torch.nn.Parameter]) -> float:
    """||theta_t+1 - theta_t|| given pre-step snapshots aligned to `params`."""
    deltas = [p.data - prev for prev, p in zip(pre_step, params)]
    return global_norm(deltas)


def relative_update(update_norm: float, param_norm: float, eps: float = 1e-12) -> float:
    return update_norm / (param_norm + eps)


def update_sparsity(
    pre_step: list[Tensor], params: list[torch.nn.Parameter], tau: float
) -> float:
    """Fraction of coordinates whose relative update exceeds tau: |dtheta_i| > tau*|theta_i|."""
    total = 0
    hits = 0
    for prev, p in zip(pre_step, params):
        delta = (p.data - prev).abs()
        thresh = tau * prev.abs()
        hits += int((delta > thresh).sum())
        total += delta.numel()
    return hits / total if total else 0.0
