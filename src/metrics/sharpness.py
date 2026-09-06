from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.data.dataset import Batch


@dataclass
class SharpnessResult:
    eps: list[float]
    mean_increase: list[float]
    max_increase: list[float]
    std_increase: list[float]
    baseline_loss: float
    n_directions: int
    raw: list[list[float]] = field(default_factory=list)  # [eps][dir] loss increase


@torch.no_grad()
def _mean_loss(model: nn.Module, batches: list[Batch]) -> float:
    total, n = 0.0, 0
    for b in batches:
        _, loss = model(b)
        # since not all batches are the same size, we need to weight by batch size
        total += float(loss) * b.labels.size(0) 
        n += b.labels.size(0)
    return total / max(n, 1)


def _trainable(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def filter_normalized_direction(
    params: list[nn.Parameter], generator: torch.Generator
) -> list[torch.Tensor]:
    """Random Gaussian direction, each parameter tensor rescaled to that
    tensor's own norm (filter-wise normalization, Li et al. 2018). This makes
    the perturbation scale comparable across parameters and across optimizers.
    """
    out = []
    for p in params:
        d = torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype)
        dn = d.norm()
        if dn > 0:
            pn = p.data.norm()
            d = d / dn * pn
        out.append(d)
    return out


@torch.no_grad()
def sharpness_scan(
    model: nn.Module,
    eval_batches: list[Batch],
    eps_list: list[float],
    n_directions: int,
    seed: int = 0,
) -> SharpnessResult:
    """Perturbation-based local-sharpness estimate at the current parameters.

    For each random filter-normalized direction d and magnitude eps, measures
    L(theta + eps*d) - L(theta) on a fixed evaluation batch set. Smaller mean
    increase => locally flatter under this measurement.
    """
    params = _trainable(model)
    originals = [p.data.clone() for p in params]
    baseline = _mean_loss(model, eval_batches)

    gen = torch.Generator(device=params[0].device)
    gen.manual_seed(seed)

    per_eps: list[list[float]] = [[] for _ in eps_list]
    for _ in range(n_directions):
        direction = filter_normalized_direction(params, gen)
        for j, eps in enumerate(eps_list):
            for p, base, d in zip(params, originals, direction):
                p.data.copy_(base + eps * d)
            per_eps[j].append(_mean_loss(model, eval_batches) - baseline)
    for p, base in zip(params, originals):
        p.data.copy_(base)

    mean = [float(torch.tensor(v).mean()) for v in per_eps]
    mx = [float(torch.tensor(v).max()) for v in per_eps]
    std = [float(torch.tensor(v).std(unbiased=False)) for v in per_eps]
    return SharpnessResult(
        eps=list(eps_list),
        mean_increase=mean,
        max_increase=mx,
        std_increase=std,
        baseline_loss=baseline,
        n_directions=n_directions,
        raw=per_eps,
    )
