from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    na = a.norm()
    nb = b.norm()
    if na < eps or nb < eps:
        return math.nan
    return float(torch.dot(a, b) / (na * nb))


def flatten_params(tensors: list[Tensor]) -> Tensor:
    return torch.cat([t.detach().float().reshape(-1) for t in tensors])


def momentum_grad_alignment(
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
    state_key: str,
) -> float:
    """CosSim(m_{t-1}, g_t) over `params`, using the optimizer's stored first
    moment/momentum (`state_key`). Must be called BEFORE optimizer.step(), since
    both Muon and the aux-Adam update their buffers in place during the step.
    Returns NaN until the state buffers exist (i.e. after the first step).
    """
    moments, grads = [], []
    for p in params:
        if p.grad is None:
            continue
        st = optimizer.state.get(p, {})
        buf = st.get(state_key)
        if buf is None:
            continue
        moments.append(buf)
        grads.append(p.grad)
    if not moments:
        return math.nan
    return cosine_sim(flatten_params(moments), flatten_params(grads))


def probe_grad_cosine(prev: Tensor | None, current: Tensor) -> float:
    """CosSim(g_t^probe, g_{t-k}^probe). Returns NaN when there is no previous probe."""
    if prev is None:
        return math.nan
    return cosine_sim(prev, current)
