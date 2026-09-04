from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.metrics import geometry, optimization
from src.optim.factory import build_optimizer
from src.utils.config import OptimHParams
from tests.conftest import TinyClassifier, make_batch


def _param(vals):
    p = nn.Parameter(torch.tensor(vals, dtype=torch.float32))
    return p


def test_global_norm_matches_manual():
    t = [torch.tensor([3.0, 4.0]), torch.tensor([12.0])]
    assert math.isclose(optimization.global_norm(t), 13.0, rel_tol=1e-6)


def test_update_norm_and_relative_update():
    p = _param([3.0, 4.0])
    snap = [p.data.clone()]
    p.data = torch.tensor([6.0, 8.0])  # delta = (3,4), norm 5
    upd = optimization.update_global_norm(snap, [p])
    assert math.isclose(upd, 5.0, rel_tol=1e-6)
    rel = optimization.relative_update(upd, optimization.param_global_norm([p]))
    assert math.isclose(rel, 5.0 / 10.0, rel_tol=1e-6)


def test_update_sparsity_counts_relative_threshold():
    p = _param([10.0, 10.0, 10.0, 10.0])
    snap = [p.data.clone()]
    # deltas: 2,2,0.5,0.5 vs |prev|*tau (tau=0.1 -> threshold 1.0) => 2 hits of 4
    p.data = torch.tensor([12.0, 8.0, 10.5, 9.5])
    s = optimization.update_sparsity(snap, [p], tau=0.1)
    assert math.isclose(s, 0.5, rel_tol=1e-6)


def test_cosine_sim_orthogonal_and_parallel():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert abs(geometry.cosine_sim(a, b)) < 1e-6
    assert math.isclose(geometry.cosine_sim(a, a), 1.0, rel_tol=1e-6)


def test_cosine_sim_zero_vector_is_nan():
    z = torch.zeros(3)
    assert math.isnan(geometry.cosine_sim(z, torch.ones(3)))


def test_momentum_alignment_nan_before_state_then_finite():
    model = TinyClassifier()
    opt, _ = build_optimizer("adamw", model, OptimHParams())
    body = [p for n, p in model.named_parameters() if "transformer_layer.weight" in n]

    _, loss = model(make_batch(seed=1))
    loss.backward()
    # no optimizer state yet -> alignment undefined
    assert math.isnan(geometry.momentum_grad_alignment(opt, body, "exp_avg"))
    opt.step()
    opt.zero_grad()

    _, loss = model(make_batch(seed=2))
    loss.backward()
    val = geometry.momentum_grad_alignment(opt, body, "exp_avg")
    assert not math.isnan(val) and -1.0 <= val <= 1.0


def test_probe_grad_cosine_prev_none():
    assert math.isnan(geometry.probe_grad_cosine(None, torch.ones(3)))
    assert math.isclose(
        geometry.probe_grad_cosine(torch.ones(3), torch.ones(3)), 1.0, rel_tol=1e-6
    )
