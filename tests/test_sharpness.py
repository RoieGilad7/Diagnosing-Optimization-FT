from __future__ import annotations

import math

import torch

from src.metrics.sharpness import filter_normalized_direction, sharpness_scan
from tests.conftest import TinyClassifier, make_batch


def test_filter_normalization_matches_param_norm():
    model = TinyClassifier()
    params = [p for p in model.parameters() if p.requires_grad]
    gen = torch.Generator().manual_seed(0)
    direction = filter_normalized_direction(params, gen)
    for d, p in zip(direction, params):
        if p.data.norm() > 0:
            assert torch.isclose(d.norm(), p.data.norm(), rtol=1e-5)


def test_sharpness_scan_shapes_and_param_restoration():
    torch.manual_seed(0)
    model = TinyClassifier()
    batches = [make_batch(seed=1), make_batch(seed=2)]
    before = [p.data.clone() for p in model.parameters()]

    eps = [0.001, 0.01, 0.05]
    res = sharpness_scan(model, batches, eps_list=eps, n_directions=3, seed=0)

    assert res.eps == eps
    assert len(res.mean_increase) == len(eps)
    assert len(res.max_increase) == len(eps)
    assert res.n_directions == 3
    # parameters must be restored exactly after the scan
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p.data, b)


def test_zero_perturbation_gives_zero_increase():
    torch.manual_seed(0)
    model = TinyClassifier()
    batches = [make_batch(seed=3)]
    res = sharpness_scan(model, batches, eps_list=[0.0, 0.1], n_directions=8, seed=1)
    # zero perturbation must give ~0 increase; larger eps must produce a finite value
    assert abs(res.mean_increase[0]) < 1e-6
    assert math.isfinite(res.mean_increase[1])
