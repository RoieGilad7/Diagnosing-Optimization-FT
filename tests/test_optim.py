from __future__ import annotations

import torch
from muon import SingleDeviceMuonWithAuxAdam

from src.optim.factory import build_optimizer, split_muon_params
from src.utils.config import OptimHParams
from tests.conftest import TinyClassifier, make_batch


def test_split_routes_only_body_2d_weights_to_muon():
    model = TinyClassifier()
    split = split_muon_params(model)
    assert "transformer_layer.weight" in split.muon_names
    # embeddings, classifier head, and all biases must NOT be Muon-eligible
    for name in ["embeddings.weight", "classifier.weight", "transformer_layer.bias"]:
        assert name in split.aux_names
        assert name not in split.muon_names


def test_adamw_optimizes_all_parameters():
    model = TinyClassifier()
    opt, info = build_optimizer("adamw", model, OptimHParams())
    assert isinstance(opt, torch.optim.AdamW)
    n_params = sum(p.numel() for p in model.parameters())
    assert info["adamw_params"] == n_params
    assert info["muon_params"] == 0


def test_muon_hybrid_partition_is_complete_and_disjoint():
    model = TinyClassifier()
    opt, info = build_optimizer("muon", model, OptimHParams())
    assert isinstance(opt, SingleDeviceMuonWithAuxAdam)
    total = sum(p.numel() for p in model.parameters())
    assert info["muon_params"] + info["adamw_params"] == total
    assert info["muon_params"] > 0 and info["adamw_params"] > 0


def test_muon_step_creates_expected_state_keys():
    model = TinyClassifier()
    opt, _ = build_optimizer("muon", model, OptimHParams(lr=0.02, adam_lr=1e-3))
    _, loss = model(make_batch())
    loss.backward()
    opt.step()
    for group in opt.param_groups:
        key = "momentum_buffer" if group.get("use_muon") else "exp_avg"
        for p in group["params"]:
            assert key in opt.state[p]


def test_unknown_optimizer_raises():
    model = TinyClassifier()
    try:
        build_optimizer("sgd", model, OptimHParams())
        assert False, "expected ValueError"
    except ValueError:
        pass
