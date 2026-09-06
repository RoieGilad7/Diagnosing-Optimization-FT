from __future__ import annotations

from pathlib import Path

from src.utils.config import (
    ExperimentConfig,
    load_config,
    merge_overrides,
    save_config,
)

CONFIGS = Path("configs")


def test_load_default_config_types():
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.model_name == "distilbert-base-uncased"
    assert cfg.train.max_steps == 5000
    assert cfg.data.dataset_name == "stanfordnlp/sst2"


def test_muon_config_has_momentum_and_betas_tuple():
    cfg = load_config(CONFIGS / "muon.yaml")
    assert cfg.optimizer == "muon"
    assert cfg.optim.momentum == 0.95
    assert isinstance(cfg.optim.betas, tuple)


def test_merge_overrides_dotted_paths():
    cfg = ExperimentConfig()
    merged = merge_overrides(cfg, {"optim.lr": 1e-3, "train.max_steps": 5, "seed": 7})
    assert merged.optim.lr == 1e-3
    assert merged.train.max_steps == 5
    assert merged.seed == 7
    # original is untouched
    assert cfg.optim.lr != 1e-3


def test_save_load_roundtrip(tmp_path):
    cfg = merge_overrides(ExperimentConfig(), {"optim.lr": 3e-4})
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.optim.lr == 3e-4
    assert loaded.to_dict() == cfg.to_dict()
