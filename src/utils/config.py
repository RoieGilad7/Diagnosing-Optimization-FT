from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "stanfordnlp/sst2"
    dataset_config: str | None = None
    text_field: str = "sentence"
    label_field: str = "label"
    max_length: int = 128
    train_subset: int | None = None      # cap #train examples (None = all)
    val_subset: int | None = None
    test_from_train: int = 2000          # held-out test carved from train (SST-2 test labels are hidden)
    probe_size: int = 64                 # fixed deterministic probe batch for gradient stability


@dataclass
class OptimHParams:
    lr: float = 2e-5
    weight_decay: float = 0.01
    momentum: float = 0.95               # Muon only
    adam_lr: float = 3e-4                # aux-AdamW lr for the Muon hybrid
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SharpnessConfig:
    n_directions: int = 5                # random filter-normalized directions sampled
    eps_list: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.05])
    n_eval_batches: int = 4              # val batches averaged over per direction/eps


@dataclass
class TrainConfig:
    max_steps: int = 300
    batch_size: int = 32
    grad_accum: int = 1
    warmup_ratio: float = 0.1
    scheduler: str = "linear"            # linear | cosine | constant
    eval_every: int = 50                 # optimizer steps between evaluations
    metric_every: int = 10               # optimizer steps between trajectory metric rows
    probe_every: int = 10                # optimizer steps between probe-gradient measurements
    grad_clip: float | None = 1.0
    precision: str = "32-true"


@dataclass
class ExperimentConfig:
    optimizer: str = "adamw"             # adamw | muon
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    seed: int = 42
    output_dir: str = "outputs/experiments/run"
    data: DataConfig = field(default_factory=DataConfig)
    optimizer_args: OptimHParams = field(default_factory=OptimHParams)
    train: TrainConfig = field(default_factory=TrainConfig)
    sharpness: SharpnessConfig = field(default_factory=SharpnessConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, data: dict[str, Any] | None):
    if data is None:
        return cls()
    # `from __future__ import annotations` stringizes field types, so resolve
    # them to real classes before checking for nested dataclasses.
    hints = get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in field_names:
            continue
        ftype = hints.get(key)
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[key] = _build(ftype, val)
        elif key == "betas" and isinstance(val, list):
            kwargs[key] = tuple(val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(ExperimentConfig, raw)


def merge_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply a flat dict of dotted overrides, e.g. {'optim.lr': 1e-4, 'train.max_steps': 50}."""
    data = cfg.to_dict()
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return _build(ExperimentConfig, data)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
