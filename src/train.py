from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import yaml

from src.training.runner import run_experiment
from src.utils.config import ExperimentConfig, load_config, merge_overrides, save_config

CONFIG_DIR = Path("configs")


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_base_config(optimizer: str) -> ExperimentConfig:
    from src.utils.config import _build

    with open(CONFIG_DIR / "default.yaml") as f:
        base = yaml.safe_load(f) or {}
    with open(CONFIG_DIR / f"{optimizer}.yaml") as f:
        opt = yaml.safe_load(f) or {}
    return _build(ExperimentConfig, _deep_merge(base, opt))


def apply_cli_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    ov: dict = {}
    if args.max_steps is not None:
        ov["train.max_steps"] = args.max_steps
    if args.batch_size is not None:
        ov["train.batch_size"] = args.batch_size
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.train_subset is not None:
        ov["data.train_subset"] = args.train_subset
    if args.val_subset is not None:
        ov["data.val_subset"] = args.val_subset
    if args.output_dir is not None:
        ov["output_dir"] = args.output_dir
    return merge_overrides(cfg, ov) if ov else cfg


def quick_overrides() -> dict:
    return {
        "data.train_subset": 256,
        "data.val_subset": 256,
        "data.test_from_train": 128,
        "data.probe_size": 32,
        "train.max_steps": 20,
        "train.batch_size": 16,
        "train.eval_every": 10,
        "train.metric_every": 5,
        "train.probe_every": 5,
        "train.warmup_ratio": 0.1,
        "sharpness.n_directions": 5,
        "sharpness.n_eval_batches": 4,
    }


def run_phase0(args) -> None:
    cfg = load_base_config(args.optimizer)
    cfg = merge_overrides(cfg, quick_overrides())
    cfg = merge_overrides(cfg, {"output_dir": args.output_dir or f"outputs/phase0/{args.optimizer}"})
    print(f"[phase0] sanity check for {args.optimizer}")
    summary = run_experiment(cfg, do_sharpness=True, save_checkpoint=True, verbose=True)

    out = Path(cfg.output_dir)
    checks = {
        "metrics.csv": (out / "metrics.csv").exists(),
        "last.ckpt": (out / "last.ckpt").exists(),
        "config.yaml": (out / "config.yaml").exists(),
        "summary.json": (out / "summary.json").exists(),
    }
    _verify_checkpoint_loads(out / "last.ckpt", cfg)
    print("[phase0] artifact checks:", checks)
    print(f"[phase0] val_acc={summary['val_acc']:.4f} params={summary['param_groups']}")
    assert all(checks.values()), "phase0 missing artifacts"
    print("[phase0] PASS")


def _verify_checkpoint_loads(ckpt: Path, cfg: ExperimentConfig) -> None:
    import torch

    from src.training.lit_module import FineTuneModule

    module = FineTuneModule.load_from_checkpoint(str(ckpt), cfg=cfg, map_location="cpu")
    assert isinstance(module, FineTuneModule)
    del module
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def suggest_hparams(trial: optuna.Trial, optimizer: str) -> dict:
    """Intentionally small search space (see prompt): lr + weight decay, plus
    momentum for Muon. All other optimizer settings are held fixed."""
    if optimizer == "adamw":
        return {
            "optim.lr": trial.suggest_float("lr", 5e-6, 1e-4, log=True),
            "optim.weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }
    if optimizer == "muon":
        return {
            "optim.lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "optim.weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "optim.momentum": trial.suggest_float("momentum", 0.9, 0.99),
        }
    raise ValueError(optimizer)


def tune(
    base_cfg: ExperimentConfig,
    n_trials: int,
    study_dir: Path,
    verbose: bool = False,
) -> optuna.Study:
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=f"{base_cfg.optimizer}_sst2",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base_cfg.seed),
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_hparams(trial, base_cfg.optimizer)
        overrides["output_dir"] = str(study_dir / f"trial_{trial.number}")
        cfg = merge_overrides(base_cfg, overrides)
        summary = run_experiment(
            cfg, do_sharpness=False, save_checkpoint=False, verbose=verbose
        )
        trial.set_user_attr("val_loss", summary["val_loss"])
        return summary["val_acc"]

    study.optimize(objective, n_trials=n_trials)
    return study


def save_best_config(
    study: optuna.Study, phase2_cfg: ExperimentConfig, out_path: Path
) -> ExperimentConfig:
    overrides = {f"optim.{k}": v for k, v in study.best_params.items()}
    best = merge_overrides(phase2_cfg, overrides)
    save_config(best, out_path)
    return best


def run_phase1(args) -> None:
    cfg = load_base_config(args.optimizer)
    cfg = apply_cli_overrides(cfg, args)
    if args.quick:
        cfg = merge_overrides(cfg, quick_overrides())
    study_dir = Path(args.output_dir or f"outputs/optuna/{args.optimizer}")
    print(f"[phase1] tuning {args.optimizer}: {args.trials} trials -> {study_dir}")

    study = tune(cfg, n_trials=args.trials, study_dir=study_dir, verbose=args.verbose)

    phase2_cfg = merge_overrides(cfg, {"output_dir": f"outputs/experiments/{args.optimizer}_best"})
    best = save_best_config(study, phase2_cfg, study_dir / "best_config.yaml")
    print(f"[phase1] best val_acc={study.best_value:.4f} params={study.best_params}")
    print(f"[phase1] wrote {study_dir / 'best_config.yaml'}")
    del best


def run_phase2(args) -> None:
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_base_config(args.optimizer)
        cfg = merge_overrides(cfg, {"output_dir": f"outputs/experiments/{args.optimizer}_matched"})
    cfg = apply_cli_overrides(cfg, args)
    if args.quick:
        cfg = merge_overrides(cfg, quick_overrides())
    print(f"[phase2] training {cfg.optimizer} -> {cfg.output_dir}")
    summary = run_experiment(cfg, do_sharpness=True, save_checkpoint=True, verbose=args.verbose)
    print(
        f"[phase2] val_acc={summary['val_acc']:.4f} test_acc={summary['test_acc']:.4f} "
        f"params={summary['param_groups']}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AdamW vs Muon SST-2 fine-tuning")
    p.add_argument("--optimizer", choices=["adamw", "muon"], required=True)
    p.add_argument("--phase", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="tiny smoke-test settings")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    phase_to_func = {0: run_phase0, 1: run_phase1, 2: run_phase2}
    phase_to_func[args.phase](args)
    

if __name__ == "__main__":
    main()
