from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data.datamodule import SST2DataModule
from src.metrics.sharpness import sharpness_scan
from src.training.lit_module import FineTuneModule
from src.utils.config import ExperimentConfig, save_config
from src.utils.reproducibility import seed_everything, software_versions


def _accelerator() -> str:
    import torch

    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_trainer(cfg: ExperimentConfig, output_dir: Path, save_checkpoint: bool, verbose: bool):
    callbacks = []
    if save_checkpoint:
        callbacks.append(
            ModelCheckpoint(dirpath=str(output_dir), save_last=True, save_top_k=0)
        )
    return L.Trainer(
        accelerator=_accelerator(),
        devices=1,
        max_epochs=100000,
        max_steps=-1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=save_checkpoint,
        logger=False,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        callbacks=callbacks,
    )


def run_experiment(
    cfg: ExperimentConfig,
    do_sharpness: bool = True,
    save_checkpoint: bool = True,
    verbose: bool = True,
    on_eval: Callable[[int, float], None] | None = None,
) -> dict:
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = SST2DataModule(cfg.data, cfg.model_name, cfg.train.batch_size, cfg.seed)
    module = FineTuneModule(cfg, csv_path=str(output_dir / "metrics.csv"), on_eval=on_eval)

    trainer = build_trainer(cfg, output_dir, save_checkpoint, verbose)
    trainer.fit(module, datamodule=dm)

    val_loss, val_acc = module.evaluate(dm.val_dataloader())
    test_loss, test_acc = module.evaluate(dm.test_dataloader())

    summary = {
        "optimizer": cfg.optimizer,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "opt_steps": module.opt_step,
        "param_groups": module.opt_info,
        "versions": software_versions(),
    }

    if do_sharpness:
        sh = cfg.sharpness
        eval_batches = _first_batches(dm.val_dataloader(), module.device, n=sh.n_eval_batches)
        result = sharpness_scan(
            module.model,
            eval_batches,
            eps_list=sh.eps_list,
            n_directions=sh.n_directions,
            seed=cfg.seed,
        )
        summary["sharpness"] = {
            "eps": result.eps,
            "mean_increase": result.mean_increase,
            "max_increase": result.max_increase,
            "std_increase": result.std_increase,
            "baseline_loss": result.baseline_loss,
        }

    save_config(cfg, output_dir / "config.yaml")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _first_batches(loader, device, n: int):
    batches = []
    for i, b in enumerate(loader):
        if i >= n:
            break
        batches.append(b.to(device))
    return batches
