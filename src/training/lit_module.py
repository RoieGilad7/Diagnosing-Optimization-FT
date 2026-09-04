from __future__ import annotations

import math
from dataclasses import dataclass

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.data.dataset import Batch
from src.logging.csv_logger import CSVMetricLogger
from src.models.classifier import SequenceClassifier
from src.optim.factory import build_optimizer, split_muon_params
from src.training.metrics_collector import TRAJECTORY_FIELDS, MetricCollector
from src.utils.config import ExperimentConfig


@dataclass(frozen=True)
class _StepCadence:
    """Which of this optimizer step's periodic actions are due."""

    metric: bool
    eval: bool
    probe: bool

    @property
    def logged(self) -> bool:
        return self.metric or self.eval or self.probe


class FineTuneModule(L.LightningModule):
    def __init__(self, cfg: ExperimentConfig, csv_path: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.model = SequenceClassifier(cfg.model_name, cfg.num_labels)
        self.opt_info: dict = {}
        self.opt_step = 0
        self._csv_path = csv_path
        self._logger: CSVMetricLogger | None = None
        self._collector: MetricCollector | None = None
        self._scheduler = None
        self.save_hyperparameters(cfg.to_dict())

    # ---- setup -----------------------------------------------------------
    def configure_optimizers(self):
        optimizer, info = build_optimizer(self.cfg.optimizer, self.model, self.cfg.optimizer_args)
        self.opt_info = info

        warmup = int(self.cfg.train.warmup_ratio * self.cfg.train.max_steps)
        self._scheduler = get_scheduler(
            self.cfg.train.scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=self.cfg.train.max_steps,
        )

        split = split_muon_params(self.model)
        align_key = "momentum_buffer" if self.cfg.optimizer == "muon" else "exp_avg"
        self._collector = MetricCollector(
            all_params=[p for p in self.model.parameters() if p.requires_grad],
            align_params=split.muon,
            align_state_key=align_key,
        )
        return optimizer

    def on_fit_start(self) -> None:
        if self._csv_path and self._logger is None:
            self._logger = CSVMetricLogger(self._csv_path, TRAJECTORY_FIELDS)

    def on_train_start(self) -> None:
        # transformers>=5 returns models in eval mode; ensure dropout is active.
        self.model.train()

    def transfer_batch_to_device(self, batch: Batch, device, dataloader_idx: int) -> Batch:
        return batch.to(device)

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        optimizer = self.optimizers()
        loss = self._forward_backward(batch)
        if (batch_idx + 1) % self.cfg.train.grad_accum != 0:
            return  # still accumulating gradient

        self.opt_step += 1
        step = self.opt_step
        cadence = self._cadence(step)

        # Gradient/momentum reads must happen before optimizer.step(): both
        # Muon and the aux-Adam mutate their state buffers in place during the step
        pre, snapshots = self._collector.pre_step(optimizer) if cadence.metric else ({}, None)
        self._apply_optimizer_step(optimizer)

        row = self._trajectory_row(step, loss, optimizer, cadence, pre, snapshots)
        if cadence.logged and self._logger is not None:
            self._logger.log(row)

        if step >= self.cfg.train.max_steps:
            self.trainer.should_stop = True

    def _forward_backward(self, batch: Batch) -> torch.Tensor:
        """Forward + scaled backward for one micro-batch. Returns the
        (unscaled) loss of this micro-batch, used for the trajectory row."""
        _, loss = self.model(batch)
        self.manual_backward(loss / self.cfg.train.grad_accum)
        return loss

    def _cadence(self, step: int) -> _StepCadence:
        tr = self.cfg.train
        return _StepCadence(
            metric=step % tr.metric_every == 0,
            eval=step % tr.eval_every == 0,
            probe=step % tr.probe_every == 0,
        )

    def _apply_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        tr = self.cfg.train
        if tr.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tr.grad_clip)
        optimizer.step()
        self._scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def _trajectory_row(
        self,
        step: int,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        cadence: _StepCadence,
        pre: dict,
        snapshots: list[torch.Tensor] | None,
    ) -> dict:
        row: dict = {"step": step, "phase": "train", "train_loss": float(loss.detach())}
        row.update(self._lr_row(optimizer))
        if cadence.metric:
            row.update(pre)
            row.update(self._collector.post_step(snapshots))
        if cadence.probe:
            row["probe_grad_cos"] = self._probe_grad_cos()
        if cadence.eval:
            row.update(self._eval_row())
        return row

    def _probe_grad_cos(self) -> float:
        probe = self.trainer.datamodule.probe_batch().to(self.device)
        return self._collector.probe(self.model, probe)

    def _eval_row(self) -> dict:
        val_loss, val_acc = self.evaluate(self.trainer.datamodule.val_dataloader())
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_loss", val_loss)
        return {"val_loss": val_loss, "val_acc": val_acc}

    def _lr_row(self, optimizer: torch.optim.Optimizer) -> dict:
        groups = optimizer.param_groups
        if self.cfg.optimizer == "muon":
            muon_lr = next((g["lr"] for g in groups if g.get("use_muon")), math.nan)
            aux_lr = next((g["lr"] for g in groups if not g.get("use_muon")), math.nan)
            return {"lr": muon_lr, "aux_lr": aux_lr}
        return {"lr": groups[0]["lr"]}

    # ---- evaluation ------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        tot_loss, correct, n = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(self.device)
            logits, loss = self.model(batch)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch.labels).sum())
            tot_loss += float(loss) * batch.labels.size(0)
            n += batch.labels.size(0)
        self.model.train()
        return tot_loss / max(n, 1), correct / max(n, 1)
