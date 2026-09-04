from __future__ import annotations

import torch
import torch.nn as nn

from src.data.dataset import Batch
from src.metrics import geometry, optimization

# Column schema for the per-step trajectory CSV.
TRAJECTORY_FIELDS = [
    "step",
    "phase",
    "train_loss",
    "val_loss",
    "val_acc",
    "test_acc",
    "grad_norm",
    "update_norm",
    "relative_update",
    "update_sparsity",
    "lr",
    "aux_lr",
    "momentum_grad_cos",
    "probe_grad_cos",
]


class MetricCollector:
    """Collects optimization-trajectory metrics around a single optimizer step.

    Design notes:
    - All gradient/momentum reads happen BEFORE `optimizer.step()`, because both
      Muon and the aux-Adam mutate their momentum buffers \ + gradient in place.
    - Momentum-gradient alignment is measured over a fixed, documented subset:
      the encoder-body >=2-D weight matrices (the Muon-eligible parameters),
      using each optimizer's own first-moment state (`state_key`). This keeps the
      AdamW-vs-Muon comparison on identical parameters.
    - Probe-gradient cosine is measured over all trainable parameters on a fixed
      deterministic batch, with dropout disabled.
    """

    def __init__(
        self,
        all_params: list[nn.Parameter],
        align_params: list[nn.Parameter],
        align_state_key: str,
        sparsity_tau: float = 0.1,
    ) -> None:
        self.all_params = all_params
        self.align_params = align_params
        self.align_state_key = align_state_key
        self.sparsity_tau = sparsity_tau
        self._prev_probe_grad: torch.Tensor | None = None

    def pre_step(self, optimizer: torch.optim.Optimizer) -> tuple[dict, list[torch.Tensor]]:
        grad_norm = optimization.grad_global_norm(self.all_params)
        cos = geometry.momentum_grad_alignment(
            optimizer, self.align_params, self.align_state_key
        )
        snapshots = [p.data.clone() for p in self.all_params]
        return {"grad_norm": grad_norm, "momentum_grad_cos": cos}, snapshots

    def post_step(self, snapshots: list[torch.Tensor]) -> dict:
        upd = optimization.update_global_norm(snapshots, self.all_params)
        pnorm = optimization.param_global_norm(self.all_params)
        return {
            "update_norm": upd,
            "relative_update": optimization.relative_update(upd, pnorm),
            "update_sparsity": optimization.update_sparsity(
                snapshots, self.all_params, self.sparsity_tau
            ),
        }

    def probe(self, model: nn.Module, batch: Batch) -> float:
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(batch)
        loss.backward()
        grad = geometry.flatten_params(
            [p.grad for p in self.all_params if p.grad is not None]
        ).cpu()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        cos = geometry.probe_grad_cosine(self._prev_probe_grad, grad)
        self._prev_probe_grad = grad
        return cos
