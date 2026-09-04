from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from muon import SingleDeviceMuonWithAuxAdam

from src.utils.config import OptimHParams


class MuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    """Adapter accepting Lightning's manual-optimization `closure` kwarg.

    Lightning calls `optimizer.step(closure=...)`; in manual optimization the
    closure is a no-op (gradients are already populated via manual_backward), so
    it is safely ignored here."""

    def step(self, closure=None):  # type: ignore[override]
        return super().step()

# A parameter is optimized by Muon iff it is a >=2-D weight inside the encoder
# body. Embeddings, the classification head (pre_classifier/classifier), all
# biases and LayerNorm gains are routed to the auxiliary AdamW, following the
# Muon authors' guidance that only hidden weight matrices should use Muon.
BODY_KEY = "transformer"


@dataclass
class ParamSplit:
    muon: list[nn.Parameter]
    aux: list[nn.Parameter]
    muon_names: list[str]
    aux_names: list[str]


def split_muon_params(model: nn.Module, body_key: str = BODY_KEY) -> ParamSplit:
    muon, aux, muon_names, aux_names = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and body_key in name:
            muon.append(p)
            muon_names.append(name)
        else:
            aux.append(p)
            aux_names.append(name)
    return ParamSplit(muon, aux, muon_names, aux_names)


def _count(params: list[nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def build_optimizer(
    name: str, model: nn.Module, hp: OptimHParams
) -> tuple[torch.optim.Optimizer, dict]:
    if name == "adamw":
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            betas=hp.betas,
            eps=hp.eps,
        )
        info = {
            "optimizer": "adamw",
            "adamw_tensors": len(params),
            "adamw_params": _count(params),
            "muon_tensors": 0,
            "muon_params": 0,
        }
        return opt, info

    if name == "muon":
        split = split_muon_params(model)
        if not split.muon:
            raise ValueError("No Muon-eligible parameters found; check body_key/model.")
        muon_group = dict(
            params=split.muon,
            lr=hp.lr,
            momentum=hp.momentum,
            weight_decay=hp.weight_decay,
            use_muon=True,
        )
        adam_group = dict(
            params=split.aux,
            lr=hp.adam_lr,
            betas=hp.betas,
            eps=hp.eps,
            weight_decay=hp.weight_decay,
            use_muon=False,
        )
        opt = MuonWithAuxAdam([adam_group, muon_group])
        info = {
            "optimizer": "muon",
            "muon_tensors": len(split.muon),
            "muon_params": _count(split.muon),
            "adamw_tensors": len(split.aux),
            "adamw_params": _count(split.aux),
        }
        return opt, info

    raise ValueError(f"Unknown optimizer: {name!r}")


def optimizer_state_key(optimizer: torch.optim.Optimizer, group_uses_muon: bool) -> str:
    """Name of the first-moment/momentum state buffer for a given group type."""
    return "momentum_buffer" if group_uses_muon else "exp_avg"
