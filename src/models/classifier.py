from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from src.data.dataset import Batch


class SequenceClassifier(nn.Module):
    """Thin wrapper around a HF sequence-classification model.

    Keeps the training/metric code decoupled from the HF forward signature and
    exposes a plain (logits, loss) interface over our `Batch` type.
    """

    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        # Let the backbone compute its own loss from `labels` (standard HF
        # pattern) rather than re-deriving CrossEntropyLoss ourselves
        out = self.backbone(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        return out.logits, out.loss

    @torch.no_grad()
    def predictions(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)
