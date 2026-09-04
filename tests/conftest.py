from __future__ import annotations

import torch
import torch.nn as nn

from src.data.dataset import Batch


class TinyClassifier(nn.Module):
    """Minimal model matching SequenceClassifier's (logits, loss) interface.

    Parameter names mirror the real model so `split_muon_params` (which keys on
    'transformer' and ndim>=2) is exercised: `transformer_layer.weight` is
    Muon-eligible; embeddings / classifier / biases are not.
    """

    def __init__(self, vocab: int = 20, dim: int = 8, n_labels: int = 2) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab, dim)
        self.transformer_layer = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, n_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embeddings(batch.input_ids)
        x = self.transformer_layer(x).mean(dim=1)
        logits = self.classifier(x)
        return logits, self.loss_fn(logits, batch.labels)


def make_batch(bsz: int = 4, seq: int = 5, vocab: int = 20, seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (bsz, seq), generator=g)
    labels = torch.randint(0, 2, (bsz,), generator=g)
    attn = torch.ones(bsz, seq, dtype=torch.long)
    return Batch(input_ids=input_ids, attention_mask=attn, labels=labels)
