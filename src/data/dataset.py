from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


class TokenizedDataset(Dataset):
    """Wraps pre-tokenized encodings + labels into a torch Dataset."""

    def __init__(self, encodings: dict[str, list], labels: list[int]) -> None:
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, list | int]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class PadCollator:
    """Dynamic padding collator producing a `Batch` of tensors."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict]) -> Batch:
        max_len = max(len(s["input_ids"]) for s in samples)
        input_ids, attn, labels = [], [], []
        for s in samples:
            ids = s["input_ids"]
            mask = s["attention_mask"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attn.append(mask + [0] * pad)
            labels.append(s["labels"])
        return Batch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attn, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )
