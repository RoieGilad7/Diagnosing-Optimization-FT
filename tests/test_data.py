from __future__ import annotations

import torch

from src.data.dataset import Batch, PadCollator, TokenizedDataset


def test_tokenized_dataset_len_and_getitem():
    enc = {"input_ids": [[1, 2], [3, 4, 5]], "attention_mask": [[1, 1], [1, 1, 1]]}
    ds = TokenizedDataset(enc, labels=[0, 1])
    assert len(ds) == 2
    item = ds[1]
    assert item["input_ids"] == [3, 4, 5]
    assert item["labels"] == 1


def test_pad_collator_pads_to_max_length():
    collate = PadCollator(pad_token_id=0)
    samples = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 0},
        {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": 1},
    ]
    batch = collate(samples)
    assert isinstance(batch, Batch)
    assert batch.input_ids.shape == (2, 3)
    assert torch.equal(batch.input_ids[0], torch.tensor([1, 2, 0]))
    assert torch.equal(batch.attention_mask[0], torch.tensor([1, 1, 0]))
    assert torch.equal(batch.labels, torch.tensor([0, 1]))


def test_batch_to_device_cpu():
    b = Batch(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.tensor([[1]]),
        labels=torch.tensor([0]),
    )
    moved = b.to(torch.device("cpu"))
    assert moved.input_ids.device.type == "cpu"
