from __future__ import annotations

import pytest
import torch

from src.data.dataset import Batch


@pytest.fixture(scope="module")
def classifier():
    """Loads the real DistilBERT. Skips if weights cannot be fetched (offline)."""
    from src.models.classifier import SequenceClassifier

    try:
        return SequenceClassifier("distilbert-base-uncased", num_labels=2)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"model unavailable: {exc}")


def test_forward_shapes_and_loss(classifier):
    batch = Batch(
        input_ids=torch.tensor([[101, 2054, 102], [101, 2003, 102]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
        labels=torch.tensor([0, 1]),
    )
    logits, loss = classifier(batch)
    assert logits.shape == (2, 2)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_backward_produces_gradients(classifier):
    batch = Batch(
        input_ids=torch.tensor([[101, 2054, 102]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        labels=torch.tensor([1]),
    )
    classifier.zero_grad()
    _, loss = classifier(batch)
    loss.backward()
    grads = [p.grad for p in classifier.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)
