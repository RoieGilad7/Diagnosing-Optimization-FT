# Methodology — AdamW vs Muon on SST-2

## Data: train / validation / test split

Dataset: `stanfordnlp/sst2` (GLUE SST-2, binary sentiment). GLUE's official
test split has hidden labels, so:

- **Train** — the original `train` split (67,349 examples) minus a fixed,
  seeded 2,000-example held-out slice → ~65,349 examples used for training.
- **Validation** — GLUE's official `validation` split (872 examples). Used
  for model selection and as the Optuna objective (`val_acc`).
- **Test** — the 2,000-example held-out slice of `train` (carved with a fixed
  seed before training). Used only for final reporting.

**Caveat:** `train` sentences average 9.4 words (median 7 — many short
Sentiment-Treebank phrase fragments), vs. `validation`'s 19.5 words (median
19, full sentences only). Since "test" is carved from `train`, it is
systematically easier than `validation`, so `test_acc > val_acc` is an
expected artifact of this dataset, not a modeling error.

## Pre-trained model

`distilbert-base-uncased` (67.0M params), fine-tuned with a freshly
initialized sequence-classification head (`pre_classifier` + `classifier`,
2 labels); DistilBERT's original masked-language-model head is discarded on
load. Inputs tokenized with DistilBERT's tokenizer, `max_length=128`.

Muon is a hybrid: Muon optimizes the encoder body's ≥2-D weight matrices
(36 tensors, ~42.5M params); an auxiliary AdamW optimizes everything else —
embeddings, classification head, biases, LayerNorm gains (68 tensors,
~24.5M params).

## Loss

Standard cross-entropy over 2 classes, computed by the Hugging Face
sequence-classification head from logits + labels.

## Metrics

- **Task:** train loss, val loss, val accuracy (primary metric — model
  selection and Optuna objective), test accuracy (final reporting only).
- **Optimization:** gradient norm, update norm, relative update norm,
  per-group learning rate, update sparsity.
- **Geometry:** momentum–gradient cosine (AdamW's `exp_avg` / Muon's
  `momentum_buffer`), probe-gradient temporal cosine (fixed probe batch).
- **Sharpness** (phase 2 only): filter-normalized random-direction loss
  increase — 30 directions × 3 perturbation sizes (`eps`) × 8 eval batches;
  smaller mean increase ⇒ locally flatter.

## Phase 1 — hyperparameter search (Optuna)

Reduced-fidelity budget: 1000 steps/trial, eval every 100 steps, TPE sampler
(seeded), `MedianPruner` (cuts weak trials early), 30 trials/optimizer.

**Searched:**

| Optimizer | lr | weight_decay | momentum / beta1 |
|---|---|---|---|
| AdamW | log-uniform [5e-6, 1e-4] | uniform [0, 0.3] | beta1, uniform [0.80, 0.99] |
| Muon | log-uniform [1e-4, 5e-2]¹ | uniform [0, 0.3] | momentum, uniform [0.9, 0.99] |

¹ Widened from an initial [1e-3, 1e-1] after that search showed the optimum
pinned to the lower boundary; table reflects the final search.

**Fixed** (both optimizers, both phases): beta2=0.999, eps=1e-8, Muon's
auxiliary-AdamW `adam_lr`=2e-5, `batch_size`=32, linear schedule,
`warmup_ratio`=0.1, `grad_clip`=1.0, seed=42, model/tokenizer as above.

**Raw results:**

| Optimizer | Trials (complete/pruned) | Best val_acc | Best hyperparameters |
|---|---|---|---|
| AdamW | 16 / 14 | 0.9037 | lr=5.83e-5, weight_decay=0.258, beta1=0.867 |
| Muon | 9 / 21 | 0.9071 | lr=1.03e-3, weight_decay=0.285, momentum=0.966 |

## Phase 2 — full training

5,000 steps per run (warmup + linear LR decay calibrated to this budget),
same data/schedule/seed across all four runs. Two experiments per optimizer:

- **Matched** — fixed default hyperparameters (below), isolates the optimizer.
- **Best** — Phase-1 tuned hyperparameters (above).

**Overfitting correction:** an initial 10,000-step run showed val_loss
bottoming out early (~step 1600–1800) then rising substantially while
train_loss collapsed toward zero — classic overfitting, which would also bias
the sharpness scan (computed at the final step) toward whichever optimizer
overfits more. Reported metrics below — including the sharpness scan — are
therefore taken from each run's **best checkpoint** (highest val_acc seen
during training, our primary metric), not the final step.

**Fixed default hyperparameters (Matched experiment):**

| Optimizer | lr | weight_decay | momentum / betas | other |
|---|---|---|---|---|
| AdamW | 2e-5 | 0.01 | betas=(0.9, 0.999) | — |
| Muon | 0.02 | 0.01 | momentum=0.95 | adam_lr=2e-5 (aux) |

**Raw results (best checkpoint):**

| Run | best_step | val_acc | test_acc |
|---|---|---|---|
| AdamW — matched | 1600 | 0.9117 | 0.9350 |
| AdamW — best | 1600 | 0.9106 | 0.9355 |
| Muon — matched | 4400 | 0.8131 | 0.8740 |
| Muon — best | 3800 | 0.9025 | 0.9500 |
