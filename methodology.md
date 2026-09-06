# Methodology — AdamW vs Muon on SST-2

## Data: train / validation / test split

We use `stanfordnlp/sst2` (GLUE SST-2, binary sentiment); since its official
test labels are hidden, we build our own three-way split: **train** is the
original 67,349-example split minus a fixed, seeded 2,000-example slice
(~65,349 examples actually used for training); **validation** is GLUE's
official 872-example split, used for model selection and as the Optuna
objective; **test** is that same 2,000-example slice held out from train,
used only for final reporting.

Worth noting: train's sentences average 9.4 words (many are short
Sentiment-Treebank fragments) vs. validation's 19.5 (full sentences only).
Since test is carved from train, it's systematically easier than
validation — so `test_acc > val_acc` is an expected dataset quirk, not a
modeling error.

## Pre-trained model

We fine-tune `distilbert-base-uncased` (67.0M params), swapping its
masked-language-model head for a fresh, randomly initialized classification
head (`pre_classifier` + `classifier`, 2 labels), and tokenize inputs with
DistilBERT's own tokenizer, truncated to 128 tokens.

For Muon, we only orthogonalize the encoder body's 2-D weight matrices (36
tensors, ~42.5M params) — everything else (embeddings, the classification
head, biases, LayerNorm gains: 68 tensors, ~24.5M params) goes to an
auxiliary AdamW instead.

## Loss

Standard cross-entropy over the two classes, computed by Hugging Face's
sequence-classification head directly from logits and labels.

## Metrics

We track four groups of metrics:

- **Task:** train loss, val loss, val accuracy (our primary metric — used
  for model selection and the Optuna objective), test accuracy (final
  reporting only).
- **Optimization:** gradient norm, update norm, relative update norm,
  per-group learning rate, update sparsity.
- **Geometry:** momentum–gradient cosine (AdamW's `exp_avg` / Muon's
  `momentum_buffer`), probe-gradient temporal cosine (on a fixed batch).
- **Sharpness** (phase 2 only): loss increase under filter-normalized random
  perturbations — 30 directions × 3 sizes (`eps`) × 8 eval batches; a
  smaller mean increase means a locally flatter solution.

## Phase 1 — hyperparameter search (Optuna)

A full 5,000-step run is too expensive to search over, so we tune at a
cheaper 1000-step budget instead (evaluating every 100 steps), using
Optuna's TPE sampler and a `MedianPruner` to cut weak trials short — 30
trials per optimizer.

**Searched:**

| Optimizer | lr | weight_decay | momentum / beta1 |
|---|---|---|---|
| AdamW | log-uniform [5e-6, 1e-4] | uniform [0, 0.3] | beta1, uniform [0.80, 0.99] |
| Muon | log-uniform [1e-4, 5e-2]¹ | uniform [0, 0.3] | momentum, uniform [0.9, 0.99] |

¹ We first searched Muon's lr in [1e-3, 1e-1], but the best trials kept
clustering at the lower edge, so we widened the floor to 1e-4 and re-ran;
this table shows that final search.

Everything else stayed fixed, for both optimizers and both phases: beta2 =
0.999, eps = 1e-8, Muon's auxiliary-AdamW lr = 2e-5, batch size 32, a linear
warmup+decay schedule (10% warmup), gradient clipping at 1.0, and seed = 42.

**Raw results:**

| Optimizer | Trials (complete/pruned) | Best val_acc | Best hyperparameters |
|---|---|---|---|
| AdamW | 16 / 14 | 0.9037 | lr=5.83e-5, weight_decay=0.258, beta1=0.867 |
| Muon | 9 / 21 | 0.9071 | lr=1.03e-3, weight_decay=0.285, momentum=0.966 |

## Phase 2 — full training

Each run trains for 5,000 steps (warmup and linear decay calibrated to that
budget), using the same data, schedule, and seed throughout. For each
optimizer we run two experiments — **matched**, with fixed default
hyperparameters, to isolate the optimizer's effect; and **best**, with its
Phase-1 tuned hyperparameters.

**Overfitting correction:** an earlier 10,000-step run showed val_loss
bottom out around step 1600–1800, then climb back up while train_loss kept
collapsing toward zero — classic overfitting. Since that would also bias the
sharpness scan (normally computed at the final step) toward whichever
optimizer overfits faster, we pull every metric below — sharpness included —
from each run's **best checkpoint** (highest val_acc during training, our
primary metric), not its final step.

**Fixed default hyperparameters (matched experiment):**

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
