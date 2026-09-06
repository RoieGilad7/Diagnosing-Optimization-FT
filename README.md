# Diagnosing Optimization: AdamW vs Muon on SST-2

A study of how **AdamW** and **Muon** differ when fine-tuning
`distilbert-base-uncased` on SST-2 — not just final accuracy, but
optimization dynamics, gradient/momentum geometry, convergence, and local
sharpness. Muon is used as a hybrid (it only orthogonalizes 2-D hidden
weight matrices): `Muon(encoder-body weights) + AdamW(everything else)`.

## Overview

The code has one CLI (`src/train.py`) driving two phases:

- **Phase 1 — hyperparameter search.** A small, reduced-fidelity Optuna
  search (TPE + pruning) over each optimizer's key hyperparameters (lr,
  weight decay, momentum/beta1), cheap enough to run many trials.
- **Phase 2 — full training.** Each optimizer trained to a genuinely
  converged point, twice: once with **matched** (default, untuned)
  hyperparameters to isolate the optimizer's effect, once with each
  optimizer's **best** (Phase-1 tuned) hyperparameters. Every metric of
  interest — accuracy, loss, gradient/update statistics, sharpness — is
  computed at each run's best checkpoint (highest val_acc), not an arbitrary
  final step.

A third, optional Phase 0 does a fast sanity check (forward/backward/step,
checkpoint save+load) before committing to a real run.

## Setup

Requires Python ≥ 3.11 (tested on 3.13, macOS/MPS; device auto-selects
CUDA → MPS → CPU).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .            # pinned deps from pyproject.toml
# pip install -e ".[dev]"   # add this instead to also get pytest
```

This also installs three console scripts — `sst2-train`, `sst2-analyze`,
`sst2-report` — used below (equivalent to `python -m src.train`, etc., if
you'd rather not install the package).

## How to run

```bash
# Phase 1: search AdamW's hyperparameters (30 trials, ~1-2h)
sst2-train --optimizer adamw --phase 1 --trials 30

# Phase 2: full training, default hyperparameters ("matched")
sst2-train --optimizer adamw --phase 2

# Phase 2: full training, this optimizer's tuned hyperparameters ("best")
sst2-train --optimizer adamw --phase 2 --config outputs/optuna/adamw/best_config.yaml

# Plot all runs' trajectories + sharpness on one figure
sst2-analyze --run adamw:outputs/experiments/adamw_matched \
             --run muon:outputs/experiments/muon_matched \
             --out outputs/analysis
```

Swap `adamw` ↔ `muon` for the other optimizer. Add `--quick` to any phase for
a tiny smoke-test config. `.vscode/launch.json` has ready-made debug configs
for all of the above (quick and full variants) if you're in VS Code.

## Where to find the results

| what | where |
|---|---|
| Methodology (data split, model, metrics, hyperparameters searched/fixed, raw result tables) | [`methodology.md`](methodology.md) |
| Analysis (metric selection rationale, which optimizer won, flatness, reliability, what we'd change) | [`analysis.md`](analysis.md) |
| Per-run raw results (`metrics.csv` trajectory, `summary.json`) | `outputs/experiments/{adamw,muon}_{matched,best}/` |
| Optuna search results (`study.db`, per-trial logs, `best_config.yaml`) | `outputs/optuna/{adamw,muon}/` |
| Comparison plots (`comparison.png`: loss/accuracy/gradient/sharpness trajectories, all runs overlaid) | `outputs/analysis/` (and `best/`, `matched/` subfolders for 2-run views) |
| Auto-generated one-page results table | `outputs/report.md` |

## Project layout

```
configs/         default.yaml, adamw.yaml, muon.yaml
src/data/        dataset.py (Batch, collator), datamodule.py (SST2 + probe)
src/models/      classifier.py (DistilBERT wrapper)
src/optim/       factory.py (param grouping + optimizer build), Muon adapter
src/metrics/     optimization.py, geometry.py, sharpness.py  (pure functions)
src/training/    lit_module.py (manual-opt LightningModule), metrics_collector.py, runner.py
src/train.py     CLI: phases 0/1/2 + Optuna tuning (single entry point)
src/analyze.py   trajectory/sharpness plots        src/report.py  results table
tests/           unit tests + tiny hermetic model in conftest.py
methodology.md   how the study was run, and its raw results
analysis.md      the actual findings and their reliability
```

## Testing

```bash
pytest            # 24 tests; test_model.py auto-skips if weights can't be fetched
```

## Caveats

This is a small, lightweight study — single seed, one architecture, one
(easy, near-saturated) task. See `analysis.md` for the full reliability
discussion and what a larger study would change.
