# Diagnosing Optimization: AdamW vs Muon on SST-2

The final one page report located under the root dir (Analysis — AdamW vs Muon on SST-2.pdf)

## Overview

The code has one CLI (`src/train.py`) driving two phases:

- **Phase 1 — hyperparameter search.** A small Optuna
  search (TPE + pruning) over each optimizer's key hyperparameters (lr,
  weight decay, momentum/beta1).
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
pip install -r requirements.txt
# pip install pytest==9.1.1       # add this too if you want to run the tests
```

## How to run

```bash
# Phase 1: search AdamW's hyperparameters (30 trials, ~1-2h)
python -m src.train --optimizer adamw --phase 1 --trials 30

# Phase 2: full training, default hyperparameters ("matched")
python -m src.train --optimizer adamw --phase 2

# Phase 2: full training, this optimizer's tuned hyperparameters ("best")
python -m src.train --optimizer adamw --phase 2 --config outputs/optuna/adamw/best_config.yaml

# Plot all runs' trajectories + sharpness on one figure
python -m src.analyze --run adamw:outputs/experiments/adamw_matched \
                       --run muon:outputs/experiments/muon_matched \
                       --out outputs/analysis
```

Swap `adamw` ↔ `muon` for the other optimizer. Add `--quick` to any phase for
a tiny smoke-test config.

## Where to find the results

| what | where |
|---|---|
| Methodology (data split, model, metrics, hyperparameters searched/fixed, raw result tables) | [`methodology.md`](methodology.md) |
| Analysis (metric selection rationale, which optimizer won, flatness, reliability, what we'd change) | [`Analysis — AdamW vs Muon on SST-2.pdf`] - The one page submission |
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
```

## Testing

```bash
pytest            # 24 tests; test_model.py auto-skips if weights can't be fetched
```
