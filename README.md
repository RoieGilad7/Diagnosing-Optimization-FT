# Diagnosing Optimization: AdamW vs Muon on SST-2

A lightweight, reproducible study of how **AdamW** and **Muon** differ during
supervised fine-tuning of `distilbert-base-uncased` on SST-2. The goal is not
maximum accuracy — it is to observe and compare optimization *behavior*:
optimization dynamics, gradient/momentum geometry, parameter updates,
convergence, local sharpness, and final task performance.

## What "Muon" means here

Muon only orthogonalizes 2-D hidden weight matrices. The Muon configuration is
therefore a **hybrid**:

```
Muon(encoder-body 2-D weight matrices)  +  AdamW(everything else)
```

"Everything else" = token/position embeddings, the classification head
(`pre_classifier`, `classifier`), all biases and LayerNorm gains. The exact
partition is logged in each run's `summary.json` (`param_groups`) — e.g. for
DistilBERT: 36 tensors / ~42.5M params to Muon, 68 tensors / ~24.5M to AdamW.
The upstream implementation is the official [`muon-optimizer`](https://github.com/KellerJordan/Muon)
(`SingleDeviceMuonWithAuxAdam`).

## Setup

Requires Python ≥ 3.11 (tested on 3.13, macOS/MPS).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .            # installs pinned deps from pyproject.toml
# or: pip install -e ".[dev]"  to include pytest
```

Device is auto-selected: CUDA → MPS → CPU.

## Data

The GLUE SST-2 `test` split has withheld labels (`-1`), so it cannot be scored
locally. This repo uses `stanfordnlp/sst2`:

- `train` (minus a held-out slice) → training
- `validation` → model selection and the Optuna objective (**primary metric:
  validation accuracy**)
- a fixed, seeded **held-out slice of `train`** (`data.test_from_train`) → final
  "test" reporting

A small deterministic **probe batch** is drawn from `validation` for gradient
temporal-stability measurements.

## Running

The single entry point is `python -m src.train`. Add `--quick` to any phase
for a tiny smoke configuration.

### Phase 0 — sanity check

Fast end-to-end check (forward/backward/step, checkpoint save+load, metric
logging). Run before anything else.

```bash
python -m src.train --optimizer adamw --phase 0
python -m src.train --optimizer muon  --phase 0
```

### Phase 1 — hyperparameter tuning (Optuna)

Small search space — AdamW: lr, weight decay; Muon: lr, weight decay, momentum.
Both optimizers get the same trial budget, steps, data, batch size, schedule,
and seed. Writes `study.db` and `best_config.yaml`.

```bash
python -m src.train --optimizer adamw --phase 1 --trials 20
python -m src.train --optimizer muon  --phase 1 --trials 20
# -> outputs/optuna/{adamw,muon}/best_config.yaml, study.db
```

### Phase 2 — full experiments

**Experiment A (matched):** same global config for both, default per-optimizer
hyperparameters — isolates the effect of the optimizer.

```bash
python -m src.train --optimizer adamw --phase 2   # -> outputs/experiments/adamw_matched
python -m src.train --optimizer muon  --phase 2   # -> outputs/experiments/muon_matched
```

**Experiment B (individually tuned):** each optimizer's Optuna best.

```bash
python -m src.train --optimizer adamw --phase 2 --config outputs/optuna/adamw/best_config.yaml
python -m src.train --optimizer muon  --phase 2 --config outputs/optuna/muon/best_config.yaml
# -> outputs/experiments/{adamw,muon}_best
```

Each Phase 2 run writes `metrics.csv`, `last.ckpt`, `config.yaml`, and
`summary.json` (includes the sharpness scan and software versions).

### Analysis & report

```bash
python -m src.analyze \
  --run adamw_matched:outputs/experiments/adamw_matched \
  --run muon_matched:outputs/experiments/muon_matched \
  --out outputs/analysis/matched

python -m src.report   # -> outputs/report.md
```

`analyze` produces one figure with 8 panels: training loss, validation loss,
validation accuracy, gradient norm, relative update magnitude, momentum–gradient
cosine, probe gradient temporal cosine, and the sharpness comparison.

## Metrics collected

Logged to `metrics.csv` every `metric_every` / `eval_every` / `probe_every`
optimizer steps (not every update):

- **Task:** train loss, val loss, val accuracy, final test accuracy.
- **Optimization:** gradient norm ‖g‖, update norm ‖Δθ‖, relative update
  ‖Δθ‖/‖θ‖, per-group learning rate, update sparsity
  `#{|Δθ_i| > τ|θ_i|}/N` (τ=0.1).
- **Geometry:** momentum–gradient cosine `cos(m_{t-1}, g_t)` measured over the
  encoder-body 2-D weights (the Muon-eligible subset), using AdamW's `exp_avg`
  or Muon's `momentum_buffer`; probe gradient temporal cosine
  `cos(g_t^probe, g_{t-k}^probe)` on the fixed probe batch with dropout disabled.
- **Sharpness:** filter-normalized random perturbations at several epsilons and
  directions; reports mean/max/std loss increase. Smaller ⇒ locally flatter.

All gradient/momentum reads happen **before** `optimizer.step()`, because Muon
and the aux-Adam mutate their momentum buffers (and Muon mutates the gradient)
in place during the step.

## Reproducibility

`seed_everything` seeds Python/NumPy/Torch and enables deterministic algorithms
(best-effort; MPS/CUDA are not bitwise-deterministic for all ops). Each run
stores its full `config.yaml` and `summary.json` (with software versions)
alongside `metrics.csv`. Dependencies are pinned in `pyproject.toml`.

## Testing

```bash
pytest            # 24 tests; test_model.py auto-skips if weights can't be fetched
```

Covers config loading, the data collator, optimizer parameter grouping and state
keys, the metric calculations, gradient cosine similarity, and the sharpness
scan (including parameter restoration).

## Layout

```
configs/         default.yaml, adamw.yaml, muon.yaml
src/data/        dataset.py (Batch, collator), datamodule.py (SST2 + probe)
src/models/      classifier.py (DistilBERT wrapper)
src/optim/       factory.py (param grouping + optimizer build), Muon adapter
src/metrics/     optimization.py, geometry.py, sharpness.py  (pure functions)
src/training/    lit_module.py (manual-opt LightningModule), metrics_collector.py, runner.py
src/train.py     CLI: phases 0/1/2 + Optuna tuning (single entry point)
src/analyze.py   plots                  src/report.py  one-page report
tests/           unit tests + tiny hermetic model in conftest.py
```

## Caveats

This is a small, lightweight study. Results from smoke runs (`--quick`) are
noisy by design. Do not conclude that either optimizer is universally better;
Muon's spectral learning rate and AdamW's learning rate are on different scales,
so the matched comparison matches budget/schedule/data/seed, not raw LR.
```
