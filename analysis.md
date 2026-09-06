# Analysis — AdamW vs Muon on SST-2

Based on `methodology.md` and the Phase 2 runs (5,000 steps, best checkpoint
selected by highest val_acc; see `comparison.png`). Focuses on the tuned
(best) comparison — each optimizer's individually-optimal hyperparameters.
Matched (default-hyperparameter) numbers are used only where they help
interpret the tuned result.

## Selected metrics, and why

- **Gradient norm ‖g‖** — raw signal magnitude driving each update.
- **Relative update magnitude ‖Δθ‖/‖θ‖** — step size relative to parameter
  scale. Muon orthogonalizes/spectrally-normalizes its 2-D weight updates;
  AdamW scales per-element — a structural difference this metric surfaces
  directly.
- **Momentum–gradient cosine** — how aligned each step is with recent
  history; near zero means no consistent net direction of progress.
- **Probe-gradient temporal cosine** — gradient similarity on a fixed batch
  over time; rising toward 1 signals a settled search direction, distinct
  from accuracy.
- **Sharpness scan** (filter-normalized perturbation) — direct curvature
  estimate at the selected checkpoint.
- **Val_loss volatility after the best checkpoint** — a second, empirical
  flatness proxy: how much ordinary training noise still moves the loss once
  "settled."

**Recorded values, tuned (best) runs:**

| run | best_step | val_acc | test_acc | grad_norm (mean) | rel_update (mean) | momentum_cos (mean) | probe_cos (last 25%) | sharpness @ε=0.05 |
|---|---|---|---|---|---|---|---|---|
| adamw_best | 1600 (32%) | 0.911 | 0.936 | 3.81 | 1.31e-4 | 0.002 | 0.932 | −0.0215 |
| muon_best | 3800 (76%) | 0.903 | 0.950 | 2.74 | 2.49e-4 | 0.028 | 0.967 | −0.0080 |

## Which optimizer performed better, and by what criterion?

- **AdamW leads on validation accuracy, by 0.8 points** (0.911 vs 0.903) — a
  real gap, since both runs use each optimizer's own best hyperparameters.
- **test_acc disagrees, favoring Muon instead** (0.950 vs 0.936) — worth
  noting rather than ignoring.
- **We treat val_acc as the deciding metric**: it's measured on GLUE's
  official validation split, which is meaningfully different from the
  training data (full sentences vs. train's many short phrase fragments).
  test_acc, by contrast, is a slice of `train` itself — same distribution
  the model trained on — so it doesn't test generalization the way val_acc
  does.
- **Muon is far more hyperparameter-sensitive**: under default
  hyperparameters the gap widens to 0.912 vs 0.813, since Muon's default
  lr=0.02 sits well outside the ~1e-3 range Phase 1 found effective.

## Which solution appears flatter?

- **AdamW's solution is flatter by both measures**: sharpness scan −0.0215
  vs. −0.0080; val_loss volatility after the best checkpoint 0.0408 vs.
  0.0457 (lower is flatter, both agree).
- **Muon's updates are structurally larger relative to its gradients**: its
  relative update magnitude (2.49e-4) is nearly double AdamW's (1.31e-4)
  despite a *smaller* mean gradient norm (2.74 vs. 3.81) — consistent with
  spectral normalization decoupling step size from raw gradient size.
- **Muon's best checkpoint also came much later** (76% into training vs.
  AdamW's 32%) — but we don't trust this as a real "Muon converges slower"
  finding. Published Muon results (e.g. the nanoGPT speedrun work) report
  *faster*, more sample-efficient convergence than AdamW — the opposite
  direction. Our own reruns of the identical Muon config also landed on very
  different best_steps (400, 1800, 3800 across three attempts), so this
  looks like noise from a single small fine-tuning run, not a generalizable
  effect — especially since Muon's documented speed advantage is established
  in from-scratch pretraining, not fine-tuning an already-converged model.

## How reliable are these conclusions?

- **Run-to-run variation is large**: rerunning identical hyperparameters and
  seed can shift val_acc by several points (MPS training isn't
  bitwise-deterministic) — a swing comparable to the 0.8-point gap above.
- **test_acc's absolute values are inflated**: it's drawn from `train`,
  which averages half `validation`'s sentence length (see methodology.md),
  so its high scores don't reflect true generalization the way val_acc's do.
- **The sharpness scan is noisy**: only 8 eval batches and 30 directions at
  one checkpoint.
- **This is a light NLP task, run once**: single seed, single architecture,
  and SST-2 fine-tuning on an already well-pretrained small model reaches
  ~90%+ almost immediately — these conclusions may not hold on larger
  datasets or genuinely difficult tasks, where the optimizers would have
  more room to differ.

## What would we change in a larger study?

- **Harder, more complex tasks.** Both optimizers reach ~90%+ within a
  fraction of an epoch, near DistilBERT's ~91% published ceiling — the task
  is nearly solved by pretraining alone. A task the model hasn't already
  mostly learned would give the optimizers real work to do and a real chance
  to diverge.
- **Larger architectures with more 2-D weight matrices.** Muon's advantage
  is specifically about orthogonalizing large 2-D matrices; DistilBERT's are
  comparatively small. A bigger model tests that mechanism more fairly.
- **Multiple seeds, a genuinely held-out test set, and a lower-variance
  sharpness estimate** (e.g. Hessian top-eigenvalue via power iteration) —
  these sharpen measurement precision, not the deeper limit that this task
  and model leave little room for optimizers to differ in the first place.
