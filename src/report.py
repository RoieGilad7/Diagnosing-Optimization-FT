from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load(path: str) -> dict | None:
    f = Path(path) / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def _acc_table(runs: dict[str, dict]) -> str:
    lines = ["| run | optimizer | val_acc | val_loss | test_acc |", "|---|---|---|---|---|"]
    for label, s in runs.items():
        lines.append(
            f"| {label} | {s['optimizer']} | {_fmt(s['val_acc'])} | "
            f"{_fmt(s['val_loss'])} | {_fmt(s['test_acc'])} |"
        )
    return "\n".join(lines)


def _sharpness_line(label: str, s: dict) -> str:
    sh = s.get("sharpness")
    if not sh:
        return f"- {label}: sharpness not recorded"
    pairs = ", ".join(f"eps={e}: {_fmt(m)}" for e, m in zip(sh["eps"], sh["mean_increase"]))
    return f"- {label}: mean loss increase [{pairs}]"


def _better(runs: dict[str, dict], a: str, b: str) -> str:
    if a not in runs or b not in runs:
        return "insufficient data"
    va, vb = runs[a]["val_acc"], runs[b]["val_acc"]
    if abs(va - vb) < 1e-6:
        return "tie"
    return a if va > vb else b


def build_report(matched: dict[str, dict], best: dict[str, dict]) -> str:
    all_runs = {**matched, **best}
    matched_winner = _better(matched, "adamw_matched", "muon_matched")
    best_winner = _better(best, "adamw_best", "muon_best")

    sharp_lines = "\n".join(_sharpness_line(k, v) for k, v in all_runs.items())

    return f"""# AdamW vs Muon on SST-2 — Report

## Results

{_acc_table(all_runs)}

## Sharpness (mean loss increase under filter-normalized perturbation)

{sharp_lines}

Smaller mean increase => locally flatter under this measurement.

## Answers

1. **Which optimizer performed better, and by which criterion?**
   Primary criterion is validation accuracy.
   - Matched setup (same global config): **{matched_winner}**.
   - Individually tuned (Optuna best): **{best_winner}**.

2. **Which solution appears flatter?** Compare the sharpness values above at a
   fixed epsilon; the run with the smaller mean loss increase is locally flatter.

3. **How reliable are these conclusions?** Single seed, few optimizer steps, a
   small data subset in smoke runs, and only a handful of perturbation
   directions. Treat differences as indicative, not conclusive.

4. **Limitations.** GLUE SST-2 test labels are withheld, so "test" here is a
   held-out slice of train. Muon's spectral learning rate and AdamW's learning
   rate live on different scales, so the matched comparison cannot match the raw
   LR; it matches budget, schedule, data, and seed instead. Sharpness and
   gradient-geometry metrics are estimates on small probe/eval batches.

5. **What to change in a larger study.** Multiple seeds with confidence
   intervals, the full training set, more optimizer steps, more perturbation
   directions (and a Hessian-eigenvalue cross-check), and a wider Optuna budget.

## Comparison type

- **Controlled/matched** isolates the optimizer change (same budget/schedule/seed).
- **Individually tuned** compares each optimizer's best practical result.

This is a small, lightweight study; it does not claim either optimizer is
universally better.
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Generate the one-page AdamW vs Muon report")
    p.add_argument("--exp-dir", default="outputs/experiments")
    p.add_argument("--out", default="outputs/report.md")
    args = p.parse_args()

    base = Path(args.exp_dir)
    matched = {
        k: _load(base / k)
        for k in ["adamw_matched", "muon_matched"]
        if _load(base / k)
    }
    best = {
        k: _load(base / k) for k in ["adamw_best", "muon_best"] if _load(base / k)
    }
    report = build_report(matched, best)
    Path(args.out).write_text(report)
    logger.info("[report] wrote %s", args.out)


if __name__ == "__main__":
    main()
