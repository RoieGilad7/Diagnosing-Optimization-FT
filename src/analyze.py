from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

TRAJECTORY_PLOTS = [
    ("train_loss", "Training loss"),
    ("val_loss", "Validation loss"),
    ("val_acc", "Validation accuracy"),
    ("grad_norm", "Gradient norm"),
    ("relative_update", "Relative update magnitude"),
    ("momentum_grad_cos", "Momentum-gradient cosine"),
    ("probe_grad_cos", "Probe gradient temporal cosine"),
]


def load_runs(runs: dict[str, str]) -> dict[str, pd.DataFrame]:
    out = {}
    for label, path in runs.items():
        df = pd.read_csv(Path(path) / "metrics.csv")
        out[label] = df.apply(pd.to_numeric, errors="coerce")
    return out


def plot_trajectory(dfs: dict[str, pd.DataFrame], column: str, title: str, ax) -> None:
    for label, df in dfs.items():
        if column not in df:
            continue
        sub = df[["step", column]].dropna()
        if sub.empty:
            continue
        ax.plot(sub["step"], sub[column], marker="o", ms=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_sharpness(runs: dict[str, str], ax) -> None:
    plotted = False
    for label, path in runs.items():
        sfile = Path(path) / "summary.json"
        if not sfile.exists():
            continue
        summary = json.loads(sfile.read_text())
        sharp = summary.get("sharpness")
        if not sharp:
            continue
        ax.plot(sharp["eps"], sharp["mean_increase"], marker="o", label=f"{label} (mean)")
        plotted = True
    ax.set_title("Sharpness: loss increase vs perturbation")
    ax.set_xlabel("epsilon (filter-normalized)")
    ax.set_ylabel("mean loss increase")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)


def analyze(runs: dict[str, str], out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dfs = load_runs(runs)

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, TRAJECTORY_PLOTS):
        plot_trajectory(dfs, col, title, ax)
    plot_sharpness(runs, axes[7])
    fig.tight_layout()
    path = out / "comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _parse_run(spec: str) -> tuple[str, str]:
    label, _, path = spec.partition(":")
    return label, path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Plot AdamW vs Muon trajectories")
    p.add_argument("--run", action="append", required=True, help="label:path/to/run_dir")
    p.add_argument("--out", default="outputs/analysis")
    args = p.parse_args()
    runs = dict(_parse_run(s) for s in args.run)
    path = analyze(runs, args.out)
    logger.info("[analyze] wrote %s", path)


if __name__ == "__main__":
    main()
