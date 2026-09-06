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

# (column, title, y-scale, extra set_yscale kwargs). grad_norm/relative_update
# commonly span orders of magnitude across runs (an unstable run can dwarf
# well-behaved ones on a linear axis) -- symlog keeps every run legible
# without breaking on near-zero values the way a plain log scale would.
# Both are always >=0, so an explicit small linthresh (rather than symlog's
# auto-picked one, which produced an ugly offset-notation axis label on this
# data) keeps the "linear near zero" region tight and the tick labels plain.
TRAJECTORY_PLOTS = [
    ("train_loss", "Training loss", "linear", {}),
    ("val_loss", "Validation loss", "linear", {}),
    ("val_acc", "Validation accuracy", "linear", {}),
    ("grad_norm", "Gradient norm", "symlog", {"linthresh": 1e-1}),
    ("relative_update", "Relative update magnitude", "symlog", {"linthresh": 1e-5}),
    ("momentum_grad_cos", "Momentum-gradient cosine", "linear", {}),
    ("probe_grad_cos", "Probe gradient temporal cosine", "linear", {}),
]

# Staggered label offsets (points) so stars that land on the same step across
# runs don't have their text collide.
_LABEL_OFFSETS = [(4, 6), (4, -14), (-48, 6), (-48, -14)]


def load_runs(runs: dict[str, str]) -> dict[str, pd.DataFrame]:
    out = {}
    for label, path in runs.items():
        df = pd.read_csv(Path(path) / "metrics.csv")
        out[label] = df.apply(pd.to_numeric, errors="coerce")
    return out


def load_summaries(runs: dict[str, str]) -> dict[str, dict]:
    out = {}
    for label, path in runs.items():
        sfile = Path(path) / "summary.json"
        out[label] = json.loads(sfile.read_text()) if sfile.exists() else {}
    return out


def _mark_best_checkpoint(ax, sub: pd.DataFrame, column: str, summary: dict, color, offset_idx: int) -> None:
    """Star-marks the (best_step, value) point on this run's curve and labels
    it with test_acc -- the model actually used for final reporting/sharpness,
    which may differ from the run's last step (see overfitting correction)."""
    best_step, test_acc = summary.get("best_step"), summary.get("test_acc")
    if best_step is None or test_acc is None:
        return
    idx = (sub["step"] - best_step).abs().idxmin()
    x, y = sub.loc[idx, "step"], sub.loc[idx, column]
    ax.plot(x, y, marker="*", ms=14, mec="black", mew=0.6, color=color, zorder=5)
    xytext = _LABEL_OFFSETS[offset_idx % len(_LABEL_OFFSETS)]
    ax.annotate(f"test={test_acc:.3f}", (x, y), textcoords="offset points",
                xytext=xytext, fontsize=7, color=color)


def plot_trajectory(
    dfs: dict[str, pd.DataFrame], column: str, title: str, ax,
    summaries: dict[str, dict] | None = None, yscale: str = "linear", yscale_kwargs: dict | None = None,
) -> None:
    summaries = summaries or {}
    for i, (label, df) in enumerate(dfs.items()):
        if column not in df:
            continue
        sub = df[["step", column]].dropna()
        if sub.empty:
            continue
        line, = ax.plot(sub["step"], sub[column], marker="o", ms=3, label=label)
        _mark_best_checkpoint(ax, sub, column, summaries.get(label, {}), line.get_color(), i)
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_yscale(yscale, **(yscale_kwargs or {}))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_sharpness(runs: dict[str, str], ax, summaries: dict[str, dict] | None = None) -> None:
    summaries = summaries if summaries is not None else load_summaries(runs)
    plotted = False
    for label in runs:
        summary = summaries.get(label, {})
        sharp = summary.get("sharpness")
        if not sharp:
            continue
        test_acc = summary.get("test_acc")
        suffix = f", test={test_acc:.3f}" if test_acc is not None else ""
        ax.plot(sharp["eps"], sharp["mean_increase"], marker="o", label=f"{label} (mean{suffix})")
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
    summaries = load_summaries(runs)

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    axes = axes.flatten()
    for ax, (col, title, yscale, yscale_kwargs) in zip(axes, TRAJECTORY_PLOTS):
        plot_trajectory(dfs, col, title, ax, summaries=summaries, yscale=yscale, yscale_kwargs=yscale_kwargs)
    plot_sharpness(runs, axes[7], summaries=summaries)
    fig.suptitle("★ = selected (highest val_acc) checkpoint -- label is test_acc there", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
