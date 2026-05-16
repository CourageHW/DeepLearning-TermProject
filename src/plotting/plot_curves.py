#!/usr/bin/env python3
"""
Overlay training curves from multiple experiments.

Usage:
    python3 plot_curves.py \
        --inputs history_stu_baseline.csv history_stu_e1_kd0.csv history_stu_e2_singleskip.csv \
        --labels "Baseline" "KD off (E1)" "Single-skip (E2)" \
        --output curves_phase_b.png \
        --title "Phase B (Student) — Training Curves"

If --labels is omitted, file stems are used. Files can be CSV or PKL (auto-detect).
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_METRICS = [
    ("loss",  "train_loss", "val_loss",   "Loss"),
    ("top1",  "train_top1", "val_top1",   "Top-1 (%)"),
]


def load_history(path):
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in (".pkl", ".pickle"):
        with open(p, "rb") as f:
            return pd.DataFrame(pickle.load(f))
    raise ValueError(f"Unsupported file extension: {p.suffix}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs",  nargs="+", required=True, help="history_*.csv / .pkl files")
    ap.add_argument("--labels",  nargs="+", default=None,  help="Legend labels (one per input)")
    ap.add_argument("--output",  default="curves_comparison.png", help="Output image path")
    ap.add_argument("--title",   default="Training Curves", help="Figure title")
    ap.add_argument("--dpi",     type=int, default=150, help="Output DPI")
    ap.add_argument("--metric",  choices=[m[0] for m in PLOT_METRICS] + ["both"],
                    default="both", help="Which metric to plot")
    ap.add_argument("--ema",     action="store_true",
                    help="Overlay EMA val_top1 (only meaningful for Phase A / teacher)")
    args = ap.parse_args()

    if args.labels is None:
        args.labels = [Path(p).stem.replace("history_", "").replace("stu_", "") for p in args.inputs]
    if len(args.labels) != len(args.inputs):
        ap.error(f"--labels length ({len(args.labels)}) != --inputs length ({len(args.inputs)})")

    histories = [load_history(p) for p in args.inputs]

    metrics_to_plot = PLOT_METRICS if args.metric == "both" else [m for m in PLOT_METRICS if m[0] == args.metric]
    n_panels = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col_idx, (key, train_col, val_col, ylabel) in enumerate(metrics_to_plot):
        ax = axes[col_idx]
        for i, (df, label) in enumerate(zip(histories, args.labels)):
            color = color_cycle[i % len(color_cycle)]
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col], linestyle="--",
                        color=color, alpha=0.45, linewidth=1.2)
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col], linestyle="-",
                        color=color, label=label, linewidth=1.8)
            if args.ema and "val_top1_ema" in df.columns and key == "top1":
                ax.plot(df["epoch"], df["val_top1_ema"], linestyle=":",
                        color=color, alpha=0.7, linewidth=1.2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} (dashed=train, solid=val{', dotted=EMA' if args.ema else ''})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output}")

    # Also report final-epoch summary
    print("\n=== Final epoch summary ===")
    for df, label in zip(histories, args.labels):
        last = df.iloc[-1]
        if "val_top1" in df.columns:
            vals = df["val_top1"].to_numpy()
            best_val = float(vals.max())
            best_ep  = int(vals.argmax()) + 1
        else:
            best_val, best_ep = float("nan"), -1
        print(f"  {label:30s} | last_ep={int(last['epoch']):3d} "
              f"final_val={float(last.get('val_top1', float('nan'))):.2f}% "
              f"best_val={best_val:.2f}% (ep {best_ep})")


if __name__ == "__main__":
    main()
