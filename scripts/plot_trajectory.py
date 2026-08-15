"""Trajectory figure for ONE run: R_ID / R_tail / R_OOD vs epoch + oracle epochs.

Host-side (no GPU): reads only ``metrics.jsonl``. Each panel draws the RAW logged
risk plus the 3-epoch moving average that the analysis uses to locate the oracle
epoch, and every panel carries all three oracle epochs as vertical rules -- the
divergence between those rules IS the hypothesis under test. The dots on the
off-objective rules are the cross-regret read-off points.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from analysis.io import load_run, risk_trajectories, static_tail_from_reference  # noqa: E402
from analysis.ncr import OBJECTIVES, moving_average, oracle_epoch                # noqa: E402

# Okabe-Ito, a colour-vision-deficiency-safe qualitative set. Fixed order, never cycled.
COLORS = {"ID": "#0072B2", "tail": "#D55E00", "OOD": "#009E73"}
LABELS = {"ID": "$R_{\\mathrm{ID}}$\nclean-test error",
          "tail": "$R_{\\mathrm{tail}}$\nstatic tail-class error",
          "OOD": "$R_{\\mathrm{OOD}}$\n1 - AUROC (energy,\nsemantic pools)"}


def plot_run(run_dir: str, out_path: str, smooth_w: int = 3, tail_reference: str = "") -> str:
    run = load_run(run_dir)
    if not run.epochs:
        raise SystemExit(f"no epoch records in {run_dir}/metrics.jsonl")
    ref = load_run(tail_reference) if tail_reference else run
    tail_classes = static_tail_from_reference(ref)
    risks = risk_trajectories(run, tail_classes)
    epochs = run.series("epoch")

    t_star = {j: oracle_epoch(risks[j], smooth_w)[0] for j in OBJECTIVES}

    fig, axes = plt.subplots(len(OBJECTIVES), 1, figsize=(8.2, 8.4), sharex=True)
    for ax, j in zip(axes, OBJECTIVES):
        y = risks[j]
        ax.plot(epochs, y, lw=2.0, color=COLORS[j], label="raw (logged)")
        ax.plot(epochs, moving_average(y, smooth_w), lw=1.2, ls="--", color=COLORS[j],
                alpha=0.55, label=f"{smooth_w}-epoch MA (analysis only)")
        for k in OBJECTIVES:                       # all three oracle epochs on every panel
            own = (k == j)
            ax.axvline(epochs[t_star[k]], color=COLORS[k], lw=1.6 if own else 1.0,
                       ls="-" if own else ":", alpha=0.9 if own else 0.7, zorder=0)
            ax.plot([epochs[t_star[k]]], [y[t_star[k]]], "o", ms=9 if own else 7,
                    color=COLORS[k], mec="white", mew=1.5, zorder=3)
        ax.annotate(f"$t^*$={int(epochs[t_star[j]])}\n{y[t_star[j]]:.4f}",
                    xy=(epochs[t_star[j]], y[t_star[j]]), xytext=(6, 8),
                    textcoords="offset points", fontsize=9, color="#333333")
        ax.set_ylabel(LABELS[j], fontsize=9, labelpad=8)
        ax.margins(y=0.16)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    axes[-1].set_xlabel("epoch")
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    gaps = "   ".join(f"|t*({a}) - t*({b})| = {abs(t_star[a] - t_star[b])}"
                      for i, a in enumerate(OBJECTIVES) for b in OBJECTIVES[i + 1:])
    m = run.meta
    smoke = "  [SMOKE]" if m.get("smoke") else ""
    co = f"  co-tenant={m['co_tenant']}" if m.get("co_tenant") else ""
    fig.suptitle(f"{run.run_id}{smoke}{co}\noracle-epoch gaps:  {gaps}", fontsize=11)
    fig.text(0.01, 0.005,
             "Solid rule = this panel's own oracle epoch; dotted rules = the other two "
             "objectives' oracle epochs.\nThe dot height on a dotted rule is the risk paid "
             "by deploying that objective's checkpoint here (cross-regret).",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="path to results/runs/<run_id>")
    p.add_argument("--out", default="", help="output png (default results/figures/<run_id>_trajectory.png)")
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--tail-reference", default="",
                   help="run dir defining the cell's static tail set (default: the run itself)")
    a = p.parse_args(argv)
    run_dir = os.path.abspath(a.run)
    out = a.out or os.path.join(ROOT, "results", "figures",
                                f"{os.path.basename(run_dir)}_trajectory.png")
    print(plot_run(run_dir, out, a.smooth_window, a.tail_reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
