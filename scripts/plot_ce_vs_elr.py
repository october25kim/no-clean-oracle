"""CE vs ELR on shared axes: memorization + the three audited risks.

Host-side (no GPU); reads only ``metrics.jsonl``. Both runs must be the same
(dataset x noise x seed) cell, and both use that cell's static tail-class set, which
is defined by the CE seed-0 run's final epoch.

Rows 1-2 are the memorization signal.

Row 1 is the fraction of NOISY training labels the model currently predicts, against
the fraction of those labels that are actually clean. Fitting up to the clean fraction
is learning; going above it is memorizing corrupted labels, so the dashed reference
line is the memorization threshold.

Row 2 is the per-sample cross-entropy on the noisy labels at the median and the 90th
percentile. Reading memorization off the MEDIAN alone is misleading at high noise:
with ~40% corruption the median sample is a correctly-labelled one, so a learner that
fits the clean majority harder shows a LOWER median while refusing to fit the
corrupted tail (which shows as a much higher p90). Both curves are the plain CE term
for both learners, unlike ``train_loss_noisy``, which is whatever objective each
learner actually optimises (CE for CE, CE + lambda * regularizer for ELR).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from analysis.io import load_run, risk_trajectories, static_tail_from_reference  # noqa: E402
from analysis.ncr import OBJECTIVES, oracle_epoch                                # noqa: E402
from data.noise import NoiseConfig                                               # noqa: E402

LEARNER_COLOR = {"ce": "#0072B2", "elr": "#D55E00"}       # Okabe-Ito, fixed order
ROWS = ["fit", "loss"] + list(OBJECTIVES)
ROW_LABELS = {
    "fit": "fraction of NOISY train\nlabels predicted",
    "loss": "per-sample CE on\nNOISY labels",
    "ID": "$R_{\\mathrm{ID}}$\nclean-test error",
    "tail": "$R_{\\mathrm{tail}}$\nstatic tail-class error",
    "OOD": "$R_{\\mathrm{OOD}}$\n1 - AUROC (energy,\nsemantic pools)",
}


def _quantiles(run, key: str) -> np.ndarray:
    return np.asarray([e["train_loss_quantiles"][key] for e in run.epochs], dtype=np.float64)


def _noisy_label_fit_rate(run) -> np.ndarray:
    """Share of training samples whose prediction equals the (possibly wrong) label."""
    return np.asarray([np.nanmean(e["per_class_train_acc"]) for e in run.epochs],
                      dtype=np.float64)


def clean_label_fraction(run) -> float:
    """1 - empirical flip rate of the cell's saved mask: the memorization threshold."""
    m = run.meta
    cfg = NoiseConfig(m["dataset"], m["noise_type"], float(m["eta"]), int(m["seed"]))
    path = os.path.join(ROOT, "results", "noise_masks", f"{cfg.name}.npz")
    d = np.load(path)
    return float((d["clean"] == d["noisy"]).mean())


def plot_compare(ce_dir: str, elr_dir: str, out_path: str, smooth_w: int = 3) -> str:
    runs = {"ce": load_run(ce_dir), "elr": load_run(elr_dir)}
    for name, r in runs.items():
        if r.meta.get("learner") != name:
            raise SystemExit(f"{r.run_dir} has learner={r.meta.get('learner')!r}, expected {name!r}")
    if runs["ce"].cell != runs["elr"].cell:
        raise SystemExit(f"cell mismatch: {runs['ce'].cell} vs {runs['elr'].cell}")

    tail_classes = static_tail_from_reference(runs["ce"])      # the cell's fixed tail set
    risks = {k: risk_trajectories(r, tail_classes) for k, r in runs.items()}

    fig, axes = plt.subplots(len(ROWS), 1, figsize=(8.4, 12.8), sharex=True)
    clean_frac = clean_label_fraction(runs["ce"])
    axes[0].axhline(clean_frac, color="#444444", lw=1.2, ls="--",
                    label=f"clean-label fraction = {clean_frac:.3f}\n(above this = memorizing)")

    for k, r in runs.items():
        ep = r.series("epoch")
        c = LEARNER_COLOR[k]
        lbl = k.upper()
        if k == "elr":
            p = r.meta.get("learner_params", {})
            lbl += f"  ($\\lambda$={p.get('lambda')}, $\\beta$={p.get('beta')})"

        axes[0].plot(ep, _noisy_label_fit_rate(r), lw=2.0, color=c, label=lbl)
        axes[1].plot(ep, _quantiles(r, "p50"), lw=2.0, color=c, label=f"{k.upper()} p50")
        axes[1].plot(ep, _quantiles(r, "p90"), lw=1.6, ls="--", color=c,
                     label=f"{k.upper()} p90")
        for ax, j in zip(axes[2:], OBJECTIVES):
            y = risks[k][j]
            t = oracle_epoch(y, smooth_w)[0]
            ax.plot(ep, y, lw=2.0, color=c, label=lbl)
            ax.axvline(ep[t], color=c, lw=1.2, ls=":", alpha=0.8, zorder=0)
            ax.plot([ep[t]], [y[t]], "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
            ax.annotate(f"$t^*$={int(ep[t])}", xy=(ep[t], y[t]), xytext=(5, 7),
                        textcoords="offset points", fontsize=8, color=c)

    for ax, key in zip(axes, ROWS):
        ax.set_ylabel(ROW_LABELS[key], fontsize=9, labelpad=8)
        ax.margins(y=0.16)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower right")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    axes[-1].set_xlabel("epoch")
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    m = runs["ce"].meta
    smoke = "  [SMOKE]" if m.get("smoke") else ""
    co = f"  co-tenant={m['co_tenant']}" if m.get("co_tenant") else ""
    fig.suptitle(f"CE vs ELR — {runs['ce'].cell} seed{m['seed']}{smoke}{co}", fontsize=12)
    fig.text(0.01, 0.004,
             "Rows 1-2 use the plain CE term on the noisy labels for BOTH learners, not each "
             "learner's own objective, so memorization is comparable.\nAt this noise level the "
             "median sample is correctly labelled, so p50 tracks fitting the clean majority while "
             "p90 tracks refusal to fit the corrupted tail.\nDotted rules mark each learner's "
             "oracle epoch per objective; the static tail-class set is the CE run's final-epoch set.",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.032, 1, 0.972))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ce", required=True, help="path to the CE run dir")
    p.add_argument("--elr", required=True, help="path to the ELR run dir")
    p.add_argument("--out", default="")
    p.add_argument("--smooth-window", type=int, default=3)
    a = p.parse_args(argv)
    ce, elr = os.path.abspath(a.ce), os.path.abspath(a.elr)
    out = a.out or os.path.join(ROOT, "results", "figures",
                                f"{os.path.basename(ce)}_vs_elr.png")
    print(plot_compare(ce, elr, out, a.smooth_window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
