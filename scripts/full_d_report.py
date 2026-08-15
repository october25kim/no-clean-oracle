"""full-D: loss-distribution separation from per-sample losses (B2 forward pass).

EXPLORATORY. Added after observing the coarse-D null, to test the proxy-limitation
hypothesis -- that five logged quantiles were blind to the two-mode structure the
method depends on, not that the structure is absent. It is not part of the
confirmatory set, and both outcomes are reported the same way:

  non-null -> supports the proxy-limitation interpretation
  null     -> updates toward absence of effect. Do NOT reframe or go looking for a
              slicing that rescues it without a follow-up instruction.

Method. At each retained checkpoint, per-sample cross-entropy against the NOISY label
is computed from the stored logits, and a two-component Gaussian mixture is fitted to
those 50,000 values -- the fit coarse-D could not do. Separation is reported three
ways, all read off one epoch at a time so nothing here is inter-epoch churn:

  gmm_separation  |mu_hi - mu_lo| / sqrt((var_hi + var_lo) / 2)      (effect size)
  gmm_auc         AUC of the low-component posterior at separating the truly clean
                  from the truly mislabeled samples -- an oracle diagnostic, using
                  labels a real selector would not have, reported to show what the
                  loss distribution could support at best
  quantile_proxy  p90 - p50, the coarse-D statistic recomputed on THESE losses, so
                  coarse-D and full-D differ only in what the statistic sees

The selected epoch is the peak of each statistic, scored exactly like every other
baseline via analysis.selection.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.io import load_run, static_tail_from_reference   # noqa: E402
from analysis.ncr import OBJECTIVES, moving_average            # noqa: E402
from analysis.selection import regret_at_epoch                 # noqa: E402

ANALYSIS_CLASS = "exploratory"
REASON = ("post-hoc follow-up to coarse-D null; proxy-limitation hypothesis. "
          "non-null outcome does not isolate resolution as the cause; full-D differs "
          "from the coarse-D source (train_loss_quantiles) in transform (eval vs "
          "augmented) and mode (eval vs train) as well as resolution.")


def per_sample_ce(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(z).sum(axis=1))
    return logZ - z[np.arange(len(labels)), labels]


def fit_two_component(x: np.ndarray, iters: int = 200, tol: float = 1e-7) -> dict:
    """1-D two-component Gaussian mixture by EM, seeded at the loss median split."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mu = np.array([x[x <= med].mean(), x[x > med].mean()])
    var = np.array([max(x[x <= med].var(), 1e-8), max(x[x > med].var(), 1e-8)])
    pi = np.array([0.5, 0.5])
    prev = -np.inf
    for _ in range(iters):
        d = np.stack([pi[k] * np.exp(-0.5 * (x - mu[k]) ** 2 / var[k])
                      / np.sqrt(2 * np.pi * var[k]) for k in range(2)])
        tot = d.sum(axis=0) + 1e-300
        r = d / tot
        ll = float(np.log(tot).mean())
        nk = r.sum(axis=1) + 1e-12
        pi = nk / len(x)
        mu = (r * x).sum(axis=1) / nk
        var = np.maximum((r * (x - mu[:, None]) ** 2).sum(axis=1) / nk, 1e-8)
        if abs(ll - prev) < tol:
            break
        prev = ll
    lo = int(np.argmin(mu))
    hi = 1 - lo
    sep = abs(mu[hi] - mu[lo]) / np.sqrt((var[hi] + var[lo]) / 2.0)
    return dict(mu_lo=float(mu[lo]), mu_hi=float(mu[hi]), var_lo=float(var[lo]),
                var_hi=float(var[hi]), pi_lo=float(pi[lo]), pi_hi=float(pi[hi]),
                separation=float(sep), posterior_lo=r[lo], loglik=float(prev))


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUC of ``scores`` at ranking ``positive`` above the rest (rank statistic)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = int(positive.sum()); nneg = len(scores) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--forward", default=os.path.join(ROOT, "results", "forward_b2"))
    p.add_argument("--runs", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "full_d_exploratory.json"))
    a = p.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])

    run_dirs = sorted(d for d in glob.glob(os.path.join(a.forward, "*"))
                      if os.path.isdir(d) and os.path.isfile(os.path.join(d, "labels.npz")))
    per_run: List[dict] = []

    for fdir in run_dirs:
        run_id = os.path.basename(fdir)
        run = load_run(os.path.join(a.runs, run_id))            # G1 frame, certified
        group_ce0 = os.path.join(a.runs, f"{run.meta['dataset']}_{run.meta['noise_type']}"
                                          f"{run.meta['eta']:g}_ce_seed0")
        tail = static_tail_from_reference(load_run(group_ce0))
        lab = np.load(os.path.join(fdir, "labels.npz"))
        noisy, clean = lab["train_noisy"], lab["train_clean"]
        mislabeled = noisy != clean

        eps, stats = [], {"gmm_separation": [], "gmm_auc": [], "quantile_proxy": []}
        for epdir in sorted(glob.glob(os.path.join(fdir, "ep*"))):
            epoch = int(os.path.basename(epdir)[2:])
            logits = np.load(os.path.join(epdir, "logits_train.npy"))
            ce = per_sample_ce(logits, noisy)
            fit = fit_two_component(ce)
            eps.append(epoch)
            stats["gmm_separation"].append(fit["separation"])
            stats["gmm_auc"].append(_auc(-fit["posterior_lo"], mislabeled))
            stats["quantile_proxy"].append(float(np.percentile(ce, 90) - np.percentile(ce, 50)))

        eps = np.asarray(eps)
        rec = dict(run_id=run_id, cell=run.cell, learner=run.meta["learner"],
                   seed=run.meta["seed"], epochs=eps.tolist(), statistics={})
        for name, vals in stats.items():
            v = np.asarray(vals, dtype=np.float64)
            sm = moving_average(v, smooth_w)
            stop = int(eps[int(np.nanargmax(sm))])
            rec["statistics"][name] = dict(
                curve=v.tolist(), stop_epoch=stop,
                peak=float(np.nanmax(v)), first=float(v[0]), last=float(v[-1]),
                objectives=regret_at_epoch(run, tail, stop, smooth_w))
        per_run.append(rec)
        print(f"[full-D] {run_id}: " + "  ".join(
            f"{k} stop={rec['statistics'][k]['stop_epoch']}" for k in stats), flush=True)

    summary = {}
    for name in ("gmm_separation", "gmm_auc", "quantile_proxy"):
        stops = np.array([r["statistics"][name]["stop_epoch"] for r in per_run])
        summary[name] = dict(
            median_stop_epoch=float(np.median(stops)),
            min_stop_epoch=int(stops.min()), max_stop_epoch=int(stops.max()),
            median_normalized_regret={
                j: float(np.median([r["statistics"][name]["objectives"][j]["normalized_regret"]
                                    for r in per_run])) for j in OBJECTIVES},
            median_epoch_gap={
                j: float(np.median([r["statistics"][name]["objectives"][j]["epoch_gap"]
                                    for r in per_run])) for j in OBJECTIVES})

    print()
    print(f"{'statistic':16s} {'median':>7s} | {'median epoch gap':^22s} | "
          f"{'median normalized regret':^24s}")
    print(f"{'':16s} {'stop':>7s} | " + "  ".join(f"{j:>6s}" for j in OBJECTIVES)
          + "  | " + "  ".join(f"{j:>6s}" for j in OBJECTIVES))
    for name, s in summary.items():
        print(f"{name:16s} {s['median_stop_epoch']:7.0f} | "
              + "  ".join(f"{s['median_epoch_gap'][j]:+6.0f}" for j in OBJECTIVES)
              + "  | " + "  ".join(f"{s['median_normalized_regret'][j]:+6.3f}"
                                   for j in OBJECTIVES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(analysis_class=ANALYSIS_CLASS, reason=REASON,
                       counts_toward_insufficiency_gate=False,
                       reference_frame="G1", smooth_window=smooth_w,
                       checkpoint_grid="every 5 epochs (24 per run)",
                       n_runs=len(per_run), summary=summary, runs=per_run), fh, indent=2)
    print(f"\n[full-D] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
