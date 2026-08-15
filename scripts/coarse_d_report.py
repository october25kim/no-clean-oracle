"""Coarse baseline D: loss-distribution separation, from the logged quantiles only.

EXPLORATORY AND NON-DECISIONAL. Design-review input, not a gate result.

The small-loss principle: under label noise the per-sample training loss splits into a
low mode (clean labels) and a high mode (mislabeled ones). The separation widens while
the model is learning the clean structure and collapses as it memorizes the corrupted
labels, so the epoch of PEAK separation is a natural stopping point. It is read off one
epoch at a time, so it carries none of the inter-epoch churn that makes Label Wave
schedule-coupled.

What makes this version coarse: G1 logs only five quantiles of the per-sample loss
(p10/p25/p50/p75/p90), not the per-sample values, so the two-component mixture fit the
literature versions use is impossible here. These spread statistics are proxies for
that separation, and their job is to indicate whether the full version is worth the
forward passes -- not to stand in for it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.ncr import OBJECTIVES, moving_average                    # noqa: E402
from analysis.selection import rank_agreement, regret_at_epoch         # noqa: E402
from make_report import build_cells, discover_runs                     # noqa: E402

STATUS = ("EXPLORATORY / NON-DECISIONAL. Coarse proxy for the loss-distribution "
          "separation baseline, computed from five logged quantiles because G1 does not "
          "log per-sample losses. Design-review input only; it does not count toward the "
          "G2 baseline-insufficiency gate, and the full version (per-sample losses, "
          "two-component fit) is what would.")

# Higher = the two loss modes are further apart = less memorized.
STATISTICS = {
    "p90_minus_p50": lambda q: q["p90"] - q["p50"],
    "p90_minus_p10": lambda q: q["p90"] - q["p10"],
    "p75_minus_p25": lambda q: q["p75"] - q["p25"],
    "upper_over_lower": lambda q: (q["p90"] - q["p50"]) / np.maximum(q["p50"] - q["p10"], 1e-12),
}


def quantile_curves(run) -> Dict[str, np.ndarray]:
    keys = ["p10", "p25", "p50", "p75", "p90"]
    return {k: np.asarray([e["train_loss_quantiles"][k] for e in run.epochs],
                          dtype=np.float64) for k in keys}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--results", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "coarse_d_exploratory.json"))
    a = p.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])
    cells = build_cells(discover_runs(a.results), smooth_w)

    per_run: List[dict] = []
    for cell, c in cells.items():
        tail = np.asarray(c["tail_classes"])
        for e in c["entries"]:
            run = e["run"]
            q = quantile_curves(run)
            rec = dict(cell=cell, run_id=run.run_id, learner=run.meta["learner"],
                       seed=run.meta["seed"], statistics={})
            for name, fn in STATISTICS.items():
                s = fn(q)
                sm = moving_average(s, smooth_w)
                stop = int(np.argmax(sm))          # peak separation
                rec["statistics"][name] = dict(
                    stop_epoch=stop,
                    peak_value=float(s[stop]), first=float(s[0]), last=float(s[-1]),
                    objectives=regret_at_epoch(run, tail, stop, smooth_w),
                    rank_agreement=rank_agreement(s, run, tail, higher_is_better=True))
            per_run.append(rec)

    summary = {}
    for name in STATISTICS:
        stops = np.array([r["statistics"][name]["stop_epoch"] for r in per_run])
        summary[name] = dict(
            median_stop_epoch=float(np.median(stops)),
            min_stop_epoch=int(stops.min()), max_stop_epoch=int(stops.max()),
            median_normalized_regret={
                j: float(np.median([r["statistics"][name]["objectives"][j]["normalized_regret"]
                                    for r in per_run])) for j in OBJECTIVES},
            median_epoch_gap={
                j: float(np.median([r["statistics"][name]["objectives"][j]["epoch_gap"]
                                    for r in per_run])) for j in OBJECTIVES},
            median_rank_agreement={
                j: float(np.median([r["statistics"][name]["rank_agreement"][j]
                                    for r in per_run])) for j in OBJECTIVES})

    oracle_med = {j: float(np.median([r["statistics"]["p90_minus_p50"]["objectives"][j]
                                      ["oracle_epoch"] for r in per_run]))
                  for j in OBJECTIVES}

    print("Coarse baseline D — loss-distribution separation from logged quantiles")
    print(f"  STATUS: {STATUS}\n")
    print(f"  runs: {len(per_run)}   oracle epoch medians: "
          + "  ".join(f"{j}={oracle_med[j]:.0f}" for j in OBJECTIVES))
    print()
    print(f"  {'statistic':18s} {'median':>7s} | {'median epoch gap':^22s} | "
          f"{'median norm. regret':^24s} | {'median rank agreement':^24s}")
    print(f"  {'':18s} {'stop':>7s} | " + "  ".join(f"{j:>6s}" for j in OBJECTIVES)
          + "  | " + "  ".join(f"{j:>6s}" for j in OBJECTIVES)
          + "  | " + "  ".join(f"{j:>6s}" for j in OBJECTIVES))
    for name, s in summary.items():
        print(f"  {name:18s} {s['median_stop_epoch']:7.0f} | "
              + "  ".join(f"{s['median_epoch_gap'][j]:+6.0f}" for j in OBJECTIVES)
              + "  | " + "  ".join(f"{s['median_normalized_regret'][j]:+6.3f}" for j in OBJECTIVES)
              + "  | " + "  ".join(f"{s['median_rank_agreement'][j]:+6.3f}" for j in OBJECTIVES))
    print("\n  rank agreement = -Kendall tau(statistic, risk): positive means the "
          "statistic orders epochs\n  the way that objective's risk does; ~0 means it "
          "carries no ordering information.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(status=STATUS, non_decisional=True,
                       counts_toward_insufficiency_gate=False,
                       smooth_window=smooth_w, n_runs=len(per_run),
                       oracle_epoch_medians=oracle_med,
                       summary=summary, runs=per_run), fh, indent=2)
    print(f"\n[coarse_d] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
