"""Run the Label Wave baseline over the G1 logs and score it against the oracles.

Log-only and CPU-only: it reads ``pred_flip_count``, which is exactly the paper's PC
metric, so no checkpoint and no re-training is involved. What it answers is: if Label
Wave had been the stopping rule, which epoch would it have picked, and what would that
have cost against each objective's oracle epoch?
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

from analysis.label_wave import (PAPER, PAPER_BEST_K, label_wave, naive_argmin,  # noqa: E402
                                 prediction_changes)
from analysis.ncr import OBJECTIVES                                              # noqa: E402
from analysis.selection import regret_at_epoch                                    # noqa: E402
from make_report import build_cells, discover_runs                               # noqa: E402

# Owner decision 2026-08-12. Emitted with every number this script produces so the
# figures can never travel without the annotation that qualifies them.
STATUS_OF_RECORD = (
    "NOT DIRECTLY APPLICABLE under the canonical cosine schedule: premise violation. "
    "The paper's Appendix B fixes LR = 0.01, G1 anneals cosine to 0 over 120 epochs, "
    "and an annealed LR freezes predictions independently of memorization, which "
    "removes the prediction-changes rise the rule keys on. The paper's Appendix C.3 "
    "predicts exactly this failure mode (no learning-confusion stage -> no stopping "
    "point), and names robust regularization as one cause, which also covers the ELR "
    "cells at any learning rate. These numbers are APPENDIX material, annotated as "
    "premise-violated, and do NOT count toward the G2 baseline-insufficiency gate. "
    "G1's cosine schedule was sealed before Label Wave was evaluated."
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--results", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "label_wave_baseline.json"))
    p.add_argument("--k", type=int, default=PAPER_BEST_K)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--k-grid", default="1,2,3,5,10")
    p.add_argument("--patience-grid", default="3,5,10,20")
    a = p.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])
    cells = build_cells(discover_runs(a.results), smooth_w)

    per_run: List[dict] = []
    for cell, c in cells.items():
        tail_classes = np.asarray(c["tail_classes"])
        for e in c["entries"]:
            run = e["run"]
            res = label_wave(run.epochs, k=a.k, patience=a.patience)
            pc = prediction_changes(run.epochs)
            stop = res.stop_epoch if res.stop_epoch is not None else len(run.epochs) - 1
            per_run.append(dict(
                cell=cell, run_id=run.run_id, learner=run.meta["learner"],
                seed=run.meta["seed"], n_epochs=len(run.epochs),
                label_wave=dict(stop_epoch=res.stop_epoch, halted_epoch=res.halted_epoch,
                                exhausted=res.exhausted),
                degenerate_naive_argmin_epoch=naive_argmin(run.epochs),
                pc_first=float(pc[0]), pc_last=float(pc[-1]), pc_min=float(pc.min()),
                scored_at_epoch=int(stop),
                objectives=regret_at_epoch(run, tail_classes, int(stop), smooth_w)))

    # grids over the two parameters the paper leaves unspecified
    grid = []
    for k in [int(x) for x in a.k_grid.split(",") if x]:
        for pat in [int(x) for x in a.patience_grid.split(",") if x]:
            stops, exhausted = [], 0
            for c in cells.values():
                for e in c["entries"]:
                    r = label_wave(e["run"].epochs, k=k, patience=pat)
                    stops.append(r.stop_epoch if r.stop_epoch is not None else np.nan)
                    exhausted += int(r.exhausted)
            s = np.asarray(stops, dtype=np.float64)
            grid.append(dict(k=k, patience=pat, median_stop_epoch=float(np.nanmedian(s)),
                             min_stop_epoch=float(np.nanmin(s)), max_stop_epoch=float(np.nanmax(s)),
                             runs_never_halted=exhausted, n_runs=len(stops)))

    n = len(per_run)
    exhausted = sum(r["label_wave"]["exhausted"] for r in per_run)
    med = {j: float(np.median([r["objectives"][j]["normalized_regret"] for r in per_run]))
           for j in OBJECTIVES}
    med_gap = {j: float(np.median([r["objectives"][j]["epoch_gap"] for r in per_run]))
               for j in OBJECTIVES}

    print(f"Label Wave ({PAPER})")
    print(f"  STATUS: {STATUS_OF_RECORD}\n")
    print(f"  k={a.k} (paper's Appendix-E best), patience={a.patience} "
          f"(NOT specified in the paper)")
    print(f"  runs: {n}   never reached the patience trip: {exhausted}/{n}")
    print(f"  median selected epoch: "
          f"{np.median([r['scored_at_epoch'] for r in per_run]):.0f} of {per_run[0]['n_epochs']-1}")
    print("  median normalized regret vs each oracle:  "
          + "  ".join(f"{j}={med[j]:+.3f}" for j in OBJECTIVES))
    print("  median epoch gap vs each oracle:          "
          + "  ".join(f"{j}={med_gap[j]:+.0f}" for j in OBJECTIVES))
    print("\n  k / patience grid (median selected epoch, runs never halted):")
    for g in grid:
        print(f"    k={g['k']:>2} p={g['patience']:>2}  median stop {g['median_stop_epoch']:>5.0f}  "
              f"never-halted {g['runs_never_halted']:>2}/{g['n_runs']}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(paper=PAPER, rule="Eq.2 PC, Eq.3 trailing mean over k, Eq.4 first "
                                         "local minimum via Algorithm 1 patience",
                       status_of_record=STATUS_OF_RECORD,
                       counts_toward_insufficiency_gate=False,
                       needs_verification=[
                           "patience p has no numeric value anywhere in the paper",
                           "k default not stated; Appendix E reports k=3 as strongest",
                           "behaviour of PC'_t before the k-window fills is unspecified"],
                       k=a.k, patience=a.patience, n_runs=n, runs_never_halted=exhausted,
                       median_normalized_regret=med, median_epoch_gap=med_gap,
                       parameter_grid=grid, runs=per_run), fh, indent=2)
    print(f"\n[label_wave] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
