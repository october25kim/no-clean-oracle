"""Binding check for the B2 rerun: are its trajectories the SAME trajectories as G1's?

B2 reruns five G1 cells only to retain the per-epoch checkpoints G1 discarded. Those
checkpoints are only usable as samples of the G1 verdict's trajectories if the rerun
reproduced them. This diffs each B2 run's metrics.jsonl against the G1 original
epoch by epoch and reports the maximum absolute deviation per metric.

Outcome, per the owner's condition:
  - every deviation exactly 0  -> B2 checkpoints are certified samples of the G1
    trajectories, and G2 may use G1 oracle epochs with B2 checkpoints.
  - ANY nonzero deviation      -> B2 becomes its own reference frame. G2 oracle epochs
    and regrets for these cells must then be recomputed from B2's own logs, and G1
    oracle epochs must NOT be mixed with B2 checkpoints.

The verdict of record is untouched either way; this only decides which logs G2 reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from analysis.io import load_run   # noqa: E402

# Scalar per-epoch metrics that the analysis depends on. Wall-clock fields are excluded
# on purpose: they measure scheduling under a varying co-tenant, never results.
SCALARS = ["R_ID", "R_tail_dynamic", "R_OOD_primary_energy_semantic", "train_loss_noisy", "lr"]
VECTORS = ["per_class_test_error", "per_class_train_acc"]
NESTED = {"R_OOD": None, "train_loss_quantiles": None}
INTEGERS = ["pred_flip_count"]
IGNORED = ["epoch_time_s", "train_time_s", "eval_time_s"]


def _dev(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    d = np.abs(a - b)
    d = d[~np.isnan(d)]
    return float(d.max()) if d.size else 0.0


def compare_run(g1_dir: str, b2_dir: str) -> dict:
    g1, b2 = load_run(g1_dir), load_run(b2_dir)
    out: Dict[str, object] = dict(run_id=g1.run_id, g1_epochs=len(g1.epochs),
                                  b2_epochs=len(b2.epochs))
    if len(g1.epochs) != len(b2.epochs):
        out["error"] = "epoch count differs"
        return out

    devs: Dict[str, float] = {}
    for k in SCALARS:
        devs[k] = _dev([e[k] for e in g1.epochs], [e[k] for e in b2.epochs])
    for k in VECTORS:
        devs[k] = _dev([e[k] for e in g1.epochs], [e[k] for e in b2.epochs])
    for k in NESTED:
        sub = sorted(g1.epochs[0][k])
        devs[k] = max(_dev([e[k][s] for e in g1.epochs], [e[k][s] for e in b2.epochs])
                      for s in sub)
    for k in INTEGERS:
        # epoch 0 has no predecessor and is null in both
        devs[k] = _dev([e[k] for e in g1.epochs[1:]], [e[k] for e in b2.epochs[1:]])

    out["deviations"] = devs
    out["max_abs_deviation"] = max(devs.values())
    out["identical"] = all(v == 0.0 for v in devs.values())
    # metadata that must match for the comparison to mean anything
    out["seed_match"] = g1.meta["seed"] == b2.meta["seed"]
    out["worker_match"] = (g1.meta.get("train_dataloader_workers")
                           == b2.meta.get("train_dataloader_workers"))
    out["effective_noise_match"] = (g1.meta["effective_overall_noise"]
                                    == b2.meta["effective_overall_noise"])
    out["g1_gpu_uuid"] = g1.meta.get("gpu_uuid")
    out["b2_gpu_uuid"] = b2.meta.get("gpu_uuid")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--g1", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--b2", default=os.path.join(ROOT, "results", "runs_b2"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "b2_trajectory_identity.json"))
    a = p.parse_args(argv)

    run_ids = sorted(d for d in os.listdir(a.b2)
                     if os.path.isfile(os.path.join(a.b2, d, "TERMINAL.json")))
    if not run_ids:
        print(f"no terminal runs under {a.b2}")
        return 1

    results: List[dict] = []
    for run_id in run_ids:
        g1_dir = os.path.join(a.g1, run_id)
        if not os.path.isfile(os.path.join(g1_dir, "metrics.jsonl")):
            results.append(dict(run_id=run_id, error="no G1 original to compare against"))
            continue
        results.append(compare_run(g1_dir, os.path.join(a.b2, run_id)))

    clean = [r for r in results if "error" not in r]
    identical = [r for r in clean if r["identical"]]
    overall = bool(clean) and len(identical) == len(clean) and len(clean) == len(results)

    print(f"{'run':34s} {'max abs dev':>12s}  identical  worst metric")
    for r in results:
        if "error" in r:
            print(f"{r['run_id']:34s} {'-':>12s}  ERROR      {r['error']}")
            continue
        worst = max(r["deviations"], key=lambda k: r["deviations"][k])
        print(f"{r['run_id']:34s} {r['max_abs_deviation']:12.3e}  "
              f"{str(r['identical']):9s}  {worst}")

    frame = ("G1 (B2 checkpoints certified as samples of the G1 trajectories)" if overall
             else "B2's OWN logs (G1 oracle epochs must NOT be mixed with B2 checkpoints)")
    print(f"\nruns compared : {len(clean)}/{len(results)}")
    print(f"bit-identical : {len(identical)}/{len(clean)}")
    print(f"G2 REFERENCE FRAME -> {frame}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(all_identical=overall, reference_frame=("G1" if overall else "B2"),
                       n_runs=len(results), n_identical=len(identical), runs=results),
                  fh, indent=2)
    print(f"[identity] {a.out}")
    return 0 if overall else 2      # 2 = not identical; a decision, not a crash


if __name__ == "__main__":
    raise SystemExit(main())
