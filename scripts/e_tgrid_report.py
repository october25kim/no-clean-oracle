"""Baseline E — single-snapshot confidence / entropy, T-grid stability.

Confirmatory. Executes the procedure registered in the E amendment of
docs/G2_design_memo.md, with no deviation:

  primary   : fixed grid T in {0.5, 1, 2, 5}; the selection epoch is computed per run
              per T, and a run is "T-robust" if the four selections are identical,
              "T-sensitive" otherwise, in which case the full epoch-by-T table is
              reported.
  secondary : per-sample logit normalization, reported alongside; gates nothing.
  variant 3 : label-free per-epoch T_t -- NOT computed, per the amendment.

Inputs are the stored forward-pass logits only. No checkpoint is loaded, no GPU is
used, and no comparison against the frozen E predictions is made here -- adjudication
happens at review.

Degrees of freedom the registered text leaves open are NOT resolved here. Each is
computed both ways where it is binary, and every choice is recorded in the output so
review can pin it without a recomputation:

  * statistic: the memo says "mean max-softmax, OR mean predictive entropy". Both are
    computed and each gets its own T-robust/T-sensitive label.
  * direction: taken as argmax of confidence / argmin of entropy. This is the reading
    the frozen prediction presupposes -- it predicts a LATE bias on the grounds that
    memorization raises training-set confidence, which only follows if the rule
    prefers high confidence.
  * smoothing: none. The amendment specifies no moving average, and with 24
    checkpoints spaced 5 epochs apart a 3-point window would span 15 epochs.
  * normalization form for the secondary: per-sample L2 normalization of the logit
    vector before the softmax.
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

from analysis.io import load_run, static_tail_from_reference   # noqa: E402
from analysis.ncr import OBJECTIVES                            # noqa: E402
from analysis.selection import regret_at_epoch                 # noqa: E402
from provenance import code_stamp                              # noqa: E402

T_GRID = [0.5, 1.0, 2.0, 5.0]
UNPINNED = {
    "statistic": "memo says 'mean max-softmax, OR mean predictive entropy'; both "
                 "computed, each with its own T-robust/T-sensitive label",
    "direction": "argmax(confidence) / argmin(entropy); the reading the frozen "
                 "late-bias prediction presupposes",
    "smoothing": "none applied; the amendment specifies none",
    "secondary_normalization": "per-sample L2 normalization of the logit vector",
}


def _softmax(z: np.ndarray, T: float) -> np.ndarray:
    z = z.astype(np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def statistics_at(logits: np.ndarray, T: float) -> Dict[str, float]:
    p = _softmax(logits, T)
    conf = float(p.max(axis=1).mean())
    ent = float((-(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)).mean())
    return dict(mean_max_softmax=conf, mean_entropy=ent)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--forward", default=os.path.join(ROOT, "results", "forward_b2"))
    p.add_argument("--runs", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report", "E_tgrid.json"))
    a = p.parse_args(argv)

    stamp = code_stamp()                    # new process launch: the stamp must appear
    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])   # used only by the regret scorer

    run_dirs = sorted(d for d in glob.glob(os.path.join(a.forward, "*"))
                      if os.path.isdir(d) and os.path.isfile(os.path.join(d, "labels.npz")))
    per_run: List[dict] = []

    for fdir in run_dirs:
        run_id = os.path.basename(fdir)
        run = load_run(os.path.join(a.runs, run_id))          # G1 frame (certified)
        m = run.meta
        ref = os.path.join(a.runs, f"{m['dataset']}_{m['noise_type']}{m['eta']:g}_ce_seed0")
        tail = static_tail_from_reference(load_run(ref))

        epochs, curves = [], {}
        for epdir in sorted(glob.glob(os.path.join(fdir, "ep*"))):
            logits = np.load(os.path.join(epdir, "logits_train.npy"))
            epochs.append(int(os.path.basename(epdir)[2:]))
            for T in T_GRID:
                s = statistics_at(logits, T)
                for name, v in s.items():
                    curves.setdefault(f"T{T:g}", {}).setdefault(name, []).append(v)
            norm = logits / np.maximum(np.linalg.norm(logits, axis=1, keepdims=True), 1e-12)
            for name, v in statistics_at(norm, 1.0).items():
                curves.setdefault("l2norm", {}).setdefault(name, []).append(v)

        eps = np.asarray(epochs)
        sel: Dict[str, Dict[str, int]] = {}
        for key, stats in curves.items():
            sel[key] = dict(
                mean_max_softmax=int(eps[int(np.argmax(stats["mean_max_softmax"]))]),
                mean_entropy=int(eps[int(np.argmin(stats["mean_entropy"]))]))

        grid_keys = [f"T{T:g}" for T in T_GRID]
        labels, chosen = {}, {}
        for stat in ("mean_max_softmax", "mean_entropy"):
            picks = [sel[k][stat] for k in grid_keys]
            labels[stat] = "T-robust" if len(set(picks)) == 1 else "T-sensitive"
            chosen[stat] = dict(zip(grid_keys, picks))

        regrets = {stat: {k: regret_at_epoch(run, tail, sel[k][stat], smooth_w)
                          for k in grid_keys + ["l2norm"]}
                   for stat in ("mean_max_softmax", "mean_entropy")}

        per_run.append(dict(
            run_id=run_id, cell=run.cell, learner=m["learner"], seed=m["seed"],
            checkpoint_epochs=eps.tolist(),
            t_label=labels, selected_epoch_by_T=chosen,
            secondary_l2norm_selected_epoch=sel["l2norm"],
            curves=curves, regrets=regrets))
        print(f"[E] {run_id:34s} conf {labels['mean_max_softmax']:<12s} "
              f"{list(chosen['mean_max_softmax'].values())}   "
              f"ent {labels['mean_entropy']:<12s} {list(chosen['mean_entropy'].values())}",
              flush=True)

    summary = {}
    for stat in ("mean_max_softmax", "mean_entropy"):
        lab = [r["t_label"][stat] for r in per_run]
        summary[stat] = dict(
            n_T_robust=sum(1 for x in lab if x == "T-robust"),
            n_T_sensitive=sum(1 for x in lab if x == "T-sensitive"),
            median_selected_epoch_by_T={
                k: float(np.median([r["selected_epoch_by_T"][stat][k] for r in per_run]))
                for k in [f"T{T:g}" for T in T_GRID]},
            median_normalized_regret_by_T={
                k: {j: float(np.median([r["regrets"][stat][k][j]["normalized_regret"]
                                        for r in per_run])) for j in OBJECTIVES}
                for k in [f"T{T:g}" for T in T_GRID] + ["l2norm"]})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(
            selector="E (single-snapshot confidence / entropy)",
            analysis_class="confirmatory", reference_frame="G1",
            registered_in="docs/G2_design_memo.md, E amendment 2026-08-13",
            t_grid=T_GRID, variant_3_computed=False,
            unpinned_degrees_of_freedom=UNPINNED,
            no_prediction_adjudication_in_this_run=True,
            code_stamp=stamp, n_runs=len(per_run), summary=summary, runs=per_run), fh,
            indent=2)
    print(f"\n[E] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
