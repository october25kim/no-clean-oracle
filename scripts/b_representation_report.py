"""(b) — representation baseline, method B-i (effective rank), run exactly to the pins.

Confirmatory. Every choice is quoted from the pin block committed in
docs/g2_pin_requests.md ("C1 and (b) pins — pre-computation").

  input      feats_train, all samples, no subsampling
  statistic  mean-center the features; form the 512x512 covariance; effective rank =
             exp(Shannon entropy of the eigenvalue distribution normalized to sum 1)
             (Roy & Vetterli)
  binding    argmax of effective rank over the 24-epoch grid; tie-break earliest epoch;
             no smoothing
  sensitivity argmin, reported alongside, gates nothing
  curves     the full 24-point effective-rank curve is saved per run

The covariance's normalization (n vs n-1) is not pinned and does not need to be:
effective rank normalizes the eigenvalues to sum 1, so any positive rescaling of the
covariance leaves the statistic unchanged. n-1 is used. For a PSD covariance the
eigenvalues equal the singular values, so this matches Roy & Vetterli's singular-value
formulation exactly.

Inputs are the stored forward-pass arrays only: no checkpoint is loaded, no GPU is used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from analysis.io import load_run, static_tail_from_reference   # noqa: E402
from analysis.ncr import OBJECTIVES                            # noqa: E402
from analysis.selection import regret_at_epoch                 # noqa: E402
from provenance import code_stamp                              # noqa: E402

DIRECTION_RATIONALE = (
    "argmax is registered on the hypothesis that representation quality peaks before "
    "memorization-driven collapse; the late-epoch rank dynamic under label noise is not "
    "settled in the literature — needs verification. argmin is reported as a "
    "non-binding sensitivity.")


def effective_rank(feats: np.ndarray) -> float:
    """exp(H) of the covariance's eigenvalue distribution normalized to sum 1."""
    x = feats.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                 # mean-center
    cov = (x.T @ x) / (len(x) - 1)                        # 512 x 512
    w = np.linalg.eigvalsh(cov)                           # ascending, real (PSD)
    w = np.clip(w, 0.0, None)                             # kill float-noise negatives
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 0]                                          # 0 log 0 := 0
    return float(np.exp(-(p * np.log(p)).sum()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--forward", default=os.path.join(ROOT, "results", "forward_b2"))
    p.add_argument("--runs", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "B_representation.json"))
    a = p.parse_args(argv)

    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):                       # protocol rule R2
        raise SystemExit(
            "R2 violation: the working tree is dirty, so this registered output would "
            f"carry git_tree_dirty=true. Dirty paths: {stamp.get('git_dirty_paths')}. "
            "Commit before executing.")

    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])          # regret scorer only

    run_dirs = sorted(d for d in glob.glob(os.path.join(a.forward, "*"))
                      if os.path.isdir(d) and os.path.isfile(os.path.join(d, "labels.npz")))
    per_run: List[dict] = []

    for fdir in run_dirs:
        run_id = os.path.basename(fdir)
        run = load_run(os.path.join(a.runs, run_id))      # G1 frame (certified)
        m = run.meta
        ref = os.path.join(a.runs, f"{m['dataset']}_{m['noise_type']}{m['eta']:g}_ce_seed0")
        tail = static_tail_from_reference(load_run(ref))

        epochs, erank = [], []
        for epdir in sorted(glob.glob(os.path.join(fdir, "ep*"))):
            feats = np.load(os.path.join(epdir, "feats_train.npy"))
            epochs.append(int(os.path.basename(epdir)[2:]))
            erank.append(effective_rank(feats))

        eps = np.asarray(epochs)
        # numpy argmax/argmin return the FIRST extremum and the grid ascends in epoch,
        # which is exactly the pinned earliest-epoch tie-break.
        sel_max = int(eps[int(np.argmax(erank))])
        sel_min = int(eps[int(np.argmin(erank))])

        per_run.append(dict(
            run_id=run_id, cell=run.cell, learner=m["learner"], seed=m["seed"],
            n_features=int(np.load(os.path.join(sorted(glob.glob(os.path.join(fdir, "ep*")))[0],
                                                "feats_train.npy")).shape[1]),
            checkpoint_epochs=eps.tolist(),
            effective_rank_curve=erank,
            selected_epoch=sel_max,
            sensitivity_argmin_epoch=sel_min,
            regrets=regret_at_epoch(run, tail, sel_max, smooth_w),
            sensitivity_regrets=regret_at_epoch(run, tail, sel_min, smooth_w)))
        print(f"[b] {run_id:34s} argmax ep{sel_max:>3d}  argmin ep{sel_min:>3d}  "
              f"erank {erank[0]:.1f}->{max(erank):.1f}->{erank[-1]:.1f}", flush=True)

    summary = dict(
        median_selected_epoch=float(np.median([r["selected_epoch"] for r in per_run])),
        median_sensitivity_argmin_epoch=float(
            np.median([r["sensitivity_argmin_epoch"] for r in per_run])),
        median_normalized_regret={
            j: float(np.median([r["regrets"][j]["normalized_regret"] for r in per_run]))
            for j in OBJECTIVES},
        median_sensitivity_normalized_regret={
            j: float(np.median([r["sensitivity_regrets"][j]["normalized_regret"]
                                for r in per_run])) for j in OBJECTIVES})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(
            selector="(b) representation-based, method B-i (effective rank)",
            analysis_class="confirmatory", reference_frame="G1",
            pinned_in="docs/g2_pin_requests.md, commit 'C1 and (b) pins — pre-computation'",
            statistic="exp(Shannon entropy of the covariance eigenvalue distribution "
                      "normalized to sum 1) (Roy & Vetterli); features mean-centered; "
                      "all samples, no subsampling",
            selection="argmax over the 24-epoch grid, earliest-epoch tie-break",
            smoothing="none",
            direction_rationale=DIRECTION_RATIONALE,
            sensitivity="argmin selection, non-gating",
            b_iv_status="TABLED, not rejected; faithful TopoGeoScore reproduction "
                        "requires transcription from the original paper under a "
                        "separate instruction",
            code_stamp=stamp, n_runs=len(per_run), summary=summary, runs=per_run),
            fh, indent=2)
    print(f"\n[b] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
