"""C1 — noisy-validation baseline, run exactly to the pins.

Confirmatory. Every choice below is quoted from the pin block committed in
docs/g2_pin_requests.md ("C1 and (b) pins — pre-computation"); nothing is improvised.

  split      N = 5,000 from the noisy training set, stratified by NOISY label, equal
             per-class counts, remainder (if any) distributed by ascending class index
  sampling   numpy.random.default_rng(20260813); one permutation per class of that
             class's sample indices, in the order they appear in the stored labels
             array; take the first N / num_classes
  statistic  accuracy against the NOISY labels on the split; selection = argmax over
             the 24-epoch grid; tie-break = earliest epoch
  secondary  mean CE against the noisy labels on the same split; selection = argmin;
             non-gating
  smoothing  none

Inputs are the stored forward-pass arrays only: no checkpoint is loaded, no GPU is used,
and the sealed masks are not touched -- the noisy labels come from the labels.npz written
alongside the logits.
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

SPLIT_N = 5000
SPLIT_SEED = 20260813

LIMITATION = ("split is drawn from data used in training (no true held-out noisy split "
              "exists without retraining; out of Phase-0 scope). Memorization inflates "
              "noisy accuracy at late epochs; this bias is a property of the baseline "
              "under audit and is reported, not corrected.")


def draw_split(noisy: np.ndarray, n_classes: int, n_total: int = SPLIT_N,
               seed: int = SPLIT_SEED) -> np.ndarray:
    """The pinned stratified draw. Deterministic from this function alone."""
    per_class = n_total // n_classes
    remainder = n_total - per_class * n_classes          # 0 for both datasets; kept
    rng = np.random.default_rng(seed)                    # so the code matches the pin
    picked: List[np.ndarray] = []
    for c in range(n_classes):                           # ascending class index
        idx_c = np.flatnonzero(noisy == c)               # order as stored
        take = per_class + (1 if c < remainder else 0)
        picked.append(rng.permutation(idx_c)[:take])
    split = np.concatenate(picked)
    assert len(split) == n_total, f"split is {len(split)}, expected {n_total}"
    assert len(np.unique(split)) == n_total, "split contains duplicate indices"
    return np.sort(split)


def _ce(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(labels)), labels]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--forward", default=os.path.join(ROOT, "results", "forward_b2"))
    p.add_argument("--runs", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "C1_noisyval.json"))
    a = p.parse_args(argv)

    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):                      # protocol rule R2
        raise SystemExit(
            "R2 violation: the working tree is dirty, so this registered output would "
            f"carry git_tree_dirty=true. Dirty paths: {stamp.get('git_dirty_paths')}. "
            "Commit before executing.")

    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])         # used only by the regret scorer

    run_dirs = sorted(d for d in glob.glob(os.path.join(a.forward, "*"))
                      if os.path.isdir(d) and os.path.isfile(os.path.join(d, "labels.npz")))
    per_run: List[dict] = []

    for fdir in run_dirs:
        run_id = os.path.basename(fdir)
        run = load_run(os.path.join(a.runs, run_id))     # G1 frame (certified)
        m = run.meta
        n_classes = int(m["n_classes"])
        ref = os.path.join(a.runs, f"{m['dataset']}_{m['noise_type']}{m['eta']:g}_ce_seed0")
        tail = static_tail_from_reference(load_run(ref))

        noisy = np.load(os.path.join(fdir, "labels.npz"))["train_noisy"]
        split = draw_split(noisy, n_classes)
        y = noisy[split]

        epochs, acc, loss = [], [], []
        for epdir in sorted(glob.glob(os.path.join(fdir, "ep*"))):
            logits = np.load(os.path.join(epdir, "logits_train.npy"))[split]
            epochs.append(int(os.path.basename(epdir)[2:]))
            acc.append(float((logits.argmax(1) == y).mean()))
            loss.append(float(_ce(logits, y).mean()))

        eps = np.asarray(epochs)
        # argmax / argmin with the pinned earliest-epoch tie-break: numpy's argmax and
        # argmin already return the FIRST extremum, and the grid is ascending in epoch.
        sel_acc = int(eps[int(np.argmax(acc))])
        sel_loss = int(eps[int(np.argmin(loss))])

        per_run.append(dict(
            run_id=run_id, cell=run.cell, learner=m["learner"], seed=m["seed"],
            n_classes=n_classes, checkpoint_epochs=eps.tolist(),
            split_size=int(len(split)),
            split_class_counts=np.bincount(y, minlength=n_classes).tolist(),
            split_index_sha256=__import__("hashlib").sha256(
                np.ascontiguousarray(split).tobytes()).hexdigest(),
            noisy_accuracy_curve=acc, noisy_ce_curve=loss,
            selected_epoch=sel_acc,
            secondary_selected_epoch_by_loss=sel_loss,
            regrets=regret_at_epoch(run, tail, sel_acc, smooth_w),
            secondary_regrets=regret_at_epoch(run, tail, sel_loss, smooth_w)))
        print(f"[C1] {run_id:34s} acc-sel ep{sel_acc:>3d}  loss-sel ep{sel_loss:>3d}  "
              f"acc {acc[0]:.3f}->{acc[-1]:.3f}", flush=True)

    summary = dict(
        median_selected_epoch=float(np.median([r["selected_epoch"] for r in per_run])),
        median_secondary_selected_epoch=float(
            np.median([r["secondary_selected_epoch_by_loss"] for r in per_run])),
        median_normalized_regret={
            j: float(np.median([r["regrets"][j]["normalized_regret"] for r in per_run]))
            for j in OBJECTIVES},
        median_secondary_normalized_regret={
            j: float(np.median([r["secondary_regrets"][j]["normalized_regret"]
                                for r in per_run])) for j in OBJECTIVES})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(
            selector="C1 (noisy-validation, train-subset reuse)",
            analysis_class="confirmatory", reference_frame="G1",
            pinned_in="docs/g2_pin_requests.md, commit 'C1 and (b) pins — pre-computation'",
            split=dict(n=SPLIT_N, seed=SPLIT_SEED, stratified_by="noisy label",
                       rule="one rng.permutation per class in ascending class index, "
                            "first N/num_classes; remainder by ascending class index"),
            statistic="accuracy vs noisy labels, argmax, earliest-epoch tie-break",
            secondary="mean CE vs noisy labels, argmin, non-gating",
            smoothing="none",
            limitation=LIMITATION,
            code_stamp=stamp, n_runs=len(per_run), summary=summary, runs=per_run),
            fh, indent=2)
    print(f"\n[C1] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
