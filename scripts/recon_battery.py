"""A1-A5 analogs on the recon forward pass.  [EXPLORATORY-UNVERIFIED-PROVENANCE]

Nothing this produces may enter registered artifacts, paper claims, or be pooled with any
registered result.  Outputs live under ``results/exploratory_c1m/`` only.

The arithmetic is the registered corrected frame, reached through the same modules the
adjudicated battery used: ``axis_frame`` for the per-axis oracle, denominator and normalized
regret; ``ReconRunFrame`` for the joint quantities; ``classify`` for the taxonomy;
``effective_rank`` for the ER selector; ``per_class_error`` and ``static_tail_classes`` for
the worst-class axis.  Nothing is retyped, so the recon cannot quietly diverge in a
definition.

What the recon does *not* have, and therefore does not report:

* **No OOD axis.**  There is no OOD pool, so ``max_a`` runs over ID and WC only.  A1's OOD
  row, A9's two-world certificate and every OOD-conditioned quantity are absent rather than
  filled with a placeholder.
* **No frozen-historical comparison.**  Registered A1 sets the corrected oracle against the
  frozen 120-epoch moving-average oracle.  The recon has no 120-epoch series and no frozen
  definition of its own, so there is nothing to compare against and the A1 row carries the
  corrected side only.

Two pins that the registered battery makes by rule and that the recon inherits by the same
rule rather than by a fresh choice:

* **The worst-class tail is static, taken from the ``ce_seed0`` run.**  Registered
  ``ref_id`` is the ``ce_seed0`` run of the condition; here that is ``c1m_ce_seed0``.  The
  tail is the bottom 30% of classes by final-epoch per-class accuracy on the clean test set,
  computed once and held for all four runs.  At 14 classes, ``k = round(0.30 * 14) = 4``.
* **The selectors are the three registered ones** -- E(tau=1), NA, ER-argmax -- evaluated on
  the NA split rather than on a full training forward, which is the deviation
  ``recon_forward.py`` records.  LW-N is not computed: it is reported and never gating in
  the registered frame, and the recon has no use for a non-gating quantity.

The delta grid is the registered (0.05, 0.10, 0.20) with 0.10 primary, so the taxonomy is
read at the same threshold as the registered work even though the two may never be pooled.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.corrected import (DELTA_PRIMARY, DELTAS, INDETERMINATE_BAND,   # noqa: E402
                                REGISTERED_SELECTORS, axis_frame, classify,
                                effective_rank, iqr)
from eval.tail import per_class_error, static_tail_classes                   # noqa: E402
from e_tgrid_report import statistics_at                                     # noqa: E402
from recon_frame import RECON_AXES, RECON_GRID, ReconRunFrame                # noqa: E402
from provenance import code_stamp                                            # noqa: E402

FWD = os.path.join(ROOT, "results", "exploratory_c1m", "forward")
OUT = os.path.join(ROOT, "results", "exploratory_c1m", "battery_recon.json")
REFERENCE_RUN = "c1m_ce_seed0"          # the registered ref_id rule: ce_seed0 of the condition
TAIL_FRAC = 0.30
N_CLASSES = 14
TAG = "EXPLORATORY-UNVERIFIED-PROVENANCE"


class ReconRun:
    """Everything one run contributes, read once from the forward output."""

    def __init__(self, run_id: str) -> None:
        d = os.path.join(FWD, run_id)
        lab = np.load(os.path.join(d, "labels.npz"))
        self.run_id = run_id
        self.y_test = np.asarray(lab["y_test"], np.int64)
        self.y_split = np.asarray(lab["y_split_noisy"], np.int64)
        self.epochs = sorted(int(e[2:]) for e in os.listdir(d) if e.startswith("ep"))
        if list(self.epochs) != list(RECON_GRID):
            raise SystemExit(f"{run_id}: epochs {self.epochs} != recon grid {list(RECON_GRID)}")

        n = len(self.epochs)
        self.correct = np.zeros((n, len(self.y_test)), bool)
        self.na_acc, self.e_stat_T1, self.er = np.zeros(n), np.zeros(n), np.zeros(n)
        self.pce = np.zeros((n, N_CLASSES))
        for i, ep in enumerate(self.epochs):
            e = os.path.join(d, f"ep{ep:03d}")
            lt = np.load(os.path.join(e, "logits_test.npy"))
            pred = lt.argmax(1)
            self.correct[i] = pred == self.y_test
            corr, tot = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
            np.add.at(tot, self.y_test, 1.0)
            np.add.at(corr, self.y_test[self.correct[i]], 1.0)
            self.pce[i] = per_class_error(corr, tot)

            ls = np.load(os.path.join(e, "logits_split.npy"))
            self.na_acc[i] = float((ls.argmax(1) == self.y_split).mean())
            self.e_stat_T1[i] = statistics_at(ls, 1.0)["mean_max_softmax"]
            self.er[i] = effective_rank(np.load(os.path.join(e, "feats_split.npy")))

    def r_id(self) -> np.ndarray:
        return 1.0 - self.correct.mean(axis=1)

    def r_wc(self, tail: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.nanmean(self.pce[i][tail]))
                           for i in range(len(self.epochs))])

    def selectors(self) -> Dict[str, int]:
        return {"E_tau1": int(np.argmax(self.e_stat_T1)),
                "NA": int(np.argmax(self.na_acc)),
                "ER_argmax": int(np.argmax(self.er))}


def static_tail(ref: ReconRun) -> np.ndarray:
    """Bottom 30% of classes by FINAL-epoch per-class accuracy on the reference run."""
    final_acc = 1.0 - ref.pce[-1]
    return static_tail_classes(final_acc, TAIL_FRAC)


def a1(run: ReconRun, fr: ReconRunFrame) -> dict:
    out = {}
    for a in RECON_AXES:
        f = fr.axes[a]
        out[a] = dict(t_star_grid_index=f.t_star, t_star_epoch=int(RECON_GRID[f.t_star]),
                      R_star=f.R_star, d=f.d, excluded_fail_closed=f.excluded,
                      iqr_20=iqr(f.risk), delta_raw=f.delta_raw.tolist(),
                      ghat=None if f.excluded else f.ghat.tolist())
    return dict(layer="LE", axes=out,
                frozen_historical="absent: the recon has no 120-epoch series and no frozen "
                                  "definition, so there is nothing to compare against",
                omitted_axes=["OOD"])


def a2_a3(fr: ReconRunFrame, sel_J: Dict[str, float]) -> dict:
    rho = fr.rho_star_le()
    per_delta = {f"delta_{d:g}": dict(
        F_delta=[int(RECON_GRID[i]) for i in fr.feasible_set(d)],
        w_delta=fr.w_delta(d),
        taxonomy=classify(rho, sel_J, d)) for d in DELTAS}
    return dict(layer="LE", rho_star_LE=rho, per_delta=per_delta,
                indeterminate_band=INDETERMINATE_BAND,
                gated_by=list(REGISTERED_SELECTORS),
                w_delta_denominator=len(RECON_GRID))


def a4(fr: ReconRunFrame, sel: Dict[str, int]) -> dict:
    out = {s: dict(grid_index=gi, epoch=int(RECON_GRID[gi]),
                   J_LE=fr.J(gi), eta_LE=fr.eta(gi),
                   per_axis_ghat={a: (None if fr.axes[a].excluded
                                      else float(fr.axes[a].ghat[gi])) for a in RECON_AXES})
           for s, gi in sel.items()}
    return dict(layer="LE", rho_star_LE=fr.rho_star_le(), selectors=out,
                evaluated_on="NA split (5000 rows), not a full training forward",
                nonnegativity="eta is J - rho within the LE layer only (spec D.1)")


def a5(fr: ReconRunFrame) -> dict:
    return dict(layer="LE", exact=True, no_binomial=True,
                per_delta={f"delta_{d:g}": fr.uniform_baseline(d) for d in DELTAS})


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    a = p.parse_args(argv)

    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to emit a battery from a dirty tree")

    runs = {r: ReconRun(r) for r in a.runs}
    if REFERENCE_RUN not in runs:
        raise SystemExit(f"the worst-class tail is defined from {REFERENCE_RUN}, which is "
                         f"not among the runs given")
    tail = static_tail(runs[REFERENCE_RUN])
    print(f"[battery] static tail from {REFERENCE_RUN}: classes {tail.tolist()} "
          f"(k={len(tail)} of {N_CLASSES})  [{TAG}]")

    per_run: List[dict] = []
    for rid, run in runs.items():
        axes = {"ID": axis_frame("ID", run.r_id()), "WC": axis_frame("WC", run.r_wc(tail))}
        excluded = [a_ for a_, f in axes.items() if f.excluded]
        fr = ReconRunFrame(run_id=rid, axes=axes, excluded_axes=excluded)
        sel = run.selectors()
        sel_J = {s: fr.J(i) for s, i in sel.items()}
        per_run.append(dict(run_id=rid, classification=TAG,
                            A1=a1(run, fr), A2=a2_a3(fr, sel_J), A3=a2_a3(fr, sel_J),
                            A4=a4(fr, sel), A5=a5(fr)))
        t = per_run[-1]["A2"]["per_delta"][f"delta_{DELTA_PRIMARY:g}"]
        print(f"  {rid}: rho*_LE={fr.rho_star_le():.4f} "
              f"|F_{DELTA_PRIMARY:g}|={len(t['F_delta'])} w={t['w_delta']:.3f} "
              f"{t['taxonomy']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(dict(classification=TAG, code_stamp=stamp, per_run=per_run,
                       axes=list(RECON_AXES), omitted_axes=["OOD"],
                       grid=list(RECON_GRID), deltas=list(DELTAS),
                       delta_primary=DELTA_PRIMARY,
                       worst_class_tail=dict(reference_run=REFERENCE_RUN,
                                             frac=TAIL_FRAC, classes=tail.tolist()),
                       selectors=list(REGISTERED_SELECTORS)), fh, indent=1)
    os.replace(tmp, OUT)
    print(f"[battery] wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
