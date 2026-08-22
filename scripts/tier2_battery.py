"""Tier-2 A1-A5, on its own table and never pooled with Tier 1.

Amendment §4 is explicit: the 20-point half-epoch grid over a 10-epoch pretrained trajectory
is not comparable to Tier 1's 24 points over 120 epochs, and Tier 2 is an existence check at
scale rather than a statistical pillar. Every output here is therefore keyed to Tier 2 alone,
and the module writes no aggregate that could be read across the two tiers.

The frame is ``recon_frame.ReconRunFrame``, whose axis set and grid are instance state -- the
same subclass built for the two-axis recon, here carrying three axes and twenty points. The
registered ``analysis.corrected.RunFrame`` is again left untouched: it is the code the
adjudicated Tier-1 battery ran, and widening it for a track that is barred from being pooled
with Tier 1 would put the frozen results and these behind a module that has changed.

Registered specifics:

* **R_tail** is the FROZEN k=4 set [3, 4, 10, 12], sealed from noisy-label frequency. It is
  read from the forward pass's ``labels.npz`` rather than recomputed, so the analysis cannot
  quietly re-derive a subset the seal fixed.
* **R_OOD** is the mean over the two pools of ``1 - AUROC(energy)``, matching the registered
  protocol's aggregation, with the per-(score, pool) decomposition reported alongside because
  the amendment names both msp and energy.
* **Selectors** are the amendment's §5 pins: E over the T-grid, C1 on the 5,000 stratified
  noisy split, and effective rank on that split's penultimate features.
* **Selector degeneracy must be disclosed** (tier2_followup.md §4): where two registered
  selectors choose the same grid point, the output says so, because two selectors agreeing
  by being the same number is not corroboration.
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

from analysis.corrected import (DELTAS, DELTA_PRIMARY, INDETERMINATE_BAND,     # noqa: E402
                                REGISTERED_SELECTORS, axis_frame, classify, effective_rank,
                                iqr)
from eval.ood import auroc, energy_score, msp_score                            # noqa: E402
from eval.tail import per_class_error                                          # noqa: E402
from e_tgrid_report import statistics_at                                       # noqa: E402
from recon_frame import ReconRunFrame                                          # noqa: E402
from provenance import code_stamp                                              # noqa: E402

FWD = os.path.join(ROOT, "results", "forward_tier2")
OUT = os.path.join(ROOT, "results", "tier2", "battery_tier2.json")
N_CLASSES = 14
N_POINTS = 20
AXES = ("ID", "WC", "OOD")
GRID = tuple(range(N_POINTS))
POOLS = ("C1M-C-local", "ISIC2019")
SCORES = ("msp", "energy")
ENERGY_T = 1.0
T_GRID = (0.5, 1.0, 2.0, 5.0)
TAG = "TIER2-VERIFIED-OFFICIAL-DATA  (own table; never pooled with Tier 1)"
GATE_NAME = {"C1": "NA"}          # §5 vocabulary -> registered gate name; same estimand


class T2Run:
    def __init__(self, run_id: str) -> None:
        d = os.path.join(FWD, run_id)
        lab = np.load(os.path.join(d, "labels.npz"))
        self.run_id = run_id
        self.y = np.asarray(lab["y_test"], np.int64)
        self.y_split = np.asarray(lab["y_split_noisy"], np.int64)
        self.tail = np.asarray(lab["tail_classes"], np.int64)   # frozen, not recomputed
        pts = sorted(int(x[2:]) for x in os.listdir(d) if x.startswith("pt"))
        if pts != list(range(1, N_POINTS + 1)):
            raise SystemExit(f"{run_id}: points {pts[:3]}… != 1..{N_POINTS}")
        n = len(self.y)
        self.correct = np.zeros((N_POINTS, n), bool)
        self.pce = np.zeros((N_POINTS, N_CLASSES))
        self.s_test = {u: np.zeros((N_POINTS, n)) for u in SCORES}
        self.s_pool: Dict = {}
        self.na_acc, self.er = np.zeros(N_POINTS), np.zeros(N_POINTS)
        self.e_stat = {f"T{t:g}": np.zeros(N_POINTS) for t in T_GRID}
        for i, pt in enumerate(pts):
            p = os.path.join(d, f"pt{pt:03d}")
            lt = np.load(os.path.join(p, "logits_test.npy"))
            self.correct[i] = lt.argmax(1) == self.y
            cor, tot = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
            np.add.at(tot, self.y, 1.0)
            np.add.at(cor, self.y[self.correct[i]], 1.0)
            self.pce[i] = per_class_error(cor, tot)
            self.s_test["energy"][i] = energy_score(lt, ENERGY_T)
            self.s_test["msp"][i] = msp_score(lt)
            for pl in POOLS:
                lp = np.load(os.path.join(p, f"logits_{pl}.npy"))
                for u, fn in (("energy", lambda x: energy_score(x, ENERGY_T)), ("msp", msp_score)):
                    self.s_pool.setdefault((u, pl), np.zeros((N_POINTS, len(lp))))[i] = fn(lp)
            ls = np.load(os.path.join(p, "logits_split.npy"))
            self.na_acc[i] = float((ls.argmax(1) == self.y_split).mean())
            for t in T_GRID:
                self.e_stat[f"T{t:g}"][i] = statistics_at(ls, t)["mean_max_softmax"]
            self.er[i] = effective_rank(np.load(os.path.join(p, "feats_split.npy")))

    def r_id(self): return 1.0 - self.correct.mean(axis=1)

    def r_wc(self):
        return np.asarray([float(np.nanmean(self.pce[i][self.tail])) for i in range(N_POINTS)])

    def r_ood_pair(self, u, pl):
        return np.asarray([1.0 - auroc(self.s_test[u][i], self.s_pool[(u, pl)][i])
                           for i in range(N_POINTS)])

    def r_ood(self):
        return np.mean(np.vstack([self.r_ood_pair("energy", p) for p in POOLS]), axis=0)

    def selectors(self):
        return {"E_tau1": int(np.argmax(self.e_stat["T1"])),
                "C1": int(np.argmax(self.na_acc)),
                "ER_argmax": int(np.argmax(self.er))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    a = p.parse_args(argv)
    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to emit the Tier-2 battery from a dirty tree")

    per_run = []
    for rid in a.runs:
        r = T2Run(rid)
        axes = {"ID": axis_frame("ID", r.r_id()), "WC": axis_frame("WC", r.r_wc()),
                "OOD": axis_frame("OOD", r.r_ood())}
        fr = ReconRunFrame(run_id=rid, axes=axes,
                           excluded_axes=[k for k, v in axes.items() if v.excluded],
                           axis_names=AXES, grid=GRID)
        sel = r.selectors()
        # Amendment §5 names the noisy-agreement selector "C1"; the registered frame calls the
        # same estimand "NA" (theory_spec_v2: 'NA, formerly C1'), and classify() filters
        # selector_J by REGISTERED_SELECTORS. Keying it "C1" therefore dropped it from the
        # gate silently: only two of three selectors decided solved/unsolved, and a run whose
        # C1 met delta came out compatible-UNSOLVED instead of compatible-solved. A4 keeps the
        # §5 vocabulary; only the gate is renamed, and the assert makes future drift loud.
        selJ = {GATE_NAME.get(s, s): fr.J(i) for s, i in sel.items()}
        assert set(selJ) == set(REGISTERED_SELECTORS), f"gate keys {sorted(selJ)}"
        rho = fr.rho_star_le()

        # selector degeneracy, disclosed rather than left for a reader to notice
        deg = {}
        ks = list(sel)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if sel[ks[i]] == sel[ks[j]]:
                    deg[f"{ks[i]}=={ks[j]}"] = int(GRID[sel[ks[i]]]) + 1

        row = dict(
            run_id=rid, classification=TAG,
            A1=dict(axes={k: dict(t_star_point=int(GRID[v.t_star]) + 1, R_star=v.R_star,
                                  d=v.d, iqr_20=float(iqr(v.risk)),
                                  excluded_fail_closed=v.excluded,
                                  ghat=None if v.excluded else v.ghat.tolist(),
                                  risk=v.risk.tolist()) for k, v in axes.items()},
                    tail_classes=[int(c) for c in r.tail],
                    tail_source="frozen at sealing from noisy-label frequency"),
            A2=dict(rho_star_LE=rho, w_delta_denominator=N_POINTS,
                    per_delta={f"{d:g}": dict(
                        F_delta=[int(GRID[i]) + 1 for i in fr.feasible_set(d)],
                        w_delta=fr.w_delta(d), taxonomy=classify(rho, selJ, d))
                        for d in DELTAS},
                    indeterminate_band=INDETERMINATE_BAND,
                    gated_by=list(REGISTERED_SELECTORS), gate_name_map=GATE_NAME),
            A4=dict(selectors={s: dict(point=int(GRID[i]) + 1, J_LE=fr.J(i), eta_LE=fr.eta(i),
                                       per_axis_ghat={k: (None if axes[k].excluded
                                                          else float(axes[k].ghat[i]))
                                                      for k in AXES})
                               for s, i in sel.items()},
                    degeneracy=deg,
                    degeneracy_note=("two registered selectors choosing the same point are "
                                     "the same number, not corroboration")),
            A5=dict(per_delta={f"{d:g}": fr.uniform_baseline(d) for d in DELTAS}),
            # Reported alongside, never gating. The registered E procedure selects per
            # temperature and labels the run T-robust or T-sensitive; the (b) pin requires the
            # full effective-rank curve and the argmin sensitivity; the C1 pin requires the
            # accuracy curve and its limitation text. All were being computed and discarded.
            # They stay OUT of `sel` so they cannot gate and cannot pollute the §4 degeneracy
            # disclosure with pairs like E_tau1==E_T1.
            non_gating=dict(
                E_point_by_T={k: int(np.argmax(v)) + 1 for k, v in r.e_stat.items()},
                E_t_label=("T-robust"
                           if len({int(np.argmax(v)) for v in r.e_stat.values()}) == 1
                           else "T-sensitive"),
                E_curves_by_T={k: v.tolist() for k, v in r.e_stat.items()},
                effective_rank_curve=r.er.tolist(),
                ER_argmin_sensitivity_point=int(np.argmin(r.er)) + 1,
                NA_accuracy_curve=r.na_acc.tolist(),
                NA_limitation=("accuracy against noisy labels rises as the network memorises "
                               "them, so late points are inflated for reasons unrelated to "
                               "generalisation"),
                evaluated_on=(
                    "UNREGISTERED CHOICE, needs ratification: E and effective rank are "
                    "computed on the 5,000-row C1 split, not on the full training sample. "
                    "Amendment §5 pins the 5,000 draw for C1 ONLY and is silent on the input "
                    "for E and effective rank in Tier 2; the (b) pin says verbatim 'Input: "
                    "feats_train (all samples, no subsampling)', which Tier 1 satisfies. A "
                    "full-train forward is infeasible here (1M x 2048 float32 is ~8 GB of "
                    "features per checkpoint, ~640 GB over 80). Two consequences: effective "
                    "rank at n=5,000 over d=2,048 is not converged, and draw_split is "
                    "equal-per-class while the noisy train set runs 18,976 to 88,588 per "
                    "class, so Tier-2 E estimates a CLASS-BALANCED mean max-softmax -- a "
                    "different estimand from Tier 1's, not a subsample of it.")),
            A9=dict(per_score_pool={f"{u}|{pl}": r.r_ood_pair(u, pl).tolist()
                                    for u in SCORES for pl in POOLS},
                    aggregation=dict(
                        mean=r.r_ood().tolist(),
                        min=np.min(np.vstack([r.r_ood_pair("energy", p) for p in POOLS]),
                                   axis=0).tolist(),
                        max=np.max(np.vstack([r.r_ood_pair("energy", p) for p in POOLS]),
                                   axis=0).tolist())))
        per_run.append(row)
        t = row["A2"]["per_delta"][f"{DELTA_PRIMARY:g}"]
        print(f"  {rid:26} rho*={rho:.4f} |F|={len(t['F_delta'])} w={t['w_delta']:.3f} "
              f"{t['taxonomy']:22} oracles ID/WC/OOD = "
              f"{row['A1']['axes']['ID']['t_star_point']}/"
              f"{row['A1']['axes']['WC']['t_star_point']}/"
              f"{row['A1']['axes']['OOD']['t_star_point']}"
              f"{'  DEGENERATE:'+','.join(deg) if deg else ''}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(dict(classification=TAG, code_stamp=stamp, axes=list(AXES),
                       n_points=N_POINTS, deltas=list(DELTAS),
                       delta_primary=DELTA_PRIMARY, pools=list(POOLS),
                       never_pooled_with_tier1=True, per_run=per_run), fh, indent=1)
    os.replace(tmp, OUT)
    print(f"[t2] wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
