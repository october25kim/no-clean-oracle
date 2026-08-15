"""Remediation battery A1–A14 (minus A10b TGS-N), per plan v2 and theory spec v2.

Runs on stored artifacts only: the per-epoch JSONL logs and the forward-pass logits. No
GPU, no training, no checkpoint is read. Outputs go to ``results/corrected/`` with a
``code_stamp`` on every file.

Why the risks are recomputed from logits rather than read from the logs: the cross-fitted
benchmarks (A6), the OOD decomposition (A9) and the budget curves (A11) all need the risk
on *subsets* of the evaluation sample, which a logged scalar cannot provide. Recomputing
the full-sample value from the same logits is also a free consistency check against the
log, and that check is reported per run.

Layer discipline (spec A.4) is carried in the output rather than left to the reader: every
number is tagged LE, CF, or FROZEN-HISTORICAL, and ``eta`` is only ever formed from two
LE-layer terms.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from config import load_config                                    # noqa: E402
from provenance import code_stamp                                 # noqa: E402
from analysis.io import load_run, static_tail_from_reference      # noqa: E402
from analysis.ncr import bca_ci, moving_average                   # noqa: E402
from analysis.corrected import (AXES, CKPT_GRID, DELTAS, DELTA_PRIMARY, FLOORS,
                                INDETERMINATE_BAND, K_CV, REGISTERED_SELECTORS,
                                RNG_SEED, RunFrame, axis_frame, classify,
                                corrected_oracle, effective_rank, iqr,
                                random_folds, stratified_folds)         # noqa: E402
from eval.ood import auroc, energy_score, msp_score               # noqa: E402
from eval.tail import per_class_error                             # noqa: E402
from e_tgrid_report import T_GRID, statistics_at                  # noqa: E402
from c1_noisyval_report import SPLIT_N, SPLIT_SEED, draw_split, _ce  # noqa: E402
from analysis.label_wave import PAPER_BEST_K, trailing_mean       # noqa: E402

OUT = os.path.join(ROOT, "results", "corrected")
N_T = len(CKPT_GRID)
SEMANTIC = ("svhn", "cross_cifar")
POOLS = ("svhn", "cross_cifar", "CIFAR-C-local")
SCORES = ("msp", "energy")


# --------------------------------------------------------------------------- loading

class RunData:
    """Per-checkpoint evaluation primitives for one run, kept at sample resolution.

    Sample resolution is the whole point: fold-restricted and subsample-restricted risks
    are then simple index operations, and the full-sample risk is the same code path with
    every index selected.
    """

    def __init__(self, run_id: str, fdir: str, meta: dict, wc_classes: np.ndarray,
                 energy_T: float):
        self.run_id = run_id
        self.meta = meta
        self.dataset = meta["dataset"]
        self.learner = meta["learner"]
        self.seed = int(meta["seed"])
        self.group = meta.get("split") or f"{meta['dataset']}_{meta['noise_type']}_{meta['eta']:g}"
        self.wc_classes = np.asarray(wc_classes)
        lab = np.load(os.path.join(fdir, "labels.npz"))
        self.y_test = np.asarray(lab["test"], dtype=np.int64)
        self.noisy = np.asarray(lab["train_noisy"], dtype=np.int64)
        self.n_classes = int(meta["n_classes"])
        eps = sorted(int(os.path.basename(d)[2:])
                     for d in glob.glob(os.path.join(fdir, "ep*")))
        assert eps == list(CKPT_GRID), f"{run_id}: grid {eps[:3]}… != retained grid"
        self.epochs = eps

        self.correct = np.zeros((N_T, self.y_test.size), dtype=bool)
        self.e_test = np.zeros((N_T, self.y_test.size), dtype=np.float64)
        self.m_test = np.zeros((N_T, self.y_test.size), dtype=np.float64)
        self.e_pool: Dict[str, np.ndarray] = {}
        self.m_pool: Dict[str, np.ndarray] = {}
        self.na_acc: List[float] = []
        self.er: List[float] = []
        self.e_stat: Dict[str, List[float]] = {f"T{t:g}": [] for t in T_GRID}

        sel = draw_split(self.noisy, self.n_classes, SPLIT_N, SPLIT_SEED)
        y_sel = self.noisy[sel]
        for i, ep in enumerate(eps):
            d = os.path.join(fdir, f"ep{ep:03d}")
            lt = np.load(os.path.join(d, "logits_test.npy"))
            self.correct[i] = lt.argmax(1) == self.y_test
            self.e_test[i] = energy_score(lt, energy_T)
            self.m_test[i] = msp_score(lt)
            for p in POOLS:
                lp = np.load(os.path.join(d, f"logits_{p}.npy"))
                self.e_pool.setdefault(p, np.zeros((N_T, len(lp))))[i] = energy_score(lp, energy_T)
                self.m_pool.setdefault(p, np.zeros((N_T, len(lp))))[i] = msp_score(lp)
            ltr = np.load(os.path.join(d, "logits_train.npy"))
            self.na_acc.append(float((ltr[sel].argmax(1) == y_sel).mean()))
            for t in T_GRID:
                self.e_stat[f"T{t:g}"].append(statistics_at(ltr, t)["mean_max_softmax"])
            self.er.append(effective_rank(np.load(os.path.join(d, "feats_train.npy"))))

    # ---- risks, optionally restricted to a subset of the evaluation sample ----

    def r_id(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        c = self.correct if idx is None else self.correct[:, idx]
        return 1.0 - c.mean(axis=1)

    def r_wc(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        y = self.y_test if idx is None else self.y_test[idx]
        c = self.correct if idx is None else self.correct[:, idx]
        out = np.zeros(N_T)
        for i in range(N_T):
            corr = np.zeros(self.n_classes); tot = np.zeros(self.n_classes)
            np.add.at(tot, y, 1.0)
            np.add.at(corr, y[c[i]], 1.0)
            pce = per_class_error(corr, tot)
            out[i] = float(np.nanmean(pce[self.wc_classes]))
        return out

    def r_ood_pair(self, score: str, pool: str, id_idx=None, pool_idx=None) -> np.ndarray:
        idm = self.e_test if score == "energy" else self.m_test
        pm = (self.e_pool if score == "energy" else self.m_pool)[pool]
        a = idm if id_idx is None else idm[:, id_idx]
        b = pm if pool_idx is None else pm[:, pool_idx]
        return np.asarray([1.0 - auroc(a[i], b[i]) for i in range(N_T)])

    def r_ood(self, id_idx=None, pool_idx: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        cols = [self.r_ood_pair("energy", p, id_idx,
                                None if pool_idx is None else pool_idx[p])
                for p in SEMANTIC]
        return np.mean(np.vstack(cols), axis=0)

    def risks(self, id_idx=None, pool_idx=None) -> Dict[str, np.ndarray]:
        return {"ID": self.r_id(id_idx), "WC": self.r_wc(id_idx),
                "OOD": self.r_ood(id_idx, pool_idx)}


def frame(run_id: str, risks: Dict[str, np.ndarray], floor: float = 0.0) -> RunFrame:
    axes = {a: axis_frame(a, risks[a], floor) for a in AXES}
    return RunFrame(run_id=run_id, axes=axes,
                    excluded_axes=[a for a, f in axes.items() if f.excluded])


# --------------------------------------------------------------------------- selectors

def selector_indices(rd: RunData) -> Dict[str, int]:
    """Registered selectors, plus LW-N reported but never gating (pin 3)."""
    out = {
        "E_tau1": int(np.argmax(rd.e_stat["T1"])),
        "NA": int(np.argmax(rd.na_acc)),
        "ER_argmax": int(np.argmax(rd.er)),
        "ER_argmin_sensitivity": int(np.argmin(rd.er)),
    }
    for t in T_GRID:
        out[f"E_T{t:g}"] = int(np.argmax(rd.e_stat[f"T{t:g}"]))
    return out


def lw_n_index(run) -> Dict[str, object]:
    """LW-N: Label Wave replayed on the logged one-epoch statistic (F1, A10).

    The statistic is the 120-epoch ``pred_flip_count`` exactly as logged — the 24-grid
    recomputation was formally rejected because it measures a five-epoch difference. The
    selected epoch is then mapped to the nearest retained checkpoint, ties to the earlier.
    """
    pc = np.asarray([e["pred_flip_count"] for e in run.epochs[1:]], dtype=np.float64)
    sm = trailing_mean(pc, PAPER_BEST_K)
    best, t_star = np.inf, None
    for j, v in enumerate(sm):
        if np.isnan(v):
            continue
        if v < best:                      # plateau/tie -> earliest, so strict <
            best, t_star = v, j + 1
    if t_star is None:
        return dict(epoch=None, grid_index=None)
    d = [abs(t_star - g) for g in CKPT_GRID]
    return dict(epoch=int(t_star), grid_index=int(np.argmin(d)),
                mapped_epoch=int(CKPT_GRID[int(np.argmin(d))]))


# --------------------------------------------------------------------------- analyses

def a1(rd: RunData, fr: RunFrame, run) -> dict:
    """Grid + definition coherence: corrected vs frozen-historical oracles."""
    hist = {}
    for a, key in (("ID", "R_ID"), ("OOD", "R_OOD_primary_energy_semantic")):
        y120 = run.series(key)
        t_h = int(np.argmin(moving_average(y120, 3)))
        hist[a] = dict(t_star_120=t_h, R_star_120_raw=float(y120[t_h]),
                       iqr_120=float(iqr(y120)))
    out = {}
    for a in AXES:
        f = fr.axes[a]
        row = dict(t_star_grid_index=f.t_star, t_star_epoch=CKPT_GRID[f.t_star],
                   R_star=f.R_star, d=f.d, excluded_fail_closed=f.excluded,
                   iqr_24=iqr(f.risk), delta_raw=f.delta_raw.tolist(),
                   ghat=None if f.excluded else f.ghat.tolist())
        if a in hist:
            row["frozen_historical"] = hist[a]
            row["delta_epoch_vs_historical"] = CKPT_GRID[f.t_star] - hist[a]["t_star_120"]
        out[a] = row
    return dict(layer="LE vs FROZEN-HISTORICAL, each as originally defined", axes=out)


def a2_a3(fr: RunFrame, sel_J: Dict[str, float]) -> dict:
    rho = fr.rho_star_le()
    per_delta = {}
    for d in DELTAS:
        per_delta[f"delta_{d:g}"] = dict(
            F_delta=[int(CKPT_GRID[i]) for i in fr.feasible_set(d)],
            w_delta=fr.w_delta(d),
            taxonomy=classify(rho, sel_J, d))
    return dict(layer="LE", rho_star_LE=rho, per_delta=per_delta,
                indeterminate_band=INDETERMINATE_BAND,
                gated_by=list(REGISTERED_SELECTORS))


def a4(fr: RunFrame, sel: Dict[str, int]) -> dict:
    rho = fr.rho_star_le()
    out = {}
    for s, gi in sel.items():
        out[s] = dict(grid_index=gi, epoch=int(CKPT_GRID[gi]),
                      J_LE=fr.J(gi), eta_LE=fr.eta(gi),
                      per_axis_ghat={a: (None if fr.axes[a].excluded
                                         else float(fr.axes[a].ghat[gi])) for a in AXES})
    return dict(layer="LE", rho_star_LE=rho, selectors=out,
                nonnegativity="eta is J - rho within the LE layer only (spec D.1)")


def a5(fr: RunFrame) -> dict:
    return dict(layer="LE", exact=True, no_binomial=True,
                per_delta={f"delta_{d:g}": fr.uniform_baseline(d) for d in DELTAS})


def a6(rd: RunData, folds: Dict[str, np.ndarray]) -> dict:
    """Cross-fitted benchmarks (pin 8). A procedure's risk, not an oracle estimate."""
    per_axis, minimax = {}, []
    id_folds = folds["ID"]
    pool_folds = {p: folds[p] for p in POOLS}
    for q in range(K_CV):
        in_id, out_id = np.flatnonzero(id_folds != q), np.flatnonzero(id_folds == q)
        in_p = {p: np.flatnonzero(pool_folds[p] != q) for p in POOLS}
        out_p = {p: np.flatnonzero(pool_folds[p] == q) for p in POOLS}
        r_in, r_out = rd.risks(in_id, in_p), rd.risks(out_id, out_p)
        f_in = frame(rd.run_id, r_in)
        for a in AXES:
            t = corrected_oracle(r_in[a])
            per_axis.setdefault(a, []).append(float(r_out[a][t]))
        if f_in.scorable:
            t = int(np.argmin(f_in.max_g()))
            f_out = frame(rd.run_id, r_out)
            minimax.append(f_out.J(t) if f_out.scorable else float("nan"))
    return dict(layer="CF", K_CV=K_CV,
                per_axis={a: float(np.mean(v)) for a, v in per_axis.items()},
                per_axis_folds={a: v for a, v in per_axis.items()},
                minimax=float(np.nanmean(minimax)) if minimax else float("nan"),
                sign_free="selector-minus-CF may be negative; benchmark excess, not regret")


def a7(rd: RunData, risks: Dict[str, np.ndarray]) -> dict:
    """Set-valued oracles, eps in ABSOLUTE raw-risk units (pin 11)."""
    out = {}
    for eps in (0.005, 0.01, 0.02):
        sets = {a: np.flatnonzero(risks[a] <= risks[a].min() + eps).tolist() for a in AXES}
        cr = {}
        for a in AXES:
            for b in AXES:
                if a == b:
                    continue
                vals = [risks[b][t] - risks[b].min() for t in sets[a]]
                cr[f"{b}|{a}"] = dict(min=float(min(vals)), max=float(max(vals)))
        out[f"eps_{eps:g}"] = dict(sizes={a: len(sets[a]) for a in AXES},
                                   epochs={a: [int(CKPT_GRID[t]) for t in sets[a]]
                                           for a in AXES},
                                   CR_range=cr)
    return dict(layer="LE [SENSITIVITY]", units="absolute raw risk", grids=out)


def a8(rd: RunData, risks: Dict[str, np.ndarray], sel_J_lookup) -> dict:
    """Normalization sensitivity: raw, range-normalized, oracle-relative, floor grid."""
    out = {}
    for a in AXES:
        y = risks[a]
        rng = float(y.max() - y.min())
        out[a] = dict(raw_delta_max=float((y - y.min()).max()),
                      range_normalized_max=float(((y - y.min()) / rng).max())
                      if rng > 0 else None,
                      oracle_relative_applicable=bool(y.min() > 0.01),
                      oracle_relative_max=float(((y - y.min()) / y.min()).max())
                      if y.min() > 0.01 else None)
    floors = {}
    for fl in FLOORS:
        f = frame("floor", risks, floor=fl)
        floors[f"floor_{fl:g}"] = dict(
            rho_star_LE=f.rho_star_le(),
            taxonomy=classify(f.rho_star_le(), sel_J_lookup(f), DELTA_PRIMARY),
            excluded_axes=f.excluded_axes)
    return dict(layer="LE [SENSITIVITY]", per_axis=out, floor_grid=floors)


def a9(rd: RunData, folds) -> dict:
    """OOD decomposition per (score, pool), plus the two-world certificate line."""
    base = {"ID": rd.r_id(), "WC": rd.r_wc()}
    per_uo, feas = {}, {}
    for u in SCORES:
        for o in POOLS:
            r = dict(base); r["OOD"] = rd.r_ood_pair(u, o)
            f = frame(rd.run_id, r)
            key = f"{u}|{o}"
            F = f.feasible_set(DELTA_PRIMARY)
            feas[key] = set(int(CKPT_GRID[i]) for i in F)
            per_uo[key] = dict(rho_star_LE=f.rho_star_le(),
                               F_delta=sorted(feas[key]), w_delta=f.w_delta(DELTA_PRIMARY),
                               ood_t_star_epoch=int(CKPT_GRID[f.axes["OOD"].t_star]),
                               ood_iqr_24=iqr(r["OOD"]),
                               ood_zero_iqr_24=bool(iqr(r["OOD"]) == 0.0))
    # two-world certificate: any pool pair whose OOD-axis feasible sets are disjoint
    pairs, disjoint = [], 0
    for u in SCORES:
        keys = [f"{u}|{o}" for o in POOLS]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = feas[keys[i]], feas[keys[j]]
                dj = bool(a and b and not (a & b))
                pairs.append(dict(pair=[keys[i], keys[j]], disjoint=dj,
                                  sizes=[len(a), len(b)]))
                disjoint += int(dj)
    agg = {}
    for u in SCORES:
        cols = np.vstack([rd.r_ood_pair(u, p) for p in SEMANTIC])
        agg[u] = dict(mean=cols.mean(axis=0).tolist(), min=cols.min(axis=0).tolist(),
                      max=cols.max(axis=0).tolist())
    return dict(layer="LE", per_score_pool=per_uo, aggregation_sensitivity=agg,
                two_world_certificate=dict(pairs=pairs, disjoint_pair_count=disjoint,
                                           certificate_bearing=bool(disjoint > 0)))


def a11(rd: RunData, folds, n_grid=(50, 100, 200, 500, 1000, 2000), B=1000) -> dict:
    """Clean-label budget curves with disjoint selection and assessment (pin 17/18)."""
    rng = np.random.default_rng(RNG_SEED)
    y = rd.y_test
    sel_mask = np.zeros(y.size, dtype=bool)
    for c in np.unique(y):                       # class-stratified 50/50
        idx = np.flatnonzero(y == c)
        sel_mask[rng.permutation(idx)[: idx.size // 2]] = True
    D_sel, D_ass = np.flatnonzero(sel_mask), np.flatnonzero(~sel_mask)
    pool_sel, pool_ass = {}, {}
    for p in SEMANTIC:
        n = rd.e_pool[p].shape[1]
        perm = rng.permutation(n)
        pool_sel[p], pool_ass[p] = perm[: n // 2], perm[n // 2:]

    r_ass = rd.risks(D_ass, {**{p: pool_ass[p] for p in SEMANTIC},
                             "CIFAR-C-local": np.arange(rd.e_pool["CIFAR-C-local"].shape[1])})
    f_ass = frame(rd.run_id, r_ass)
    if not f_ass.scorable:
        return dict(layer="LE", scorable=False)

    out = {}
    for cond in ("ID->ID", "ID->OOD", "OOD->OOD", "mixed->OOD"):
        curve = {}
        for n in n_grid:
            hits = 0
            for b in range(B):
                rb = np.random.default_rng((RNG_SEED, hash(rd.dataset) % 10**6, b, n))
                if cond.startswith("ID"):
                    v = rb.choice(D_sel, size=min(n, D_sel.size), replace=False)
                    t = corrected_oracle(rd.r_id(v))
                elif cond.startswith("OOD"):
                    vi = rb.choice(D_sel, size=min(n // 2, D_sel.size), replace=False)
                    vp = {p: rb.choice(pool_sel[p], size=min(n // 4, pool_sel[p].size),
                                       replace=False) for p in SEMANTIC}
                    t = corrected_oracle(rd.r_ood(vi, vp))
                else:
                    vi = rb.choice(D_sel, size=min(n // 2, D_sel.size), replace=False)
                    vp = {p: rb.choice(pool_sel[p], size=min(n // 4, pool_sel[p].size),
                                       replace=False) for p in SEMANTIC}
                    sub = {"ID": rd.r_id(vi), "WC": rd.r_wc(vi), "OOD": rd.r_ood(vi, vp)}
                    fs = frame("v", sub)
                    t = int(np.argmin(fs.max_g())) if fs.scorable else corrected_oracle(sub["ID"])
                axis = "ID" if cond.endswith("ID") else "OOD"
                hits += int(f_ass.axes[axis].ghat[t] <= DELTA_PRIMARY)
            curve[str(n)] = hits / B
        vals = [curve[str(n)] for n in n_grid]
        monotone = all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))
        reached = [n for n, v in zip(n_grid, vals) if v >= 0.9]
        out[cond] = dict(q=curve, monotone=monotone,
                         n_star=(int(reached[0]) if (reached and monotone) else None),
                         n_max=int(max(n_grid)))
    return dict(layer="LE", B=B, seed=RNG_SEED, assessment="D_assess only", curves=out)


def a12(per_run: List[dict], delta: float = DELTA_PRIMARY) -> dict:
    """LOSO over seeds: does a cell's majority taxonomy class survive dropping a seed?"""
    cells: Dict[str, List[dict]] = {}
    for r in per_run:
        cells.setdefault(f"{r['group']}_{r['learner']}", []).append(r)
    out = {}
    for cell, rows in sorted(cells.items()):
        classes = [r["A3"]["per_delta"][f"delta_{delta:g}"]["taxonomy"] for r in rows]
        def majority(xs):
            return max(sorted(set(xs)), key=xs.count) if xs else None
        full = majority(classes)
        drops = {}
        for i, r in enumerate(rows):
            drops[str(r["seed"])] = majority([c for j, c in enumerate(classes) if j != i])
        out[cell] = dict(n=len(rows), classes=classes, majority=full,
                         leave_one_seed_out=drops,
                         stable=all(v == full for v in drops.values()))
    return dict(layer="LE", delta=delta, cells=out)


def a13(per_run: List[dict]) -> dict:
    """Equivalence-style reanalysis [SENSITIVITY]: margin 0.10 on NCR, BCa 95% upper."""
    vals = [r["A2"]["rho_star_LE"] for r in per_run
            if np.isfinite(r["A2"]["rho_star_LE"])]
    if len(vals) < 2:
        return dict(layer="LE [SENSITIVITY]", n=len(vals), inconclusive=True)
    arr = np.asarray(vals)
    point, lo, hi = bca_ci(arr, n_boot=2000, seed=RNG_SEED)   # returns (mean, lo, hi)
    return dict(layer="LE [SENSITIVITY]", margin=0.10, statistic="mean rho_star_LE",
                point=float(point), ci95=[float(lo), float(hi)],
                upper_below_margin=bool(hi < 0.10),
                note="descriptive; the frozen WEAK verdict is not altered")


def a14(per_run: List[dict]) -> dict:
    """E prediction re-adjudication at both granularities and both metric scales."""
    out = {}
    for scale in ("normalized_regret", "epoch_gap"):
        per_run_axis, medians = [], {}
        for r in per_run:
            sel = r["A4"]["selectors"]["E_tau1"]
            if scale == "normalized_regret":
                vals = {a: sel["per_axis_ghat"][a] for a in AXES}
            else:
                vals = {a: abs(sel["grid_index"] - r["A1"]["axes"][a]["t_star_grid_index"])
                        for a in AXES}
            if any(v is None for v in vals.values()):
                continue
            per_run_axis.append(max(AXES, key=lambda a: vals[a]))
            for a in AXES:
                medians.setdefault(a, []).append(float(vals[a]))
        out[scale] = dict(
            per_run_largest_axis={a: per_run_axis.count(a) for a in AXES},
            aggregate_median={a: float(np.median(v)) for a, v in medians.items()},
            aggregate_largest_axis=(max(medians, key=lambda a: float(np.median(medians[a])))
                                    if medians else None))
    return dict(layer="LE", granularities=["aggregate (median over runs)", "per-run"],
                scales=list(out), results=out)


# --------------------------------------------------------------------------- driver

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frame", choices=["g2", "tier1"], required=True)
    p.add_argument("--analyses", default="A1,A2,A3,A4,A5,A6,A7,A8,A9,A12,A13",
                   help="comma list; A10/A11/A14 are Tier-1 step (5) additions")
    p.add_argument("--out", default=OUT)
    a = p.parse_args(argv)
    want = {x.strip() for x in a.analyses.split(",") if x.strip()}

    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to run with an unattested or dirty tree")

    if a.frame == "g2":
        cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))
        fdir_root = os.path.join(ROOT, "results", "forward_b2")
        runs_root = os.path.join(ROOT, "results", "runs")
    else:
        cfg = load_config(os.path.join(ROOT, "configs", "ext_tier1.yaml"))
        fdir_root = os.path.join(ROOT, "results", "forward_ext")
        runs_root = os.path.join(ROOT, "results", "runs_ext")
    energy_T = float(cfg["eval"]["ood_energy_T"])

    run_ids = sorted(d for d in os.listdir(fdir_root)
                     if os.path.isfile(os.path.join(fdir_root, d, "labels.npz")))
    os.makedirs(a.out, exist_ok=True)
    print(f"[battery] frame={a.frame} runs={len(run_ids)} analyses={sorted(want)}", flush=True)

    fold_cache: Dict[str, Dict[str, np.ndarray]] = {}
    per_run: List[dict] = []
    for run_id in run_ids:
        run = load_run(os.path.join(runs_root, run_id))
        m = run.meta
        ref_id = (f"{m['split']}_ce_seed0" if m.get("split")
                  else f"{m['dataset']}_{m['noise_type']}{m['eta']:g}_ce_seed0")
        wc = static_tail_from_reference(load_run(os.path.join(runs_root, ref_id)))
        rd = RunData(run_id, os.path.join(fdir_root, run_id), m, wc, energy_T)

        if rd.dataset not in fold_cache:                     # fixed once per dataset
            f = {"ID": stratified_folds(rd.y_test, K_CV, RNG_SEED)}
            for pool in POOLS:
                f[pool] = random_folds(rd.e_pool[pool].shape[1], K_CV, RNG_SEED)
            fold_cache[rd.dataset] = f
        folds = fold_cache[rd.dataset]

        risks = rd.risks()
        fr = frame(run_id, risks)
        sel = selector_indices(rd)
        lw = lw_n_index(run)
        if lw["grid_index"] is not None:
            sel["LW_N_reported_not_gating"] = int(lw["grid_index"])
        sel_J = {s: fr.J(i) for s, i in sel.items()}

        row = dict(run_id=run_id, group=rd.group, learner=rd.learner, seed=rd.seed,
                   dataset=rd.dataset, scorable=fr.scorable,
                   excluded_axes=fr.excluded_axes,
                   log_consistency={k: float(abs(risks["ID"][i] - run.epochs[e]["R_ID"]))
                                    for i, (k, e) in enumerate([(str(x), x) for x in CKPT_GRID])})
        if "A1" in want:  row["A1"] = a1(rd, fr, run)
        if "A2" in want or "A3" in want:
            row["A2"] = a2_a3(fr, sel_J); row["A3"] = row["A2"]
        if "A4" in want:  row["A4"] = a4(fr, sel)
        if "A5" in want:  row["A5"] = a5(fr)
        if "A6" in want:  row["A6"] = a6(rd, folds)
        if "A7" in want:  row["A7"] = a7(rd, risks)
        if "A8" in want:  row["A8"] = a8(rd, risks, lambda f: {s: f.J(i) for s, i in sel.items()})
        if "A9" in want:  row["A9"] = a9(rd, folds)
        if "A10" in want: row["A10"] = dict(layer="LE", LW_N=lw, gating=False,
                                            note="TGS-N deferred to A10b")
        if "A11" in want: row["A11"] = a11(rd, folds)
        per_run.append(row)
        print(f"[battery] {run_id} done", flush=True)

    agg = dict(schema=f"corrected_battery_{a.frame}.v1", frame=a.frame,
               n_runs=len(per_run), analyses=sorted(want), delta_primary=DELTA_PRIMARY,
               deltas=list(DELTAS), grid=list(CKPT_GRID), code_stamp=stamp,
               integrity_class="internal-consistency (recomputed from stored logits; "
                               "R4 tolerance 1e-12)",
               forbidden_phrases_bind=True)
    if "A12" in want: agg["A12"] = a12(per_run)
    if "A13" in want: agg["A13"] = a13(per_run)
    if "A14" in want: agg["A14"] = a14(per_run)
    agg["per_run"] = per_run

    path = os.path.join(a.out, f"battery_{a.frame}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(agg, fh, indent=1, default=float)
    os.replace(tmp, path)
    print(f"[battery] wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
