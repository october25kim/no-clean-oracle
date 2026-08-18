"""T11 — joint paired bootstrap, under the registered fixed-cells estimand.

Registered pin (docs/remediation_plan_v2.md, "T11 estimand pin", appended before execution):

    psi = (1/12) * SUM_c P_seed(incompatible in cell c)

The 12 conditions are a **designed factorial grid**, not a sample from a population of
conditions, so no superpopulation-of-cells claim is made and none may be read out of the
interval. The grid is exactly 12 cells x 3 seeds = 36 runs, gated by ``verify_frame`` before any
resampling begins -- 36 runs, 12 cells, 3 distinct seeds each, or the run refuses to start.

**Two layers, reported separately and never combined into one number.**

(i) *Within-run evaluation bootstrap.* ID indices are drawn ONCE per replicate and shared
    across all three axes and all 24 checkpoints, so the correlation between axes and along
    the trajectory survives the resample -- resampling each axis independently would break
    exactly the dependence the audit is about. OOD pools are resampled independently of the
    ID sample and of each other. The worst-class subset is re-derived **inside the cross-fit
    folds**: fold k is scored with a tail proposed on the complement of fold k. An earlier
    draft proposed and scored the tail on the same weights, which reuses the fluctuations
    that selected it -- exactly the defect T10 exists to measure, committed inside the
    bootstrap that was supposed to quantify it. The oracle, the IQR denominator, the
    normalized regret, the feasible set and the class are all recomputed per replicate.
    Reported as F_cert / F_poss, over the full delta grid and all three OOD aggregations.

(ii) *Across-run uncertainty* with seed clustering preserved, stratified by cell: seeds are
    resampled within their own cell, so the three seeds of a cell move together as the
    cluster they are, and every replicate keeps all 12 cells. Reported as an interval for psi.

    **The input to psi is the REGISTERED per-seed taxonomy, not the layer-(i) bootstrap
    frequency.** P_seed is a probability over seeds; substituting each run's
    P_boot(incompatible) would average evaluation noise into a seed-level quantity and
    produce a joint seed-x-resample number wearing the estimand's name -- the one thing the
    pin forbids in as many words. The layer-(i) quantities stay in ``per_run`` and are never
    merged into psi.

**F_cert and F_poss.** F_cert is the set of checkpoints feasible in EVERY replicate --
certified: adequate however the evaluation sample fell. F_poss is the set feasible in at
least one -- possible: not excluded by the evidence. F_cert subset F_poss always, and the
gap between them is the evaluation-noise width of the feasible window.

**Registered scope that folds in here.** The delta grid (0.05, 0.10, 0.20), the OOD
aggregation arms (mean / min / max), and certificate robustness -- T6's two-world disjoint
pairs recomputed per replicate over all six (score, pool) combinations, reported as the
fraction of replicates in which each pair *stays* disjoint. That fraction is the only
quantity here that can promote or demote a certificate.

**Selector indices are fixed, their J values are not.** E(tau=1), NA and ER-argmax are
functions of TRAIN logits and features, which no clean-test resample can move, so the
selected checkpoint is fixed across replicates -- the same argument T10 uses. What moves is
J, because J is read off axes rebuilt on the resample.

**The AUROC inner loop is exact, not approximate.** A naive implementation re-sorts every
replicate and costs ~22 ms; at 24 checkpoints x 2 pools x 1000 replicates x 36 runs that is
prohibitive. But the sort order of the ID scores and the insertion positions of the pool
scores are FIXED per checkpoint: resampling changes only multiplicities. Precomputing both
turns each replicate into a weighted cumulative sum, 29x faster, and verified bit-identical
to the registered ``eval.ood.auroc`` on an unweighted draw (difference 0.00e+00). Ties are
carried with the same 0.5 weight the registered implementation uses.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.corrected import (CKPT_GRID, DELTA_PRIMARY, DELTAS, REGISTERED_SELECTORS,
                                RunFrame, axis_frame, classify, stratified_folds)
from analysis.io import load_run
from eval.ood import energy_score, msp_score
from eval.tail import per_class_error, static_tail_classes
from provenance import code_stamp

N_T = len(CKPT_GRID)
SEMANTIC = ("svhn", "cross_cifar")
POOLS = ("svhn", "cross_cifar", "CIFAR-C-local")
SCORES = ("msp", "energy")
K_FOLD = 5
TAIL_FRAC = 0.30
RNG_SEED = 20260814
N_CELLS = 12
FROOT = os.path.join(ROOT, "results", "forward_ext")
RROOT = os.path.join(ROOT, "results", "runs_ext")
BATTERY = os.path.join(ROOT, "results", "corrected", "battery_tier1.json")
OUT = os.path.join(ROOT, "results", "corrected", "t11_joint_bootstrap.json")


class FastAUROC:
    """AUROC under resampling, exact. Precomputes the parts a resample cannot change."""

    def __init__(self, id_scores: np.ndarray, pool_scores: np.ndarray) -> None:
        self.order = np.argsort(id_scores, kind="stable")
        s = id_scores[self.order]
        self.lo = np.searchsorted(s, pool_scores, side="left")
        self.hi = np.searchsorted(s, pool_scores, side="right")

    def __call__(self, w_id_raw: np.ndarray, w_pool: np.ndarray) -> float:
        wa = w_id_raw[self.order].astype(np.float64)
        cum = np.concatenate([[0.0], np.cumsum(wa)])
        Wa = cum[-1]
        Wb = w_pool.sum()
        if Wa <= 0 or Wb <= 0:
            return float("nan")
        greater = Wa - cum[self.hi]
        ties = cum[self.hi] - cum[self.lo]
        return float((w_pool * (greater + 0.5 * ties)).sum() / (Wa * Wb))


class T11Run:
    def __init__(self, run_id: str, meta: dict, energy_T: float) -> None:
        self.run_id, self.meta = run_id, meta
        self.n_classes = int(meta["n_classes"])
        d = os.path.join(FROOT, run_id)
        self.y = np.asarray(np.load(os.path.join(d, "labels.npz"))["test"], np.int64)
        eps = sorted(int(x[2:]) for x in os.listdir(d) if x.startswith("ep"))
        assert eps == list(CKPT_GRID), f"{run_id}: grid mismatch"
        n = self.y.size
        self.correct = np.zeros((N_T, n), bool)
        self.s_test: Dict[str, np.ndarray] = {u: np.zeros((N_T, n)) for u in SCORES}
        self.s_pool: Dict[Tuple[str, str], np.ndarray] = {}
        for i, ep in enumerate(eps):
            e = os.path.join(d, f"ep{ep:03d}")
            lt = np.load(os.path.join(e, "logits_test.npy"))
            self.correct[i] = lt.argmax(1) == self.y
            self.s_test["energy"][i] = energy_score(lt, energy_T)
            self.s_test["msp"][i] = msp_score(lt)
            for p in POOLS:
                lp = np.load(os.path.join(e, f"logits_{p}.npy"))
                for u, fn in (("energy", lambda x: energy_score(x, energy_T)), ("msp", msp_score)):
                    self.s_pool.setdefault((u, p), np.zeros((N_T, len(lp))))[i] = fn(lp)
        self.fast = {(u, p): [FastAUROC(self.s_test[u][i], self.s_pool[(u, p)][i])
                              for i in range(N_T)] for u in SCORES for p in POOLS}
        # Cross-fit folds are a property of the evaluation sample, not of a replicate, so
        # they are fixed once here with the registered helper and reused by every replicate.
        fold_id = stratified_folds(self.y, K_FOLD, RNG_SEED)
        self.folds = [np.asarray(fold_id) == k for k in range(K_FOLD)]

    # ---- weighted risks -------------------------------------------------------
    def r_id_w(self, w: np.ndarray) -> np.ndarray:
        W = w.sum()
        return np.asarray([1.0 - float(w[self.correct[i]].sum()) / W for i in range(N_T)])

    def pce_w(self, i: int, w: np.ndarray) -> np.ndarray:
        tot = np.bincount(self.y, weights=w, minlength=self.n_classes)
        cor = np.bincount(self.y[self.correct[i]], weights=w[self.correct[i]],
                          minlength=self.n_classes)
        return per_class_error(cor, tot)

    def r_wc_w(self, w: np.ndarray, classes: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.nanmean(self.pce_w(i, w)[classes])) for i in range(N_T)])

    def r_ood_pair_w(self, u: str, p: str, w_id: np.ndarray,
                     w_pool: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([1.0 - self.fast[(u, p)][i](w_id, w_pool[p]) for i in range(N_T)])

    def r_ood_w(self, w_id: np.ndarray, w_pool: Dict[str, np.ndarray],
                agg: str = "mean") -> np.ndarray:
        cols = np.vstack([self.r_ood_pair_w("energy", p, w_id, w_pool) for p in SEMANTIC])
        return {"mean": cols.mean(axis=0), "min": cols.min(axis=0),
                "max": cols.max(axis=0)}[agg]

    def tail_w(self, w: np.ndarray) -> np.ndarray:
        """Tail re-derived INSIDE the replicate, from the final grid epoch."""
        return static_tail_classes(1.0 - self.pce_w(N_T - 1, w), TAIL_FRAC)


def wc_crossfit(run: "T11Run", ref: "T11Run", w_id: np.ndarray) -> Tuple[np.ndarray, list]:
    """WC risk with the tail proposed OUT OF FOLD and scored IN FOLD.

    The pin says the subset is "re-derived inside the cross-fit folds". Deriving it from the
    same weights it is then scored on -- which an earlier draft did -- reuses the very
    fluctuations that selected it, which is precisely the defect T10 exists to measure. Here
    fold k is scored with a tail proposed on the complement of fold k, and the folds are
    averaged.
    """
    cols, tails = [], []
    for m in run.folds:
        cls = static_tail_classes(1.0 - ref.pce_w(N_T - 1, w_id * (~m)), TAIL_FRAC)
        cols.append(run.r_wc_w(w_id * m, cls))
        tails.append(tuple(int(c) for c in cls))
    return np.mean(np.vstack(cols), axis=0), tails


def replicate_frame(run: "T11Run", ref: "T11Run", w_id: np.ndarray,
                    w_pool: Dict[str, np.ndarray], sel: Dict[str, int],
                    wc_arm: str = "crossfit") -> dict:
    if wc_arm == "crossfit":
        wc, tails = wc_crossfit(run, ref, w_id)
    elif wc_arm == "plain":                       # tail selected and scored on the same draw
        cls = ref.tail_w(w_id)
        wc, tails = run.r_wc_w(w_id, cls), [tuple(int(c) for c in cls)]
    else:                                          # control: registered full-sample tail
        cls = ref.tail_w(np.ones_like(w_id))
        wc, tails = run.r_wc_w(w_id, cls), [tuple(int(c) for c in cls)]

    r_id = run.r_id_w(w_id)
    out = {}
    for agg in ("mean", "min", "max"):
        axes = {"ID": axis_frame("ID", r_id), "WC": axis_frame("WC", wc),
                "OOD": axis_frame("OOD", run.r_ood_w(w_id, w_pool, agg))}
        excluded = [a for a, f in axes.items() if f.excluded]
        fr = RunFrame(run_id=run.run_id, axes=axes, excluded_axes=excluded)
        rho = fr.rho_star_le()
        sel_J = {t: fr.J(i) for t, i in sel.items() if t in REGISTERED_SELECTORS}
        out[agg] = dict(rho=rho,
                        F={f"{d:g}": set(int(i) for i in fr.feasible_set(d)) for d in DELTAS},
                        taxonomy={f"{d:g}": classify(rho, sel_J, d) for d in DELTAS})

    # T6 under bootstrap: per (score, pool) OOD-axis feasible sets at the primary delta,
    # holding ID and WC fixed -- the same construction a9 uses on the point estimate.
    feas = {}
    for u in SCORES:
        for pl in POOLS:
            axes = {"ID": axis_frame("ID", r_id), "WC": axis_frame("WC", wc),
                    "OOD": axis_frame("OOD", run.r_ood_pair_w(u, pl, w_id, w_pool))}
            f2 = RunFrame(run_id=run.run_id, axes=axes,
                          excluded_axes=[a for a, v in axes.items() if v.excluded])
            feas[f"{u}|{pl}"] = set(int(i) for i in f2.feasible_set(DELTA_PRIMARY))
    disjoint = set()
    for u in SCORES:
        ks = [f"{u}|{pl}" for pl in POOLS]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = feas[ks[i]], feas[ks[j]]
                if a and b and not (a & b):
                    disjoint.add(f"{ks[i]}~~{ks[j]}")
    return dict(agg=out, tails=tails, disjoint_pairs=disjoint)


def bootstrap_run(run: "T11Run", ref: "T11Run", sel: Dict[str, int], B: int,
                  seed: int, wc_arm: str) -> dict:
    rng = np.random.default_rng(seed)
    n = run.y.size
    pool_n = {p: run.s_pool[("energy", p)].shape[1] for p in POOLS}
    aggs = ("mean", "min", "max")
    dkeys = [f"{d:g}" for d in DELTAS]
    rhos = {a: [] for a in aggs}
    taxa = {a: {d: [] for d in dkeys} for a in aggs}
    cert: Dict[str, Optional[set]] = {a: None for a in aggs}
    poss: Dict[str, set] = {a: set() for a in aggs}
    F_cert = {a: {d: None for d in dkeys} for a in aggs}
    F_poss = {a: {d: set() for d in dkeys} for a in aggs}
    disjoint_counts: Dict[str, int] = {}
    tails_seen = set()

    for _b in range(B):
        # ONE ID draw per replicate, shared across all axes and all checkpoints.
        w_id = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        w_pool = {p: np.bincount(rng.integers(0, m, m), minlength=m).astype(np.float64)
                  for p, m in pool_n.items()}
        r = replicate_frame(run, ref, w_id, w_pool, sel, wc_arm)
        for a in aggs:
            rhos[a].append(r["agg"][a]["rho"])
            for d in dkeys:
                taxa[a][d].append(r["agg"][a]["taxonomy"][d])
                F = r["agg"][a]["F"][d]
                F_cert[a][d] = F if F_cert[a][d] is None else (F_cert[a][d] & F)
                F_poss[a][d] |= F
        for k in r["disjoint_pairs"]:
            disjoint_counts[k] = disjoint_counts.get(k, 0) + 1
        tails_seen.update(r["tails"])

    row = dict(run_id=run.run_id, B=B, wc_arm=wc_arm, boot_seed=seed,
               distinct_tail_sets=len(tails_seen), per_agg={})
    for a in aggs:
        rho = np.asarray(rhos[a], float)
        fin = rho[np.isfinite(rho)]
        per_d = {}
        for d, dv in zip(dkeys, DELTAS):
            c = taxa[a][d]
            per_d[d] = dict(
                F_cert=sorted(int(CKPT_GRID[i]) for i in (F_cert[a][d] or set())),
                F_poss=sorted(int(CKPT_GRID[i]) for i in F_poss[a][d]),
                n_cert=len(F_cert[a][d] or set()), n_poss=len(F_poss[a][d]),
                taxonomy_counts={t: c.count(t) for t in set(c)},
                P_incompatible=c.count("incompatible") / B,
                P_rho_gt_delta=float(np.mean(fin > dv)) if fin.size else float("nan"))
        row["per_agg"][a] = dict(
            rho_mean=float(fin.mean()) if fin.size else float("nan"),
            rho_ci=[float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))]
            if fin.size else [float("nan")] * 2,
            n_finite=int(fin.size), per_delta=per_d)
    row["certificates_under_bootstrap"] = {
        k: dict(replicates_disjoint=v, fraction=v / B) for k, v in sorted(disjoint_counts.items())}
    return row


def psi_interval(cells: Dict[str, List[float]], B: int, seed: int) -> dict:
    """Layer (ii): cluster bootstrap over seeds WITHIN cell, stratified so all 12 remain.

    psi is a mean over a fixed set of 12 cells, so cells are never resampled -- resampling
    them would smuggle in the superpopulation claim the pin forbids. Only the seeds inside
    each cell are resampled, because the seed is the unit that could have come out otherwise.
    """
    rng = np.random.default_rng(seed)
    names = sorted(cells)
    assert len(names) == N_CELLS, f"{len(names)} cells, expected {N_CELLS}"
    point = float(np.mean([np.mean(cells[c]) for c in names]))
    draws = []
    for _b in range(B):
        vals = []
        for c in names:
            v = np.asarray(cells[c], float)
            vals.append(float(np.mean(rng.choice(v, size=v.size, replace=True))))
        draws.append(float(np.mean(vals)))
    d = np.asarray(draws)
    return dict(psi=point, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                se=float(d.std(ddof=1)), B=B, n_cells=len(names),
                per_cell={c: float(np.mean(cells[c])) for c in names},
                estimand="fixed-cells: psi = (1/12) sum_c P_seed(incompatible in cell c); "
                         "cells are a designed grid and are NOT resampled")


def verify_frame(run_ids: List[str], metas: Dict[str, dict]) -> Dict[str, List[str]]:
    """The structural gate, run BEFORE anything is resampled.

    An earlier draft claimed in its docstring that the grid was "verified against the frame
    before anything is resampled" while the only check ran after every within-run bootstrap
    had already executed. A gate that fires after the work is not a gate.
    """
    cells: Dict[str, List[str]] = {}
    for rid in run_ids:
        m = metas[rid]
        cells.setdefault(f"{m.get('split') or m['dataset']}|{m['learner']}", []).append(rid)
    if len(run_ids) != 36:
        raise SystemExit(f"frame has {len(run_ids)} runs, expected 36")
    if len(cells) != N_CELLS:
        raise SystemExit(f"frame has {len(cells)} cells, expected {N_CELLS}")
    for c, rs in sorted(cells.items()):
        seeds = sorted(int(metas[r]["seed"]) for r in rs)
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise SystemExit(f"cell {c} has seeds {seeds}, expected 3 distinct")
    return cells


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1000, help="within-run replicates")
    p.add_argument("--B2", type=int, default=10000, help="across-run cluster replicates")
    p.add_argument("--wc-arm", choices=["crossfit", "plain", "control"], default="crossfit")
    p.add_argument("--delta", type=float, default=DELTA_PRIMARY)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out_path = a.out or OUT.replace(".json", f"_{a.wc_arm}.json")

    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to run T11 from a dirty tree")
    energy_T = float(yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))
                     ["eval"]["ood_energy_T"])
    battery = {r["run_id"]: r for r in json.load(open(BATTERY))["per_run"]}

    run_ids = sorted(d for d in os.listdir(FROOT) if os.path.isdir(os.path.join(FROOT, d)))
    metas = {r: load_run(os.path.join(RROOT, r)).meta for r in run_ids}
    if a.limit:
        run_ids = run_ids[:a.limit]
        cell_map = None
        print(f"  [debug] --limit {a.limit}: structural gate SKIPPED, psi not computable")
    else:
        cell_map = verify_frame(run_ids, metas)
        print(f"  frame gate OK: 36 runs, {len(cell_map)} cells, 3 distinct seeds each")

    # LAYER (ii) INPUT: the pinned P_seed is a SEED-level frequency of the REGISTERED
    # taxonomy. Feeding the layer-(i) bootstrap frequency here instead would combine the two
    # layers into one number, which the pin forbids in as many words. The layer-(i) numbers
    # stay in per_run and are never averaged into psi.
    cells_registered: Dict[str, List[float]] = {}
    for rid in run_ids:
        m = metas[rid]
        tx = battery[rid]["A2"]["per_delta"][f"delta_{a.delta:g}"]["taxonomy"]
        cells_registered.setdefault(f"{m.get('split') or m['dataset']}|{m['learner']}",
                                    []).append(1.0 if tx == "incompatible" else 0.0)

    cache: Dict[str, T11Run] = {}
    per_run = []
    seeds_used = set()
    for k, rid in enumerate(run_ids):
        m = metas[rid]
        ref_id = (f"{m['split']}_ce_seed0" if m.get("split")
                  else f"{m['dataset']}_{m['noise_type']}{m['eta']:g}_ce_seed0")
        for need in (rid, ref_id):
            if need not in cache:
                cache[need] = T11Run(need, metas.get(need) or
                                     load_run(os.path.join(RROOT, need)).meta, energy_T)
        sel = {t: v["grid_index"] for t, v in battery[rid]["A4"]["selectors"].items()}
        # sha256, not hash(): Python's str hash is salted per process, so the previous seed
        # was different on every invocation and unrecoverable afterwards.
        rs = RNG_SEED + int.from_bytes(hashlib.sha256(rid.encode()).digest()[:8], "big") % 10_000_019
        if rs in seeds_used:
            raise SystemExit(f"bootstrap seed collision at {rid}")
        seeds_used.add(rs)
        row = bootstrap_run(cache[rid], cache[ref_id], sel, a.B, rs, a.wc_arm)
        row["cell"] = f"{m.get('split') or m['dataset']}|{m['learner']}"
        row["train_seed"] = int(m["seed"])
        row["registered_taxonomy"] = battery[rid]["A2"]["per_delta"][f"delta_{a.delta:g}"]["taxonomy"]
        per_run.append(row)
        pd = row["per_agg"]["mean"]["per_delta"][f"{a.delta:g}"]
        print(f"  [{k+1:2d}/{len(run_ids)}] {rid:26} P(incomp|boot)={pd['P_incompatible']:.3f} "
              f"reg={row['registered_taxonomy'][:12]:12} F_cert={pd['n_cert']:2d} "
              f"F_poss={pd['n_poss']:2d} tails={row['distinct_tail_sets']} "
              f"certs={len(row['certificates_under_bootstrap'])}", flush=True)
        if rid != ref_id:
            cache.pop(rid, None)

    out = dict(code_stamp=stamp, delta_primary=a.delta, deltas=list(DELTAS),
               B_within=a.B, B_across=a.B2, wc_arm=a.wc_arm,
               layers=dict(
                   within=("ID indices drawn once per replicate and shared across axes and "
                           "checkpoints; pools independent; WC tail proposed out-of-fold and "
                           "scored in-fold; oracle/IQR/regret/feasible set/class recomputed"),
                   across=("seed cluster bootstrap within cell, stratified; cells never "
                           "resampled; input is the REGISTERED per-seed taxonomy, never the "
                           "layer-(i) bootstrap frequency")),
               per_run=per_run,
               psi=(psi_interval(cells_registered, a.B2, RNG_SEED)
                    if cell_map is not None else dict(skipped="--limit set")))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, out_path)
    print(f"[t11] wrote {os.path.relpath(out_path, ROOT)}")
    if "psi" in out["psi"]:
        q = out["psi"]
        print(f"[t11] psi = {q['psi']:.4f}  95% CI [{q['ci95'][0]:.4f}, {q['ci95'][1]:.4f}]  "
              f"(layer ii only; layer i is reported per-run and never merged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
