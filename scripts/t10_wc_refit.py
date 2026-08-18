"""T10 — is the WC axis an artifact of picking its classes on the data it is scored on?

The registered worst-class axis fixes its class set from the reference run's FINAL-epoch
clean-test per-class accuracy, and then scores every checkpoint on that same clean test set.
Selection and assessment therefore share a sample. If the bottom-30% set is partly noise,
the WC risk it produces is optimistically shaped by the very fluctuations that chose it, and
the ID/WC divergence the audit reports could be an artifact of that reuse rather than a fact
about the runs.

This splits the two roles apart:

* **proposal half** — re-derive the tail class set here, from the reference run's final-epoch
  forward logits restricted to these indices;
* **assessment half** — compute the WC risk curve, its oracle and IQR, and then rho*_LE and
  the taxonomy at delta = 0.10, on the disjoint half.

Three arms are reported, because two of them are needed to attribute any change:

1. ``registered`` — registered class set, full sample. Reproduces the battery; it is the
   baseline and a check that this script's risk path matches the adjudicated one.
2. ``refit`` — classes proposed on the proposal half, scored on the assessment half. This is
   T10's primary.
3. ``control`` — the REGISTERED class set, scored on the assessment half. Without this arm a
   taxonomy change in (2) cannot be attributed: halving the evaluation sample moves the IQR
   denominator and the oracle on its own, quite apart from where the classes came from. The
   control holds the class set fixed and changes only the sample, so (3) vs (1) isolates the
   sample-size effect and (2) vs (3) isolates the selection effect.

**The refit path is exact.** The registered tail comes from the training run's logged
``per_class_test_error``; the refit necessarily recomputes per-class accuracy from the
forward logits, which is a different code path. Verified across all 51 paired runs before
this was written: on the full sample the two agree to ``max|diff| = 0.000e+00`` and select
identical tail sets, so the only thing separating the refit from the registered set is the
index subset -- which is the thing under test.

**What is held fixed.** OOD pools are NOT split. The question is about clean-test class
selection, and resampling the pools would move the OOD axis for an unrelated reason; only ID
indices are restricted. Selector grid indices are read from the adjudicated battery rather
than recomputed: E(tau=1), NA and ER-argmax are all functions of TRAIN logits and features,
so a clean-test split cannot move them. Their J values do change, because J is evaluated on
whatever sample the axes are built from.

**K-fold arm.** The registration allows a K-fold cross-fit "if class sizes forbid a single
split". CIFAR-10 has 1000 test images per class, so a half is 500 and a single split is
comfortable. CIFAR-100 has 100 per class, so a half is 50 and per-class accuracy moves in
steps of 2% -- thin enough that a single split could pick a tail out of quantisation noise.
Rather than decide that by assertion, both are run: the single split as the registered
primary, and a 5-fold cross-fit where each fold is assessed with classes proposed on its
complement. Disagreement between them is itself the answer about whether 50 per class is
enough.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.corrected import (CKPT_GRID, DELTA_PRIMARY, INDETERMINATE_BAND,   # noqa: E402
                                REGISTERED_SELECTORS, RunFrame, axis_frame, classify, iqr)
from analysis.io import load_run, static_tail_from_reference                    # noqa: E402
from eval.ood import auroc, energy_score                                        # noqa: E402
from eval.tail import per_class_error, static_tail_classes                      # noqa: E402
from provenance import code_stamp                                               # noqa: E402

N_T = len(CKPT_GRID)
SEMANTIC = ("svhn", "cross_cifar")
TAIL_FRAC = 0.30
RNG_SEED = 20260814
K_FOLD = 5
FRAMES = {"tier1": ("results/forward_ext", "results/runs_ext", "results/corrected/battery_tier1.json"),
          "g2": ("results/forward_b2", "results/runs", "results/corrected/battery_g2.json")}
OUT = os.path.join(ROOT, "results", "corrected", "t10_wc_refit.json")


class T10Run:
    """Only what the WC question needs: clean-test correctness and the OOD energy scores."""

    def __init__(self, run_id: str, fdir: str, meta: dict, energy_T: float) -> None:
        self.run_id, self.meta = run_id, meta
        self.n_classes = int(meta["n_classes"])
        lab = np.load(os.path.join(fdir, "labels.npz"))
        self.y = np.asarray(lab["test"], np.int64)
        eps = sorted(int(d[2:]) for d in os.listdir(fdir) if d.startswith("ep"))
        assert eps == list(CKPT_GRID), f"{run_id}: grid mismatch"
        self.correct = np.zeros((N_T, self.y.size), bool)
        self.e_test = np.zeros((N_T, self.y.size))
        self.e_pool: Dict[str, np.ndarray] = {}
        for i, ep in enumerate(eps):
            d = os.path.join(fdir, f"ep{ep:03d}")
            lt = np.load(os.path.join(d, "logits_test.npy"))
            self.correct[i] = lt.argmax(1) == self.y
            self.e_test[i] = energy_score(lt, energy_T)
            for p in SEMANTIC:
                lp = np.load(os.path.join(d, f"logits_{p}.npy"))
                self.e_pool.setdefault(p, np.zeros((N_T, len(lp))))[i] = energy_score(lp, energy_T)

    # ---- risks on an index subset -------------------------------------------------
    def r_id(self, idx: Optional[np.ndarray]) -> np.ndarray:
        c = self.correct if idx is None else self.correct[:, idx]
        return 1.0 - c.mean(axis=1)

    def pce(self, i: int, idx: Optional[np.ndarray]) -> np.ndarray:
        y = self.y if idx is None else self.y[idx]
        c = self.correct[i] if idx is None else self.correct[i][idx]
        corr, tot = np.zeros(self.n_classes), np.zeros(self.n_classes)
        np.add.at(tot, y, 1.0)
        np.add.at(corr, y[c], 1.0)
        return per_class_error(corr, tot)

    def r_wc(self, idx: Optional[np.ndarray], classes: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.nanmean(self.pce(i, idx)[classes])) for i in range(N_T)])

    def r_ood(self, idx: Optional[np.ndarray]) -> np.ndarray:
        cols = []
        for p in SEMANTIC:
            a = self.e_test if idx is None else self.e_test[:, idx]
            b = self.e_pool[p]
            cols.append([1.0 - auroc(a[i], b[i]) for i in range(N_T)])
        return np.mean(np.vstack(cols), axis=0)

    def final_tail(self, idx: Optional[np.ndarray]) -> np.ndarray:
        """Bottom-frac classes by FINAL-epoch accuracy, on the given index subset."""
        return static_tail_classes(1.0 - self.pce(N_T - 1, idx), TAIL_FRAC)


def stratified_halves(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(idx)
        h = idx.size // 2
        a.append(perm[:h]); b.append(perm[h:])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def stratified_folds(y: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: List[List[np.ndarray]] = [[] for _ in range(k)]
    for c in np.unique(y):
        perm = rng.permutation(np.flatnonzero(y == c))
        for j, part in enumerate(np.array_split(perm, k)):
            folds[j].append(part)
    return [np.sort(np.concatenate(f)) for f in folds]


def arm(run: T10Run, idx: Optional[np.ndarray], classes: np.ndarray,
        sel: Dict[str, int], delta: float) -> dict:
    axes = {"ID": axis_frame("ID", run.r_id(idx)),
            "WC": axis_frame("WC", run.r_wc(idx, classes)),
            "OOD": axis_frame("OOD", run.r_ood(idx))}
    excluded = [a for a, f in axes.items() if f.excluded]
    fr = RunFrame(run_id=run.run_id, axes=axes, excluded_axes=excluded)
    sel_J = {s: fr.J(i) for s, i in sel.items() if s in REGISTERED_SELECTORS}
    rho = fr.rho_star_le()
    return dict(rho_star_LE=rho, taxonomy=classify(rho, sel_J, delta),
                w_delta=fr.w_delta(delta),
                F_delta=[int(CKPT_GRID[i]) for i in fr.feasible_set(delta)],
                wc_oracle_epoch=int(CKPT_GRID[axes["WC"].t_star]),
                wc_R_star=axes["WC"].R_star, wc_d=axes["WC"].d,
                wc_iqr=float(iqr(axes["WC"].risk)),
                selector_J={s: sel_J[s] for s in sel_J},
                tail_classes=[int(c) for c in classes])


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    sa, sb = set(a.tolist()), set(b.tolist())
    inter = len(sa & sb)
    return dict(k=len(sa), overlap=inter, fraction=inter / max(len(sa), 1),
                jaccard=inter / max(len(sa | sb), 1),
                added=sorted(sb - sa), dropped=sorted(sa - sb))


def _write(out: dict) -> None:
    """Write via temp + rename. Called after every frame: the first full run computed all 36
    tier1 rows, then died on a missing g2 reference and wrote nothing, because the only write
    was at the end. Losing finished work to a later, unrelated failure is avoidable."""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, OUT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", nargs="+", default=list(FRAMES))
    p.add_argument("--delta", type=float, default=DELTA_PRIMARY)
    p.add_argument("--limit", type=int, default=0, help="debug: first N runs per frame")
    a = p.parse_args(argv)

    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to run T10 from a dirty tree")
    energy_T = float(yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))
                     ["eval"]["ood_energy_T"])

    out: Dict[str, object] = dict(
        estimand="WC class selection vs assessment on the same clean test sample",
        delta=a.delta, tail_frac=TAIL_FRAC, seed=RNG_SEED, k_fold=K_FOLD,
        pools_held_fixed=True, selectors_from="adjudicated battery grid indices",
        code_stamp=stamp, frames={})

    for fname in a.frames:
        froot, rroot, bpath = FRAMES[fname]
        froot, rroot, bpath = (os.path.join(ROOT, x) for x in (froot, rroot, bpath))
        battery = {r["run_id"]: r for r in json.load(open(bpath))["per_run"]}
        run_ids = sorted(d for d in os.listdir(froot) if os.path.isdir(os.path.join(froot, d)))
        if a.limit:
            run_ids = run_ids[:a.limit]

        cache: Dict[str, T10Run] = {}
        rows = []
        for rid in run_ids:
            meta = load_run(os.path.join(rroot, rid)).meta
            ref_id = (f"{meta['split']}_ce_seed0" if meta.get("split")
                      else f"{meta['dataset']}_{meta['noise_type']}{meta['eta']:g}_ce_seed0")
            for need in (rid, ref_id):
                if need in cache:
                    continue
                if not os.path.exists(os.path.join(froot, need, "labels.npz")):
                    if need == rid:
                        raise SystemExit(f"{rid}: own forward output missing")
                    continue          # reference without a forward pass; handled below
                cache[need] = T10Run(need, os.path.join(froot, need),
                                     load_run(os.path.join(rroot, need)).meta, energy_T)
            run, ref = cache[rid], cache.get(ref_id)
            sel = {s: v["grid_index"] for s, v in battery[rid]["A4"]["selectors"].items()}

            # The registered tail is taken from the reference run's LOGGED per-class error,
            # which is what the adjudicated battery itself uses and which exists for every
            # reference. Verified equal to the forward-recomputed tail on the full sample
            # across all 51 paired runs, so this is the same set, not an approximation --
            # and it means the registered and control arms do not depend on the reference
            # having a stored forward pass.
            registered_classes = static_tail_from_reference(
                load_run(os.path.join(rroot, ref_id)), TAIL_FRAC)
            prop, asmt = stratified_halves(run.y, RNG_SEED)

            row = dict(run_id=rid, reference_run=ref_id,
                       n_test=int(run.y.size), n_classes=run.n_classes,
                       per_class_full=int(np.bincount(run.y).min()),
                       per_class_half=int(np.bincount(run.y[prop]).min()),
                       registered=arm(run, None, registered_classes, sel, a.delta),
                       control=arm(run, asmt, registered_classes, sel, a.delta))

            if ref is None:
                # Refitting needs the reference's PER-SAMPLE predictions to recompute
                # per-class accuracy on an index subset, and those are only in a stored
                # forward pass. Where it is absent the refit is not computable, and that is
                # recorded as such rather than substituted with a different reference run,
                # which would silently change the registered rule.
                reason = (f"reference run {ref_id} has no stored forward output; the refit "
                          f"and cross-fit arms require its per-sample predictions")
                row["refit"] = None
                row["kfold"] = None
                row["class_agreement"] = None
                row["not_computable"] = reason
                rows.append(row)
                print(f"  {rid:34} refit NOT COMPUTABLE — {reason}", flush=True)
                continue

            refit_classes = ref.final_tail(prop)
            row["refit"] = arm(run, asmt, refit_classes, sel, a.delta)
            row["class_agreement"] = agreement(registered_classes, refit_classes)

            folds = stratified_folds(run.y, K_FOLD, RNG_SEED)
            fold_rows = []
            for j, f in enumerate(folds):
                comp = np.sort(np.concatenate([g for i, g in enumerate(folds) if i != j]))
                fold_rows.append(dict(fold=j,
                                      **arm(run, f, ref.final_tail(comp), sel, a.delta),
                                      agreement=agreement(registered_classes,
                                                          ref.final_tail(comp))))
            taxa = [f["taxonomy"] for f in fold_rows]
            row["kfold"] = dict(folds=fold_rows,
                                taxonomy_counts={t: taxa.count(t) for t in set(taxa)},
                                rho_star_min=min(f["rho_star_LE"] for f in fold_rows),
                                rho_star_max=max(f["rho_star_LE"] for f in fold_rows))
            rows.append(row)

            print(f"  {rid:34} registered={row['registered']['taxonomy']:22} "
                  f"refit={row['refit']['taxonomy']:22} control={row['control']['taxonomy']:22} "
                  f"tail_overlap={row['class_agreement']['overlap']}/{row['class_agreement']['k']}",
                  flush=True)
            if rid != ref_id:
                cache.pop(rid, None)

        out["frames"][fname] = rows  # type: ignore[index]
        _write(out)                    # checkpoint: the g2 crash discarded a finished tier1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"[t10] wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
