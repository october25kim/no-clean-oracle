"""Tier 1 Tasks 2-4: oracle epochs, registered selectors, mechanical counts -- SEALED.

Every pinned decision is imported from the G2 report scripts rather than restated, so a
pin cannot drift by transcription: ``T_GRID`` and ``statistics_at`` from
``e_tgrid_report``, ``SPLIT_N``/``SPLIT_SEED``/``draw_split``/``LIMITATION`` from
``c1_noisyval_report``, ``effective_rank`` from ``b_representation_report``. Two things
are ext-specific, both because the extension is its own frame:

* the tail reference run is ``<split>_ce_seed0`` -- protocol section 3 fixes the class set
  from the GROUP's CE seed-0 run, and for real noise the group IS the split, so the
  synthetic ``<dataset>_<noise><eta>_ce_seed0`` formula does not apply;
* selector regret is measured against the GRID-RESTRICTED oracle registered in
  docs/framing_preregistration.md ("Measurement-validity registration", 015b355):
  argmin of the same 3-epoch-smoothed curve with candidates restricted to the checkpoint
  grid, raw risk read at that epoch. The 120-point oracle is retained as the information
  ceiling and its gap is carried in its own unclamped column. The frozen
  ``analysis.selection.regret_at_epoch`` is NOT used here because it scores against the
  unrestricted oracle; it is left untouched for the G1/G2 record it already produced.

SEALING (ruling R6). Outputs are outcome-bearing and are never displayed. Each output F
is written under ``results/sealed_ext/`` with a fresh 32-byte salt beside it, and only
``(path, commitment = sha256(salt || F))`` is printed. Nothing in this script prints a
selected epoch, a regret, a count, or any summary of one -- the per-run progress line
carries the run_id and nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from config import load_config                                   # noqa: E402
from provenance import code_stamp                                # noqa: E402
from analysis.io import load_run, static_tail_from_reference, risk_trajectories  # noqa: E402
from analysis.ncr import OBJECTIVES, iqr, moving_average         # noqa: E402
from e_tgrid_report import T_GRID, UNPINNED, statistics_at       # noqa: E402
from c1_noisyval_report import SPLIT_N, SPLIT_SEED, LIMITATION, draw_split, _ce  # noqa: E402
from b_representation_report import DIRECTION_RATIONALE, effective_rank  # noqa: E402

SEAL_DIR = os.path.join(ROOT, "results", "sealed_ext")
CKPT_GRID = [e for e in range(120) if (e + 1) % 5 == 0]          # 4, 9, ..., 119
DELTAS = [0.05, 0.10, 0.20]
RELIABILITY_BAR = "29/36 at pi = 0.8 (stated, NOT applied here)"


def seal(name: str, obj: dict) -> Dict[str, str]:
    """Write F and a fresh 32-byte salt, return only the path and the commitment.

    commitment = sha256(salt || F) over the exact bytes written. The salt is what makes
    this a commitment rather than an enumerable fingerprint: a counts file has few
    plausible values, so a bare sha256 of its plaintext could be brute-forced back to
    its contents. The salt is written beside F and never printed.
    """
    os.makedirs(SEAL_DIR, exist_ok=True)
    path = os.path.join(SEAL_DIR, name)
    payload = json.dumps(obj, indent=1, sort_keys=True).encode()
    salt = os.urandom(32)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, path)
    with open(f"{path}.salt", "wb") as fh:
        fh.write(salt)
    return dict(path=os.path.relpath(path, ROOT),
                commitment=hashlib.sha256(salt + payload).hexdigest())


def grid_oracles(run, tail_classes) -> Dict[str, dict]:
    """Per-axis oracle on the registered candidate set, plus the 120-point ceiling."""
    risks = risk_trajectories(run, tail_classes)
    out = {}
    for j in OBJECTIVES:
        y = np.asarray(risks[j], dtype=np.float64)
        sm = moving_average(y, 3)
        t_grid = int(CKPT_GRID[int(np.argmin(sm[CKPT_GRID]))])
        t_120 = int(np.argmin(sm))
        scale = iqr(y)
        out[j] = dict(t_star_grid=t_grid, R_star_grid=float(y[t_grid]),
                      t_star_120=t_120, R_star_120=float(y[t_120]),
                      grid_induced_oracle_gap=float(y[t_grid] - y[t_120]),
                      iqr=scale, risk_curve=y.tolist())
    return out


def regret_vs_grid_oracle(oracles: Dict[str, dict], stop_epoch: int) -> Dict[str, dict]:
    """Primary selector endpoint: R_raw[stop] - R_raw[t*_grid], unclamped."""
    out = {}
    for j in OBJECTIVES:
        o = oracles[j]
        y = o["risk_curve"]
        r = float(y[stop_epoch] - o["R_star_grid"])
        out[j] = dict(selected_epoch=int(stop_epoch), risk_at_stop=float(y[stop_epoch]),
                      regret=r,
                      normalized_regret=(r / o["iqr"] if o["iqr"] > 0 else 0.0),
                      epoch_gap=int(stop_epoch - o["t_star_grid"]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "ext_tier1.yaml"))
    p.add_argument("--runs", default=os.path.join(ROOT, "results", "runs_ext"))
    p.add_argument("--forward", default=os.path.join(ROOT, "results", "forward_ext"))
    a = p.parse_args(argv)

    stamp = code_stamp()
    cfg = load_config(a.config)
    ecfg = cfg["eval"]
    run_ids = sorted(d for d in os.listdir(a.forward)
                     if os.path.isdir(os.path.join(a.forward, d))
                     and os.path.isfile(os.path.join(a.forward, d, "labels.npz")))
    print(f"[ext] {len(run_ids)} run(s); candidate set = {len(CKPT_GRID)} checkpoint "
          f"epochs {CKPT_GRID[0]}..{CKPT_GRID[-1]} step 5", flush=True)

    oracle_rows: List[dict] = []
    e_rows: List[dict] = []
    c1_rows: List[dict] = []
    er_rows: List[dict] = []

    for run_id in run_ids:
        run = load_run(os.path.join(a.runs, run_id))
        split = run.meta["split"]
        ref = load_run(os.path.join(a.runs, f"{split}_ce_seed0"))   # protocol section 3
        tail = static_tail_from_reference(ref)
        oracles = grid_oracles(run, tail)
        oracle_rows.append(dict(run_id=run_id, split=split, learner=run.meta["learner"],
                                seed=run.meta["seed"],
                                axes={j: {k: v for k, v in oracles[j].items()
                                          if k != "risk_curve"} for j in OBJECTIVES}))

        fdir = os.path.join(a.forward, run_id)
        eps = sorted(int(os.path.basename(d)[2:])
                     for d in os.listdir(fdir) if d.startswith("ep"))
        labels = np.load(os.path.join(fdir, "labels.npz"))
        noisy = labels["train_noisy"]
        n_classes = int(run.meta["n_classes"])
        idx = draw_split(noisy, n_classes, SPLIT_N, SPLIT_SEED)
        y_split = noisy[idx]

        e_stats = {k: [] for k in ("mean_max_softmax", "mean_entropy")}
        e_by_T = {f"T{T:g}": {k: [] for k in e_stats} for T in T_GRID}
        acc, ce_mean, erank = [], [], []
        for ep in eps:
            epd = os.path.join(fdir, f"ep{ep:03d}")
            lg = np.load(os.path.join(epd, "logits_train.npy"))
            for T in T_GRID:
                s = statistics_at(lg, T)
                for k in e_stats:
                    e_by_T[f"T{T:g}"][k].append(s[k])
            sub = lg[idx]
            acc.append(float((sub.argmax(1) == y_split).mean()))
            ce_mean.append(float(_ce(sub, y_split).mean()))
            erank.append(effective_rank(np.load(os.path.join(epd, "feats_train.npy"))))

        # E: per-T selection, then T-robust / T-sensitive per statistic
        chosen = {}
        labels_T = {}
        for stat in e_stats:
            picks = {}
            for T in T_GRID:
                v = np.asarray(e_by_T[f"T{T:g}"][stat])
                j = int(np.argmax(v)) if stat == "mean_max_softmax" else int(np.argmin(v))
                picks[f"T{T:g}"] = int(eps[j])
            chosen[stat] = picks
            labels_T[stat] = ("T-robust" if len(set(picks.values())) == 1
                              else "T-sensitive")
        e_rows.append(dict(run_id=run_id, split=split, learner=run.meta["learner"],
                           seed=run.meta["seed"], selected_epoch_by_T=chosen,
                           t_label=labels_T,
                           regrets={stat: regret_vs_grid_oracle(oracles, chosen[stat]["T1"])
                                    for stat in e_stats},
                           curves_by_T=e_by_T))

        sel_acc = int(eps[int(np.argmax(acc))])          # earliest-epoch tie-break
        sel_ce = int(eps[int(np.argmin(ce_mean))])
        c1_rows.append(dict(run_id=run_id, split=split, learner=run.meta["learner"],
                            seed=run.meta["seed"], selected_epoch=sel_acc,
                            regrets=regret_vs_grid_oracle(oracles, sel_acc),
                            secondary_selected_epoch_by_loss=sel_ce,
                            secondary_regrets=regret_vs_grid_oracle(oracles, sel_ce),
                            accuracy_curve=acc, ce_curve=ce_mean,
                            sop_observation_frame=(
                                "SOP stored logits are the raw f(x); the sparse term "
                                "u^2 - v^2 is excluded and was asserted per checkpoint "
                                "in the forward pass. SOP is NOT excluded from C1."
                                if run.meta["learner"] == "sop" else None)))

        sel_max = int(eps[int(np.argmax(erank))])
        sel_min = int(eps[int(np.argmin(erank))])
        er_rows.append(dict(run_id=run_id, split=split, learner=run.meta["learner"],
                            seed=run.meta["seed"], selected_epoch=sel_max,
                            regrets=regret_vs_grid_oracle(oracles, sel_max),
                            sensitivity_argmin_epoch=sel_min,
                            sensitivity_regrets=regret_vs_grid_oracle(oracles, sel_min),
                            effective_rank_curve=erank))
        print(f"[ext] {run_id} done", flush=True)

    commitments = {}
    commitments["ext_oracle_epochs"] = seal("ext_oracle_epochs.json", dict(
        schema="ext_oracle_epochs.v1", task="Task 2", reference_frame="ext_tier1",
        candidate_set=dict(kind="checkpoint grid (registered grid-restricted oracle)",
                           epochs=CKPT_GRID, n=len(CKPT_GRID), step=5,
                           registered_in="docs/framing_preregistration.md "
                                         "'Measurement-validity registration' (015b355)"),
        oracle_rule="argmin of moving_average(R_j, 3) restricted to candidate_set; "
                    "R* read on the RAW curve at that epoch; first index on ties",
        ceiling_rule="120-point oracle retained as the information ceiling; "
                     "grid_induced_oracle_gap = R[t*_grid] - R[t*_120], UNCLAMPED, "
                     "sign non-monotone by construction",
        pooling="none — each run is its own frame", objectives=list(OBJECTIVES),
        n_runs=len(oracle_rows), code_stamp=stamp, per_run=oracle_rows))

    commitments["ext_E_tgrid"] = seal("ext_E_tgrid.json", dict(
        schema="ext_E_tgrid.v1", task="Task 3 / selector E", reference_frame="ext_tier1",
        t_grid=T_GRID, primary_statistic="mean max-softmax",
        sensitivity_statistic="mean predictive entropy",
        variant_3_computed=False, unpinned_degrees_of_freedom=UNPINNED,
        regret_reference="grid-restricted oracle t*_grid, unclamped",
        n_runs=len(e_rows), code_stamp=stamp, per_run=e_rows))

    commitments["ext_C1_noisyval"] = seal("ext_C1_noisyval.json", dict(
        schema="ext_C1_noisyval.v1", task="Task 3 / selector C1 (NA)",
        reference_frame="ext_tier1",
        split=dict(n=SPLIT_N, seed=SPLIT_SEED, stratified_by="noisy label",
                   per_class={"cifar10": SPLIT_N // 10, "cifar100": SPLIT_N // 100}),
        statistic="accuracy vs noisy labels, argmax, earliest-epoch tie-break",
        secondary="mean CE vs noisy labels, argmin, non-gating",
        limitation=LIMITATION, sop_excluded=False,
        regret_reference="grid-restricted oracle t*_grid, unclamped",
        n_runs=len(c1_rows), code_stamp=stamp, per_run=c1_rows))

    commitments["ext_B_representation"] = seal("ext_B_representation.json", dict(
        schema="ext_B_representation.v1", task="Task 3 / selector (b) ER",
        reference_frame="ext_tier1",
        statistic="effective rank (Roy & Vetterli) of the penultimate embedding, "
                  "all training samples",
        selection="argmax over the candidate set, earliest-epoch tie-break",
        sensitivity="argmin selection, non-gating",
        direction_rationale=DIRECTION_RATIONALE,
        regret_reference="grid-restricted oracle t*_grid, unclamped",
        n_runs=len(er_rows), code_stamp=stamp, per_run=er_rows))

    # ---- Task 4: mechanical counts. No classification, no branch adjudication. ----
    selectors = {"E_mean_max_softmax": [(r["run_id"], r["regrets"]["mean_max_softmax"])
                                        for r in e_rows],
                 "E_mean_entropy": [(r["run_id"], r["regrets"]["mean_entropy"])
                                    for r in e_rows],
                 "C1_accuracy": [(r["run_id"], r["regrets"]) for r in c1_rows],
                 "C1_secondary_ce": [(r["run_id"], r["secondary_regrets"]) for r in c1_rows],
                 "B_effective_rank": [(r["run_id"], r["regrets"]) for r in er_rows],
                 "B_sensitivity_argmin": [(r["run_id"], r["sensitivity_regrets"])
                                          for r in er_rows]}
    success = {s: {f"delta_{d:g}": {j: sum(1 for _rid, reg in rows
                                           if reg[j]["normalized_regret"] <= d)
                                    for j in OBJECTIVES}
                   for d in DELTAS} for s, rows in selectors.items()}
    disagreement = []
    for rid in [r["run_id"] for r in c1_rows]:
        best = {}
        for j in OBJECTIVES:
            cand = [(reg[j]["normalized_regret"], s)
                    for s, rows in selectors.items()
                    for r2, reg in rows if r2 == rid]
            cand.sort(key=lambda t: (t[0], t[1]))       # G2 tie convention: name order
            best[j] = cand[0][1]
        disagreement.append(dict(run_id=rid, best_by_axis=best,
                                 distinct=len(set(best.values()))))
    e_epochs = [r["selected_epoch_by_T"]["mean_max_softmax"]["T1"] for r in e_rows]
    p90_epoch = CKPT_GRID[int(round(0.9 * (len(CKPT_GRID) - 1)))]
    p2 = {}
    for metric in ("normalized_regret", "epoch_gap"):
        counts = {j: 0 for j in OBJECTIVES}
        for r in e_rows:
            reg = r["regrets"]["mean_max_softmax"]
            counts[max(OBJECTIVES, key=lambda j: reg[j][metric])] += 1
        p2[metric] = counts
    min_ood = min(reg["OOD"]["normalized_regret"]
                  for rows in selectors.values() for _rid, reg in rows)

    commitments["ext_counts"] = seal("ext_counts.json", dict(
        schema="ext_counts.v1", task="Task 4", reference_frame="ext_tier1",
        note="counts only — classification executed at review against "
             "extension_prereg section 6",
        deltas=DELTAS, success_criterion="normalized_regret <= delta",
        reliability_bar=RELIABILITY_BAR, n_runs=len(c1_rows),
        selectors=sorted(selectors), success_counts=success,
        axis_disagreement=disagreement,
        E_prediction_statistics=dict(
            P1=dict(selected_epoch_distribution=sorted(e_epochs),
                    median_selected_epoch=float(np.median(e_epochs)),
                    grid_P90_epoch=p90_epoch,
                    n_at_or_after_grid_P90=sum(1 for x in e_epochs if x >= p90_epoch)),
            P2=p2),
        min_OOD_normalized_regret_over_all_cells=min_ood,
        code_stamp=stamp))

    print("\n[ext] SEALED — commitments only (sha256(salt || F)); "
          "plaintext and salts stay under results/sealed_ext/ (gitignored)")
    for k, v in commitments.items():
        print(f"  {v['path']}\n    schema     {k}.v1\n    commitment {v['commitment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
