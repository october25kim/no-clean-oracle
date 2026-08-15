"""Mechanically recompute the classification counts from the three selector JSONs.

Independent of the review side: it reads only the committed selector outputs and the
thresholds registered in docs/framing_prereg.md, recomputes every count the adjudication
rests on, and compares them against the expected values supplied on the command line.

Nothing here interprets. It answers one question -- do the numbers reproduce -- and
exits non-zero if any expected value is contradicted.

Selector variants used, and why:
  E   max-softmax is the post-hoc statistic pin (entropy carried as sensitivity). The
      grid T is NOT pinned, so every T is recomputed and reported; the run also reports
      which (statistic, T) combinations reproduce a given expected count, rather than
      assuming one.
  C1  accuracy-selected epoch (the pinned primary); loss-selected carried as secondary.
  b   argmax of effective rank (the pinned binding selection); argmin as sensitivity.
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

from analysis.ncr import OBJECTIVES          # noqa: E402
from provenance import code_stamp            # noqa: E402

DELTAS = [0.05, 0.10, 0.20]
PI_COUNT = 12          # reliable success = >= 12 of 15, per docs/framing_prereg.md
N_RUNS = 15


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def e_variants(e: dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{variant_key: {run_id: {axis: r}}} for every (statistic, T) and the l2norm."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in e["runs"]:
        for stat, by_t in r["regrets"].items():
            for tkey, axes in by_t.items():
                key = f"E:{stat}:{tkey}"
                out.setdefault(key, {})[r["run_id"]] = {
                    j: axes[j]["normalized_regret"] for j in OBJECTIVES}
    return out


def simple_variant(doc: dict, regret_key: str, name: str) -> Dict[str, Dict[str, float]]:
    return {r["run_id"]: {j: r[regret_key][j]["normalized_regret"] for j in OBJECTIVES}
            for r in doc["runs"]}


def success_counts(reg: Dict[str, Dict[str, float]], delta: float) -> Dict[str, int]:
    return {j: int(sum(1 for v in reg.values() if v[j] <= delta)) for j in OBJECTIVES}


def disagreement_count(variants: Dict[str, Dict[str, Dict[str, float]]],
                       run_ids: List[str]) -> dict:
    """Runs where the per-axis best selector is not the same selector on all axes.

    Ties count as NO disagreement, matching the record's stated convention: a run is
    counted as disagreeing only if some selector is the unique minimum on one axis and a
    DIFFERENT selector is the unique minimum on another. If any axis has a tied minimum,
    that run is not counted.
    """
    names = list(variants)
    per_run = {}
    n_dis = 0
    for k in run_ids:
        best, tied = {}, False
        for j in OBJECTIVES:
            vals = [(variants[s][k][j], s) for s in names]
            m = min(v for v, _ in vals)
            winners = [s for v, s in vals if v == m]
            if len(winners) > 1:
                tied = True
            best[j] = winners[0]
        differs = (not tied) and len(set(best.values())) > 1
        n_dis += int(differs)
        per_run[k] = dict(best_per_axis=best, tied=tied, disagrees=bool(differs))
    return dict(count=n_dis, per_run=per_run)


def e_prediction_counts(e: dict, stat: str, tkey: str) -> dict:
    """Late-epoch bias and per-run OOD-largest, under several readings.

    The frozen prediction says E selects late, 'biased past the oracle epochs in the
    same direction on all three objectives, with the largest gap on OOD'. 'Largest gap'
    is reported both by epoch gap and by normalized regret, because the prediction does
    not say which, and 'late' is reported both as all-three-axes and as any-axis.
    """
    late_all = late_any = 0
    ood_largest_gap = ood_largest_regret = 0
    for r in e["runs"]:
        ax = r["regrets"][stat][tkey]
        gaps = {j: ax[j]["epoch_gap"] for j in OBJECTIVES}
        regs = {j: ax[j]["normalized_regret"] for j in OBJECTIVES}
        late_all += int(all(g > 0 for g in gaps.values()))
        late_any += int(any(g > 0 for g in gaps.values()))
        ood_largest_gap += int(gaps["OOD"] == max(gaps.values())
                               and list(gaps.values()).count(max(gaps.values())) == 1)
        ood_largest_regret += int(regs["OOD"] == max(regs.values()))
    return dict(late_all_axes=late_all, late_any_axis=late_any,
                ood_largest_by_epoch_gap=ood_largest_gap,
                ood_largest_by_normalized_regret=ood_largest_regret)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", default=os.path.join(ROOT, "results", "report"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "report",
                                                 "classification_verification.json"))
    p.add_argument("--expect-max-axis-success", type=int, default=None)
    p.add_argument("--expect-ood-success-at-020", type=int, default=None)
    p.add_argument("--expect-disagreement", type=int, default=None)
    p.add_argument("--expect-ood-largest-per-run", type=int, default=None)
    p.add_argument("--expect-late-all-axes", type=int, default=None)
    p.add_argument("--expect-ood-largest-epoch-gap", type=int, default=None,
                   help="the epoch-gap reading of P2, asserted alongside the regret one")
    p.add_argument("--expect-counts", default="",
                   help="per-selector per-axis counts at the binding delta, "
                        "e.g. 'E:1,3,0;C1:2,3,0;b:1,3,0'")
    p.add_argument("--expect-min-ood-regret", type=float, default=None,
                   help="minimum OOD normalized regret over all confirmatory cells")
    p.add_argument("--expect-e-median-epoch", type=float, default=None)
    p.add_argument("--expect-e-runs-at-least", default="",
                   help="'THRESHOLD:COUNT', e.g. '104:13'")
    p.add_argument("--expect-t-robust", default="",
                   help="'MAXSOFTMAX,ENTROPY', e.g. '4,3'")
    p.add_argument("--expect-idtail-successes-all-elr", type=int, default=None,
                   help="1 to assert every ID/tail success at the binding delta is an "
                        "ELR cell, 0 to assert not")
    p.add_argument("--binding-delta", type=float, default=0.10,
                   help="the registered binding delta; the max-success check is scoped "
                        "to it, the other deltas are non-binding sensitivity")
    p.add_argument("--ood-largest-metric", default="normalized_regret",
                   choices=["normalized_regret", "epoch_gap"],
                   help="which reading of 'largest gap on OOD' the expected count uses; "
                        "the frozen prediction does not say, so both are always reported")
    p.add_argument("--e-stat", default="mean_max_softmax")
    p.add_argument("--e-t", default="T1")
    a = p.parse_args(argv)

    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit(f"R2 violation: git_available={stamp.get('git_available')} "
                       f"dirty={stamp.get('git_tree_dirty')} {stamp.get('git_dirty_paths')}")

    e = load(os.path.join(a.report_dir, "E_tgrid.json"))
    c1 = load(os.path.join(a.report_dir, "C1_noisyval.json"))
    b = load(os.path.join(a.report_dir, "B_representation.json"))
    run_ids = sorted(r["run_id"] for r in c1["runs"])
    assert len(run_ids) == N_RUNS, f"{len(run_ids)} runs, expected {N_RUNS}"

    variants = e_variants(e)
    variants["C1:accuracy"] = simple_variant(c1, "regrets", "C1")
    variants["C1:loss(secondary)"] = simple_variant(c1, "secondary_regrets", "C1s")
    variants["b:argmax(binding)"] = simple_variant(b, "regrets", "b")
    variants["b:argmin(sensitivity)"] = simple_variant(b, "sensitivity_regrets", "bs")

    counts = {v: {f"delta={d}": success_counts(variants[v], d) for d in DELTAS}
              for v in variants}

    # the confirmatory triple, per the registered set and the post-hoc statistic pin
    confirmatory = {"E": f"E:{a.e_stat}:{a.e_t}",
                    "C1": "C1:accuracy", "b": "b:argmax(binding)"}
    conf_counts = {k: counts[v] for k, v in confirmatory.items()}

    bd = f"delta={a.binding_delta:g}"
    max_axis_success = max(conf_counts[k][bd][j]
                           for k in confirmatory for j in OBJECTIVES)
    max_axis_success_any_delta = max(conf_counts[k][f"delta={d}"][j]
                                     for k in confirmatory for d in DELTAS
                                     for j in OBJECTIVES)
    ood_at_020 = sum(conf_counts[k]["delta=0.2"]["OOD"] for k in confirmatory)
    reliable = {k: {f"delta={d}": {j: conf_counts[k][f"delta={d}"][j] >= PI_COUNT
                                   for j in OBJECTIVES} for d in DELTAS}
                for k in confirmatory}
    s2 = {f"delta={d}": any(all(reliable[k][f"delta={d}"][j] for j in OBJECTIVES)
                            for k in confirmatory) for d in DELTAS}

    dis = disagreement_count({k: variants[v] for k, v in confirmatory.items()}, run_ids)
    preds = e_prediction_counts(e, a.e_stat, a.e_t)

    # which (statistic, T) combinations would reproduce a given expected count
    e_keys = [k for k in variants if k.startswith("E:")]
    e_scan = {k: {f"delta={d}": success_counts(variants[k], d) for d in DELTAS}
              for k in e_keys}

    checks = []

    def check(name, got, want):
        if want is None:
            checks.append(dict(name=name, got=got, expected=None, status="not supplied"))
        else:
            checks.append(dict(name=name, got=got, expected=want,
                               status="MATCH" if got == want else "MISMATCH"))

    check(f"max per-axis success (confirmatory, binding delta={a.binding_delta:g})",
          max_axis_success, a.expect_max_axis_success)
    check("OOD successes summed over 3 selectors at delta=0.20", ood_at_020,
          a.expect_ood_success_at_020)
    check("S1 axis-disagreement runs", dis["count"], a.expect_disagreement)
    check(f"E per-run OOD-largest (by {a.ood_largest_metric})",
          preds[f"ood_largest_by_{a.ood_largest_metric}"], a.expect_ood_largest_per_run)
    check("E late-epoch bias, all three axes past the oracle",
          preds["late_all_axes"], a.expect_late_all_axes)
    check("E per-run OOD-largest (by epoch_gap, P2 sensitivity)",
          preds["ood_largest_by_epoch_gap"], a.expect_ood_largest_epoch_gap)

    out = dict(
        inputs={n: __import__("hashlib").sha256(
            open(os.path.join(a.report_dir, f"{n}.json"), "rb").read()).hexdigest()
            for n in ("E_tgrid", "C1_noisyval", "B_representation")},
        thresholds=dict(deltas=DELTAS, pi_count=PI_COUNT, n_runs=N_RUNS,
                        registered_in="docs/framing_prereg.md (commit 3366144)"),
        confirmatory_variants=confirmatory,
        success_counts_all_variants=counts,
        reliable_success=reliable, s2_by_delta=s2,
        binding_delta=a.binding_delta,
        max_axis_success_at_binding_delta=max_axis_success,
        max_axis_success_any_delta=max_axis_success_any_delta,
        ood_successes_at_delta_020=ood_at_020,
        s1_disagreement=dis, e_prediction_counts=preds,
        e_variant_scan=e_scan, checks=checks,
        code_stamp=stamp)

    print(f"{'variant':30s} " + "  ".join(f"d={d}: ID/tail/OOD" for d in DELTAS))
    for v in variants:
        cells = "   ".join("/".join(str(counts[v][f'delta={d}'][j]) for j in OBJECTIVES)
                           for d in DELTAS)
        mark = " *" if v in confirmatory.values() else "  "
        print(f"{v:30s}{mark}{cells}")
    print(f"\nmax per-axis success @ binding d={a.binding_delta:g} : {max_axis_success}  "
          f"(reliable needs {PI_COUNT}); any delta: {max_axis_success_any_delta}")
    print(f"OOD successes at delta=0.20 (3 sel) : {ood_at_020} / {3*N_RUNS}")
    print(f"S2 satisfied by delta               : {s2}")
    print(f"S1 axis-disagreement runs           : {dis['count']} / {N_RUNS}")
    print(f"E prediction counts                 : {preds}")
    # --- record-derived checks -------------------------------------------------
    e_eps = [r["selected_epoch_by_T"][a.e_stat][a.e_t] for r in e["runs"]]
    all_ood = [variants[v][k]["OOD"] for v in confirmatory.values() for k in run_ids]
    min_ood = float(min(all_ood))
    idtail_hits = [(sel, k, j) for sel, v in confirmatory.items() for k in run_ids
                   for j in ("ID", "tail") if variants[v][k][j] <= a.binding_delta]
    all_elr = all("_elr_" in k for _, k, _ in idtail_hits) and bool(idtail_hits)

    if a.expect_counts:
        for spec in a.expect_counts.split(";"):
            name, nums = spec.split(":")
            want = [int(x) for x in nums.split(",")]
            got = [conf_counts[name.strip()][bd][j] for j in OBJECTIVES]
            check(f"{name.strip()} per-axis counts at {bd} (ID/tail/OOD)", got, want)
    check("min OOD normalized regret over all confirmatory cells",
          round(min_ood, 3), a.expect_min_ood_regret)
    check(f"E({a.e_t}) median selected epoch",
          float(np.median(e_eps)), a.expect_e_median_epoch)
    if a.expect_e_runs_at_least:
        thr, want = (int(x) for x in a.expect_e_runs_at_least.split(":"))
        check(f"E({a.e_t}) runs selecting epoch >= {thr}",
              int(sum(1 for x in e_eps if x >= thr)), want)
    if a.expect_t_robust:
        want = [int(x) for x in a.expect_t_robust.split(",")]
        check("T-robust counts (max-softmax, entropy)",
              [e["summary"]["mean_max_softmax"]["n_T_robust"],
               e["summary"]["mean_entropy"]["n_T_robust"]], want)
    if a.expect_idtail_successes_all_elr is not None:
        check("every ID/tail success at the binding delta is an ELR cell",
              int(all_elr), a.expect_idtail_successes_all_elr)

    out["record_derived"] = dict(
        e_selected_epochs=e_eps, min_ood_regret=min_ood,
        idtail_success_cells=[f"{s_}:{k}:{j}" for s_, k, j in idtail_hits],
        all_idtail_successes_in_elr=bool(all_elr))

    print("\nchecks:")
    for c in checks:
        print(f"  [{c['status']:>12s}] {c['name']}: got {c['got']}, expected {c['expected']}")

    mismatches = [c for c in checks if c["status"] == "MISMATCH"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(all_supplied_checks_match=not mismatches, **out), fh, indent=2)
    print(f"\n[verify] {a.out}")
    if mismatches:
        print(f"[verify] {len(mismatches)} MISMATCH(es) — STOP and report; do not edit the record")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
