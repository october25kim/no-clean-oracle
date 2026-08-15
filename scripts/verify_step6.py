"""Mechanical recount of every figure in the step-6 adjudication record.

Verify-then-commit, per the classification_record precedent: the record is committed only
if this recomputes each of its numbers from the two battery artifacts and the unsealed
counts file. Any mismatch stops the commit.

Expected values are transcribed here from the record as literals. That is the weakness
R3-note names, so each check also carries the record's own wording in its label — a
transcription slip then shows up as a labelled mismatch rather than as a silent pass.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import code_stamp                     # noqa: E402

G2 = os.path.join(ROOT, "results", "corrected", "battery_g2.json")
T1 = os.path.join(ROOT, "results", "corrected", "battery_tier1.json")
COUNTS = os.path.join(ROOT, "results", "sealed_ext", "ext_counts.json")

checks: List[Dict[str, Any]] = []


def check(label: str, got: Any, want: Any) -> None:
    if isinstance(got, float) and isinstance(want, float):
        ok = abs(got - want) <= 1e-4
    else:
        ok = got == want
    checks.append(dict(label=label, expected=want, observed=got, pass_=bool(ok)))


def taxonomy(runs, delta: str) -> Dict[str, int]:
    out = {"incompatible": 0, "compatible-solved": 0,
           "compatible-unsolved": 0, "indeterminate": 0}
    for r in runs:
        out[r["A2"]["per_delta"][f"delta_{delta}"]["taxonomy"]] += 1
    return out


def main() -> int:
    g2 = json.load(open(G2)); t1 = json.load(open(T1))
    G, T = g2["per_run"], t1["per_run"]
    counts = json.load(open(COUNTS))

    # --- 1. corrected taxonomy, all three deltas, both frames ---
    check("G2-15 taxonomy delta=0.10 (record: 14 / 0 / 1 / 0)", taxonomy(G, "0.1"),
          {"incompatible": 14, "compatible-solved": 0,
           "compatible-unsolved": 1, "indeterminate": 0})
    check("G2-15 taxonomy delta=0.05", taxonomy(G, "0.05"),
          {"incompatible": 14, "compatible-solved": 0,
           "compatible-unsolved": 1, "indeterminate": 0})
    check("G2-15 taxonomy delta=0.20", taxonomy(G, "0.2"),
          {"incompatible": 10, "compatible-solved": 0,
           "compatible-unsolved": 3, "indeterminate": 2})
    check("Tier1-36 taxonomy delta=0.10 (record: 27 / 3 / 4 / 2)", taxonomy(T, "0.1"),
          {"incompatible": 27, "compatible-solved": 3,
           "compatible-unsolved": 4, "indeterminate": 2})
    check("Tier1-36 taxonomy delta=0.05", taxonomy(T, "0.05"),
          {"incompatible": 29, "compatible-solved": 0,
           "compatible-unsolved": 2, "indeterminate": 5})
    check("Tier1-36 taxonomy delta=0.20 (record cites compatible-solved 8)",
          taxonomy(T, "0.2"),
          {"incompatible": 19, "compatible-solved": 8,
           "compatible-unsolved": 4, "indeterminate": 5})

    # --- 2. selector successes on compatible runs only ---
    def compat(runs):
        return [r for r in runs
                if r["A2"]["per_delta"]["delta_0.1"]["taxonomy"].startswith("compatible")]

    def succ(runs, sel):
        return sum(1 for r in runs
                   if sel in r["A4"]["selectors"]
                   and r["A4"]["selectors"][sel]["J_LE"] <= 0.10)

    cT, cG = compat(T), compat(G)
    check("Tier1 compatible-run count (record: 7)", len(cT), 7)
    check("G2 compatible-run count (record: 1)", len(cG), 1)
    check("Tier1 E(tau=1) successes on compatible (record: 1/7)", succ(cT, "E_tau1"), 1)
    check("Tier1 NA successes on compatible (record: 1/7)", succ(cT, "NA"), 1)
    check("Tier1 ER-argmax successes on compatible (record: 2/7)", succ(cT, "ER_argmax"), 2)
    check("Tier1 LW-N successes on compatible (record: 1/7)",
          succ(cT, "LW_N_reported_not_gating"), 1)
    for s, lbl in (("E_tau1", "E"), ("NA", "NA"), ("ER_argmax", "ER"),
                   ("LW_N_reported_not_gating", "LW-N")):
        check(f"G2 {lbl} successes on compatible (record: 0/1)", succ(cG, s), 0)

    # --- 3. cross-frame compatible totals cited in record section 2 ---
    check("compatible runs across frames (record: 8)", len(cT) + len(cG), 8)
    check("compatible-solved across frames (record: 3)",
          taxonomy(T, "0.1")["compatible-solved"] + taxonomy(G, "0.1")["compatible-solved"], 3)
    only = cG[0] if cG else None
    check("G2 compatible run rho*_LE (record: 0.0000)",
          round(float(only["A2"]["rho_star_LE"]), 4) if only else None, 0.0)
    check("G2 compatible run w_delta (record: 1/24 = 0.042)",
          round(float(only["A2"]["per_delta"]["delta_0.1"]["w_delta"]), 4) if only else None,
          round(1 / 24, 4))

    # --- 4. frozen branch derivations ---
    joint = {}
    for s in ("E_tau1", "NA", "ER_argmax"):
        joint[s] = sum(1 for r in T if r["A4"]["selectors"][s]["J_LE"] <= 0.10)
    check("Tier1 max joint reliability over registered selectors (record: 2/36)",
          max(joint.values()), 2)
    check("R2 rejected: max joint < 29/36 reliability bar", max(joint.values()) < 29, True)
    ood_succ = counts["success_counts"]["E_mean_max_softmax"]["delta_0.1"]["OOD"]
    check("ext_counts per-axis OOD successes exist (R1 derivation, > 0 required)",
          bool(sum(counts["success_counts"][s]["delta_0.1"]["OOD"]
                   for s in counts["success_counts"]) > 0), True)
    check("ext_counts commitment schema", counts["schema"], "ext_counts.v1")

    # --- 5. A14 both tables ---
    a14 = t1["A14"]["results"]
    check("A14 normalized-regret per-run largest axis",
          a14["normalized_regret"]["per_run_largest_axis"], {"ID": 0, "WC": 10, "OOD": 26})
    check("A14 epoch-gap per-run largest axis",
          a14["epoch_gap"]["per_run_largest_axis"], {"ID": 5, "WC": 3, "OOD": 28})
    check("A14 normalized-regret aggregate largest",
          a14["normalized_regret"]["aggregate_largest_axis"], "OOD")
    check("A14 epoch-gap aggregate largest", a14["epoch_gap"]["aggregate_largest_axis"], "OOD")

    # --- 6. certificate, budget, CF, A13, LOSO, D-7 ---
    check("two-world certificate-bearing runs (record: 3/36)",
          sum(1 for r in T if r["A9"]["two_world_certificate"]["certificate_bearing"]), 3)
    check("two-world disjoint pool pairs (record: 7)",
          sum(r["A9"]["two_world_certificate"]["disjoint_pair_count"] for r in T), 7)

    want_n = {"ID->ID": 18, "ID->OOD": 1, "OOD->OOD": 21, "mixed->OOD": 2}
    for cond, w in want_n.items():
        got = sum(1 for r in T if r["A11"].get("curves")
                  and r["A11"]["curves"][cond]["n_star"] is not None)
        check(f"budget n* reached, {cond} (record: {w}/36)", got, w)
    for cond, w in (("ID->OOD", 0.216), ("mixed->OOD", 0.301)):
        med = float(np.median([max(r["A11"]["curves"][cond]["q"].values())
                               for r in T if r["A11"].get("curves")]))
        check(f"budget median max q, {cond} (record: {w})", round(med, 3), w)

    ex = [r["A4"]["selectors"]["E_tau1"]["J_LE"] - r["A6"]["minimax"]
          for r in T if np.isfinite(r["A6"]["minimax"])]
    check("A6 median CF benchmark excess (record: +0.8243)",
          round(float(np.median(ex)), 4), 0.8243)
    check("A6 negative excesses (record: 9/36)", sum(1 for x in ex if x < 0), 9)

    ci = t1["A13"]["ci95"]
    check("A13 BCa lower (record: 0.2422)", round(float(ci[0]), 4), 0.2422)
    check("A13 BCa upper (record: 0.4084)", round(float(ci[1]), 4), 0.4084)
    check("A13 CI lies entirely above delta=0.10", bool(ci[0] > 0.10), True)

    cells = t1["A12"]["cells"]
    check("LOSO stable cells (record: 7/12)",
          sum(1 for v in cells.values() if v["stable"]), 7)
    check("LOSO cell count", len(cells), 12)

    check("D-7 corrected exclusions, G2 (record: 0)",
          sum(len(r["excluded_axes"]) for r in G), 0)
    check("D-7 corrected exclusions, Tier1 (record: 0)",
          sum(len(r["excluded_axes"]) for r in T), 0)

    # --- report ---
    failed = [c for c in checks if not c["pass_"]]
    for c in checks:
        print(f"  [{'PASS' if c['pass_'] else 'FAIL'}] {c['label']}")
        if not c["pass_"]:
            print(f"         expected {c['expected']!r}\n         observed {c['observed']!r}")
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks pass")

    stamp = code_stamp()
    out = dict(schema="step6_verification.v1",
               integrity_class="internal-consistency (recount from committed artifacts; "
                               "exact equality required for counts, 1e-4 for reported "
                               "rounded reals)",
               n_checks=len(checks), n_failed=len(failed),
               all_pass=not failed, checks=checks, code_stamp=stamp,
               inputs=dict(battery_g2=os.path.relpath(G2, ROOT),
                           battery_tier1=os.path.relpath(T1, ROOT),
                           ext_counts=os.path.relpath(COUNTS, ROOT)))
    dest = os.path.join(ROOT, "results", "corrected", "step6_verification.json")
    tmp = f"{dest}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(tmp, dest)
    print(f"  wrote {os.path.relpath(dest, ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
