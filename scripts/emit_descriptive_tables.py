"""T6-T8: descriptive tables from the corrected batteries. [CORRECTED-DESCRIPTIVE]

Post-adjudication descriptive emission. No new estimand is introduced: every quantity is
either read from ``battery_g2.json`` / ``battery_tier1.json`` or is a threshold read of a
stored curve. The one arithmetic addition is the Clopper-Pearson interval in T7, which is
an exact interval for a binomial proportion whose numerator and denominator both already
exist in the record — it summarises counts rather than estimating anything new.

The intervals are wide on purpose and the table says so. With seven compatible runs in
Tier 1 and one in Phase II, an exact interval is the honest way to show how little those
counts constrain: reporting 1/7 without an interval invites a reader to treat it as a rate.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
from scipy.stats import beta as _beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import code_stamp                       # noqa: E402

TAB = os.path.join(ROOT, "docs", "tables")
COR = os.path.join(ROOT, "results", "corrected")
GRID = [e for e in range(120) if (e + 1) % 5 == 0]
AXES = ["ID", "WC", "OOD"]
REGISTERED = [("E_tau1", "E(τ=1)"), ("NA", "NA"), ("ER_argmax", "ER-argmax")]
DELTA = 0.10
TAG = "CORRECTED-DESCRIPTIVE"


def load(frame: str) -> List[dict]:
    return json.load(open(os.path.join(COR, f"battery_{frame}.json")))["per_run"]


def feasible(r: dict, axis: str, d: float = DELTA) -> List[int]:
    g = r["A1"]["axes"][axis]["ghat"]
    return [e for e, v in zip(GRID, g) if v <= d]


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial interval. k=0 pins the lower end at 0; k=n pins the upper at 1."""
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def compatible(runs: List[dict]) -> List[dict]:
    return [r for r in runs
            if r["A2"]["per_delta"][f"delta_{DELTA:g}"]["taxonomy"].startswith("compatible")]


def succeeded(r: dict, sel: str) -> bool:
    s = r["A4"]["selectors"].get(sel)
    return bool(s and s["J_LE"] <= DELTA)


def t6(T: List[dict]) -> None:
    cert = [r for r in T if r["A9"]["two_world_certificate"]["certificate_bearing"]]
    L = [f"# T6 — Two-world certificate-bearing runs  [{TAG}]", "",
         "Runs in which at least one pair of OOD (score | pool) variants has **disjoint** "
         f"δ={DELTA:g} feasible sets on the OOD axis, with ID and WC held fixed. Sources: "
         "`per_run[].A9.two_world_certificate.pairs[]` and "
         "`per_run[].A9.per_score_pool.<u|o>.F_delta`; the ID/WC column is a threshold read "
         "of the stored `A1.axes.{ID,WC}.ghat` at δ. Checkpoint epochs, not indices.", ""]
    npairs = 0
    for r in cert:
        dj = [p for p in r["A9"]["two_world_certificate"]["pairs"] if p["disjoint"]]
        npairs += len(dj)
        idwc = sorted(set(feasible(r, "ID")) & set(feasible(r, "WC")))
        L += [f"### `{r['run_id']}` — {len(dj)} disjoint pair(s)", "",
              f"ID ∩ WC feasible set: {idwc if idwc else '∅'}", "",
              "| pair | variant | OOD-axis feasible set |", "|---|---|---|"]
        for i, p in enumerate(dj, 1):
            for v in p["pair"]:
                fs = r["A9"]["per_score_pool"][v]["F_delta"]
                L.append(f"| {i} | `{v}` | {fs if fs else '∅'} |")
        L.append("")
    L += [f"**{len(cert)} certificate-bearing runs, {npairs} disjoint pairs in total.**", "",
          "A disjoint pair means two admissible OOD evaluation choices whose δ-adequate "
          "checkpoint sets do not intersect: no single checkpoint is adequate under both. "
          "Counts only.", ""]
    open(os.path.join(TAB, "T6_certificates.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T6_certificates.md")


def t7(G: List[dict], T: List[dict]) -> None:
    L = [f"# T7 — Selector joint successes on compatible runs, exact intervals  [{TAG}]", "",
         f"Joint success = Ĵ_LE ≤ δ = {DELTA:g} on a run classified compatible at that δ. "
         "Intervals are exact Clopper–Pearson at 95%. They are wide because the "
         "denominators are small; that is the point of showing them, since a bare 1/7 "
         "reads like a rate. Sources: `A2.per_delta.delta_0.1.taxonomy` for eligibility, "
         "`A4.selectors.<s>.J_LE` for success, `A5.per_delta.delta_0.1.p_unif` for the "
         "uniform baseline.", ""]
    for frame, runs in (("Tier 1", T), ("Phase II", G)):
        comp = compatible(runs)
        n = len(comp)
        L += [f"### {frame} — {n} compatible run(s)", "",
              "| selector | successes | proportion | exact 95% CI |", "|---|---|---|---|"]
        for key, label in REGISTERED:
            k = sum(1 for r in comp if succeeded(r, key))
            lo, hi = clopper_pearson(k, n)
            L.append(f"| {label} | {k}/{n} | {k / n:.3f} | [{lo:.3f}, {hi:.3f}] |")
        k = sum(1 for r in comp if succeeded(r, "LW_N_reported_not_gating"))
        lo, hi = clopper_pearson(k, n)
        L.append(f"| LW-N *(reported, non-gating)* | {k}/{n} | {k / n:.3f} | "
                 f"[{lo:.3f}, {hi:.3f}] |")
        L += ["", "Per-run uniform baseline on the same runs:", "",
              "| run | w_δ = p_unif | " + " | ".join(l for _k, l in REGISTERED) + " |",
              "|---|---|" + "---|" * len(REGISTERED)]
        for r in sorted(comp, key=lambda x: x["run_id"]):
            w = r["A2"]["per_delta"][f"delta_{DELTA:g}"]["w_delta"]
            marks = " | ".join("hit" if succeeded(r, k) else "—" for k, _l in REGISTERED)
            L.append(f"| `{r['run_id']}` | {w:.3f} | {marks} |")
        L.append("")
    open(os.path.join(TAB, "T7_selector_intervals.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T7_selector_intervals.md")


def t8(G: List[dict], T: List[dict]) -> None:
    L = [f"# T8 — Compatible runs stratified by feasible-window width  [{TAG}]", "",
         f"w_δ at δ = {DELTA:g} is the fraction of the 24 retained checkpoints that are "
         "jointly adequate, so a stratum is a difficulty band: a run with one adequate "
         "checkpoint out of 24 is a harder target than one with several. Sources: "
         "`A2.per_delta.delta_0.1.{w_delta,taxonomy}` and `A4.selectors.<s>.J_LE`.", ""]
    for frame, runs in (("Tier 1", T), ("Phase II", G)):
        comp = compatible(runs)
        strata: Dict[float, List[dict]] = {}
        for r in comp:
            strata.setdefault(round(r["A2"]["per_delta"][f"delta_{DELTA:g}"]["w_delta"], 4),
                              []).append(r)
        L += [f"### {frame} — {len(comp)} compatible run(s), "
              f"{len(strata)} stratum/strata", "",
              "| w_δ | ≈ checkpoints of 24 | runs | " +
              " | ".join(l for _k, l in REGISTERED) + " | LW-N |",
              "|---|---|---|" + "---|" * (len(REGISTERED) + 1)]
        for w in sorted(strata):
            rs = strata[w]
            cells = " | ".join(f"{sum(1 for r in rs if succeeded(r, k))}/{len(rs)}"
                               for k, _l in REGISTERED)
            lw = sum(1 for r in rs if succeeded(r, "LW_N_reported_not_gating"))
            L.append(f"| {w:.3f} | {round(w * 24)} | {len(rs)} | {cells} | "
                     f"{lw}/{len(rs)} |")
        L += ["", "Run membership per stratum:", ""]
        for w in sorted(strata):
            L.append(f"- **w_δ = {w:.3f}** — " +
                     ", ".join(f"`{r['run_id']}`" for r in sorted(strata[w],
                                                                  key=lambda x: x["run_id"])))
        L.append("")
    open(os.path.join(TAB, "T8_success_by_window.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T8_success_by_window.md")


def main() -> int:
    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to emit from an unattested or dirty tree")
    os.makedirs(TAB, exist_ok=True)
    G, T = load("g2"), load("tier1")
    print(f"[tables] Phase II {len(G)} runs, Tier 1 {len(T)} runs")
    t6(T); t7(G, T); t8(G, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
