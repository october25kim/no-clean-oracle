"""A9-ADDENDUM: A3 taxonomy under the five non-primary OOD aggregation variants.

Registered in ``docs/remediation_plan_v2.md`` before this script was written, and executed
only after that registration was anchored. Post-adjudication sensitivity: it cannot alter
the step-6 adjudication, which stands on the primary aggregation, and its output is
published whichever way it comes out.

Everything is derived from stored fields. ID and WC are held fixed and their stored ĝ
curves are used unchanged; only the OOD axis is replaced, by one of the aggregated curves
already in ``A9.aggregation_sensitivity``. Each variant then gets its own corrected 24-grid
oracle and 24-epoch IQR denominator, fail-closed on an exactly-zero IQR, exactly as §1
specifies for the primary. Selector positions come from ``A4.selectors.<s>.grid_index``, so
the taxonomy's solved/unsolved split is decided by the same registered set as everywhere
else.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import code_stamp                                     # noqa: E402
from analysis.corrected import (AXES, CKPT_GRID, DELTAS, REGISTERED_SELECTORS,
                                classify, corrected_oracle, denominator)  # noqa: E402

TAB = os.path.join(ROOT, "docs", "tables", "A-T3_ood_decomposition.md")
VARIANTS = [("energy", "min"), ("energy", "max"),
            ("msp", "mean"), ("msp", "min"), ("msp", "max")]
PLACEHOLDER = "## Aggregation-variant rows — NOT EMITTED"


def variant_taxonomy(r: dict, score: str, agg: str, delta: float):
    """A3 class for one run under one aggregation variant, or None if unscorable."""
    if r["A1"]["axes"]["ID"]["ghat"] is None or r["A1"]["axes"]["WC"]["ghat"] is None:
        return None, None
    y = np.asarray(r["A9"]["aggregation_sensitivity"][score][agg], dtype=np.float64)
    t = corrected_oracle(y)
    d, excluded = denominator(y, 0.0)
    if excluded:
        return None, None                      # fail-closed, exactly as the primary rule
    g = {"ID": np.asarray(r["A1"]["axes"]["ID"]["ghat"], float),
         "WC": np.asarray(r["A1"]["axes"]["WC"]["ghat"], float),
         "OOD": (y - y[t]) / d}
    mx = np.max(np.vstack([g[a] for a in AXES]), axis=0)
    rho = float(np.min(mx))
    sel_J = {s: float(max(g[a][r["A4"]["selectors"][s]["grid_index"]] for a in AXES))
             for s in REGISTERED_SELECTORS if s in r["A4"]["selectors"]}
    return classify(rho, sel_J, delta), rho


def main() -> int:
    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to run with an unattested or dirty tree")
    frames = {}
    for name, f in (("Phase II (G2-15)", "battery_g2.json"),
                    ("Tier 1 (36)", "battery_tier1.json")):
        frames[name] = json.load(open(os.path.join(ROOT, "results", "corrected", f)))["per_run"]

    lines = ["## Aggregation-variant taxonomy  [SENSITIVITY, post-adjudication addendum]",
             "",
             "Registered in `docs/remediation_plan_v2.md` (A9-ADDENDUM) before computation "
             "and executed after that registration was anchored. Each variant replaces the "
             "OOD axis with the named aggregation over the semantic pools "
             "{svhn, cross_cifar}, gives it its own corrected 24-grid oracle and 24-epoch "
             "IQR denominator (fail-closed), holds ID and WC fixed, and re-derives the A3 "
             "taxonomy. **This cannot alter the anchored step-6 adjudication**, which "
             "stands on the primary aggregation.", ""]

    flips: List[str] = []
    for frame, runs in frames.items():
        n = len(runs)
        lines += [f"### {frame}", "",
                  "| aggregation | δ | incompatible | compatible-solved | "
                  "compatible-unsolved | indeterminate | unscorable |",
                  "|---|---|---|---|---|---|---|"]
        for label, score, agg in ([("**energy-mean (PRIMARY)**", "energy", "mean")]
                                  + [(f"{s}-{a}", s, a) for s, a in VARIANTS]):
            for d in DELTAS:
                cnt = {k: 0 for k in ("incompatible", "compatible-solved",
                                      "compatible-unsolved", "indeterminate")}
                uns = 0
                for r in runs:
                    if label.startswith("**"):          # primary: use the stored class
                        cnt[r["A2"]["per_delta"][f"delta_{d:g}"]["taxonomy"]] += 1
                        continue
                    c, _ = variant_taxonomy(r, score, agg, d)
                    if c is None or c == "unscorable":
                        uns += 1
                    else:
                        cnt[c] += 1
                        prim = r["A2"]["per_delta"][f"delta_{d:g}"]["taxonomy"]
                        if c != prim:
                            flips.append(f"| `{r['run_id']}` | {frame} | {score}-{agg} | "
                                         f"{d:g} | {prim} | {c} |")
                lines.append(f"| {label} | {d:g} | {cnt['incompatible']} | "
                             f"{cnt['compatible-solved']} | {cnt['compatible-unsolved']} | "
                             f"{cnt['indeterminate']} | {uns} |")
        lines += ["", f"n = {n} runs.", ""]

    lines += ["### Class flips against the primary aggregation", ""]
    if flips:
        lines += [f"{len(flips)} (run × variant × δ) cells change class relative to the "
                  f"primary energy-mean aggregation. Counts only; no interpretation.", "",
                  "| run | frame | variant | δ | primary class | variant class |",
                  "|---|---|---|---|---|---|"] + flips + [""]
    else:
        lines += ["No run changes class under any variant at any δ. The taxonomy is "
                  "invariant to the aggregation choice on both frames.", ""]
    lines += [f"Produced at `git_head` {stamp['git_head_short']}, `git_tree_dirty` "
              f"{stamp['git_tree_dirty']}.", ""]

    old = open(TAB).read()
    head = old.split(PLACEHOLDER)[0].rstrip()
    open(TAB, "w").write(head + "\n\n" + "\n".join(lines))
    print(f"  updated {os.path.relpath(TAB, ROOT)}")
    print(f"  flips: {len(flips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
