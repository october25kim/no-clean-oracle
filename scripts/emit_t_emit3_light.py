"""T9, T12, T13, T14 — the transcription-class half of T-EMIT-3. [CORRECTED-DESCRIPTIVE]

Registered in docs/remediation_plan_v2.md before computation and executed after that
registration was anchored. These four introduce no estimand: T9, T13 and T14 read the
battery summaries, T12 reads the 120-point curves already in the run logs. The heavy half
(T10 WC refit, T11 paired bootstrap) needs the per-sample logits and is registered
separately.

T12's one design point is worth stating because it is easy to get wrong: the denominator is
held FIXED on the master 120-grid across every subgrid. Recomputing the IQR on each thinned
grid would change the scale and the candidate set together, and the resulting movement
could not be attributed to either.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import code_stamp                                    # noqa: E402
from analysis.io import load_run, static_tail_from_reference, risk_trajectories  # noqa: E402
from analysis.ncr import iqr                                         # noqa: E402
from analysis.corrected import classify, corrected_oracle, REGISTERED_SELECTORS  # noqa: E402

TAB = os.path.join(ROOT, "docs", "tables")
COR = os.path.join(ROOT, "results", "corrected")
GRID = [e for e in range(120) if (e + 1) % 5 == 0]
AXES = ["ID", "WC", "OOD"]
SELECTORS = [("E_tau1", "E(τ=1)"), ("NA", "NA"), ("ER_argmax", "ER-argmax"),
             ("LW_N_reported_not_gating", "LW-N (non-gating)")]
DELTA = 0.10
TAG = "CORRECTED-DESCRIPTIVE"


def load(frame): return json.load(open(os.path.join(COR, f"battery_{frame}.json")))


def minimax_index(r: dict) -> int:
    g = np.vstack([np.asarray(r["A1"]["axes"][a]["ghat"], float) for a in AXES])
    return int(np.argmin(g.max(axis=0)))


def t9(G, T) -> None:
    L = [f"# T9 — Selector joint regret, inefficiency, and raw-unit cost  [{TAG}]", "",
         "η̂_s = Ĵ_s − ρ̂*_LE, both terms from the LE layer (spec D.1); mixing layers can go "
         "negative and would not be an inefficiency. Raw-unit columns are read at each "
         "run's **minimax checkpoint** — the epoch attaining ρ̂*_LE — and are the untransformed "
         "regrets there: ΔID and ΔWC in percentage points of error, ΔOOD in AUROC points. "
         "The IQR denominators that turn them into ĝ are given alongside so the two scales "
         "can be compared directly. Sources: `A4.selectors.*.{J_LE,eta_LE}`, "
         "`A1.axes.*.{delta_raw,d}`.", ""]
    for frame, runs in (("Tier 1", T), ("Phase II", G)):
        L += [f"## {frame}", "",
              "| run | ρ̂*_LE | " + " | ".join(f"Ĵ {l}" for _k, l in SELECTORS) + " | " +
              " | ".join(f"η̂ {l}" for _k, l in SELECTORS) + " |",
              "|---|---|" + "---|" * (2 * len(SELECTORS))]
        for r in sorted(runs, key=lambda x: x["run_id"]):
            J = " | ".join(f"{r['A4']['selectors'][k]['J_LE']:.3f}"
                           if k in r["A4"]["selectors"] else "—" for k, _l in SELECTORS)
            E = " | ".join(f"{r['A4']['selectors'][k]['eta_LE']:.3f}"
                           if k in r["A4"]["selectors"] else "—" for k, _l in SELECTORS)
            L.append(f"| `{r['run_id']}` | {r['A2']['rho_star_LE']:.4f} | {J} | {E} |")
        L += ["", f"### Raw-unit regrets at the minimax checkpoint — {frame}", "",
              "| run | epoch | ΔID (pp) | ΔWC (pp) | ΔOOD (AUROC pp) | d_ID | d_WC | d_OOD |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sorted(runs, key=lambda x: x["run_id"]):
            i = minimax_index(r)
            raw = {a: r["A1"]["axes"][a]["delta_raw"][i] for a in AXES}
            d = {a: r["A1"]["axes"][a]["d"] for a in AXES}
            L.append(f"| `{r['run_id']}` | {GRID[i]} | {100*raw['ID']:.2f} | "
                     f"{100*raw['WC']:.2f} | {100*raw['OOD']:.2f} | {d['ID']:.4f} | "
                     f"{d['WC']:.4f} | {d['OOD']:.4f} |")
        L.append("")
    open(os.path.join(TAB, "T9_eta_raw.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T9_eta_raw.md")


def t12(G, T) -> None:
    """Grid density: thin the master 120-grid, hold the denominator fixed."""
    L = [f"# T12 — Grid-density sensitivity  [{TAG}]", "",
         "The candidate set is thinned from the master 120-point logging grid by taking "
         "every T-th epoch (T = 1, 2, 5, 10; T = 5 is the retained checkpoint grid). "
         "**The denominator is held FIXED on the master 120-grid throughout**: recomputing "
         "the IQR on each thinned grid would move the scale and the candidate set at once "
         "and no movement could be attributed to either. Oracles are the raw argmin on the "
         "subgrid, earliest tie, and the classification is the registered rule at "
         f"δ = {DELTA:g}. Source: the 120-point curves in each run's `metrics.jsonl`.", ""]
    for frame, runs, root in (("Tier 1", T, "runs_ext"), ("Phase II", G, "runs")):
        rows = {t: Counter() for t in (1, 2, 5, 10)}
        n_scored = 0
        for rec in runs:
            run = load_run(os.path.join(ROOT, "results", root, rec["run_id"]))
            m = run.meta
            ref_id = (f"{m['split']}_ce_seed0" if m.get("split")
                      else f"{m['dataset']}_{m['noise_type']}{m['eta']:g}_ce_seed0")
            wc = static_tail_from_reference(load_run(os.path.join(ROOT, "results", root, ref_id)))
            tr = risk_trajectories(run, wc)
            y = {"ID": np.asarray(tr["ID"], float), "WC": np.asarray(tr["tail"], float),
                 "OOD": np.asarray(tr["OOD"], float)}
            d = {a: iqr(y[a]) for a in AXES}
            if min(d.values()) == 0.0:
                continue
            n_scored += 1
            for T_ in (1, 2, 5, 10):
                cand = np.arange(T_ - 1, 120, T_)
                g = np.vstack([(y[a][cand] - y[a][cand].min()) / d[a] for a in AXES])
                mx = g.max(axis=0)
                rho = float(mx.min())
                selJ = {s: float(max((y[a][cand] - y[a][cand].min())[
                    int(np.argmin(mx))] / d[a] for a in AXES)) for s in REGISTERED_SELECTORS}
                rows[T_][classify(rho, {}, DELTA)] += 1
                rows[T_][f"__rho_{rho:.6f}"] += 0
                rows[T_]["__n"] += 1
                rows[T_]["__F"] += int((mx <= DELTA).sum())
                rows[T_]["__rhosum"] += 0
        L += [f"## {frame} — {n_scored} scorable run(s)", "",
              "| T | candidates | incompatible | compatible-* | indeterminate | mean \\|F_δ\\| |",
              "|---|---|---|---|---|---|"]
        for T_ in (1, 2, 5, 10):
            c = rows[T_]
            comp = c["compatible-solved"] + c["compatible-unsolved"]
            L.append(f"| {T_} | {len(np.arange(T_-1,120,T_))} | {c['incompatible']} | "
                     f"{comp} | {c['indeterminate']} | "
                     f"{c['__F']/max(c['__n'],1):.2f} |")
        L += ["", "compatible-solved/unsolved are pooled here because the selector epochs "
              "are defined on the retained grid only; the split is meaningful at T = 5 and "
              "is reported in the primary tables.", "",
              "**The T = 5 row does not reproduce the primary taxonomy, and should not.** "
              "The primary normalises by the IQR over the 24 retained epochs; this table "
              "holds the denominator on the master 120-grid for every row, so that thinning "
              "the candidate set is the only thing that varies. Two quantities differ "
              "between the T = 5 row and the primary table — the candidate set agrees, the "
              "scale does not.", "",
              "The direction across rows follows from the oracle, not from feasibility "
              "getting harder: a denser candidate set finds a lower R̂*, which raises every "
              "other epoch's regret against it, so more runs fall outside δ. This is the "
              "same grid-induced oracle effect recorded in FACT 2, seen here as a "
              "sensitivity rather than as a gap column.", ""]
    open(os.path.join(TAB, "T12_grid.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T12_grid.md")


def t13(T) -> None:
    r0 = T[0]
    keys = list(r0["A9"]["per_score_pool"].keys())
    L = [f"# T13 — OOD axis specification  [{TAG}]", "",
         "What the OOD axis actually is, end to end. Documentation of what ran; no value "
         "is recomputed.", "",
         "| aspect | specification |", "|---|---|",
         "| base metric | `1 − AUROC`, ID as the positive class |",
         "| AUROC estimator | Mann-Whitney rank identity, ties at 0.5, unit-tested against sklearn |",
         "| direction | higher score ⇒ more in-distribution; risk is `1 − AUROC`, so lower is better |",
         "| scores | `msp` = max softmax; `energy` = `T·logsumexp(logits/T)` (sign flipped so higher ⇒ ID) |",
         "| temperature | `ood_energy_T = 1.0` |",
         "| primary pools | `svhn`, `cross_cifar` (semantic) |",
         "| sensitivity pool | `CIFAR-C-local` (covariate) — logged, never in the primary |",
         "| aggregation | unweighted arithmetic **macro** mean over the two semantic pools — NOT pooled samples, so the pools contribute equally despite differing n |",
         "| per-pool n | svhn 26,032 · cross_cifar 10,000 · CIFAR-C-local 8,000 |",
         "| ID sample | the 10,000-image clean test set, shared with the ID and WC axes |",
         "| oracle | raw argmin on the 24 retained checkpoints, **earliest** index on ties |",
         "| denominator | IQR of the raw risk over those same 24 epochs; exact zero is fail-closed |",
         "| CIFAR-C-local caveat | not official CIFAR-10-C/100-C; imagecorruptions 1.1.2 over opencv-headless 4.10.0.84, severity 3, four corruptions, 2,000 per corruption, seeds 20260811 |",
         "",
         "## Per-(score \\| pool) corrected oracle epochs, Tier 1 (36 runs)", "",
         "| score \\| pool | median oracle epoch | min | max | runs with zero 24-grid IQR |",
         "|---|---|---|---|---|"]
    for k in keys:
        ep = [r["A9"]["per_score_pool"][k]["ood_t_star_epoch"] for r in T]
        z = sum(1 for r in T if r["A9"]["per_score_pool"][k]["ood_zero_iqr_24"])
        L.append(f"| `{k}` | {np.median(ep):.0f} | {min(ep)} | {max(ep)} | {z}/36 |")
    L += ["", "The primary axis is the macro mean of the two semantic `energy` rows; the "
          "other four rows are the sensitivity decomposition reported in A9.", ""]
    open(os.path.join(TAB, "T13_ood_spec.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T13_ood_spec.md")


def t14(Tfull) -> None:
    T = Tfull["per_run"]
    cells = Tfull["A12"]["cells"]
    L = [f"# T14 — Cell-level summary, Tier 1 (12 cells × 3 seeds)  [{TAG}]", "",
         f"Class counts at δ = {DELTA:g} within each (split × learner) cell, the LOSO "
         "stability flag from A12, and the margin of each seed's ρ̂*_LE from δ — negative "
         "means below the threshold. Sources: `A2.per_delta.delta_0.1.taxonomy`, "
         "`A2.rho_star_LE`, `A12.cells.*`.", "",
         "| cell | incompatible | compat-solved | compat-unsolved | indeterminate | "
         "majority | LOSO stable | ρ̂*−δ per seed |", "|---|---|---|---|---|---|---|---|"]
    for cell, v in sorted(cells.items()):
        rs = sorted([r for r in T if f"{r['group']}_{r['learner']}" == cell],
                    key=lambda x: x["seed"])
        c = Counter(r["A2"]["per_delta"][f"delta_{DELTA:g}"]["taxonomy"] for r in rs)
        marg = ", ".join(f"{r['A2']['rho_star_LE'] - DELTA:+.3f}" for r in rs)
        L.append(f"| `{cell}` | {c['incompatible']} | {c['compatible-solved']} | "
                 f"{c['compatible-unsolved']} | {c['indeterminate']} | {v['majority']} | "
                 f"{'yes' if v['stable'] else '**no**'} | {marg} |")
    L += ["", "### Margins by learner and by split", "",
          "| grouping | n runs | median ρ̂*−δ | min | max |", "|---|---|---|---|---|"]
    for key, label in (("learner", "learner"), ("group", "split")):
        for val in sorted({r[key] for r in T}):
            m = [r["A2"]["rho_star_LE"] - DELTA for r in T if r[key] == val]
            L.append(f"| {label} `{val}` | {len(m)} | {np.median(m):+.3f} | "
                     f"{min(m):+.3f} | {max(m):+.3f} |")
    L.append("")
    open(os.path.join(TAB, "T14_cells.md"), "w").write("\n".join(L) + "\n")
    print("  wrote docs/tables/T14_cells.md")


def main() -> int:
    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to emit from an unattested or dirty tree")
    g2, t1 = load("g2"), load("tier1")
    G, T = g2["per_run"], t1["per_run"]
    print(f"[T-EMIT-3 light] Phase II {len(G)}, Tier 1 {len(T)}")
    t9(G, T); t12(G, T); t13(T); t14(t1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
