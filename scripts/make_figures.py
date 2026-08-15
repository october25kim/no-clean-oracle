"""Manuscript figures v2 and the data tables. PRESENTATION ONLY.

Nothing here is computed that is not already in ``battery_g2.json`` /
``battery_tier1.json``. The transforms applied are display work: sorting, thresholding a
stored curve to draw a strip, the median/IQR band the brief specifies, and mapping stored
epochs onto an axis. Where a requested element needs a quantity the artifacts do not hold,
it is omitted and reported rather than filled in by recomputation.

Design rule for v2: one claim per figure, readable in five seconds. Each figure's payload
is named in its caption entry.

Run selection is deterministic wherever a single run is shown — the run whose
rho-hat*_LE is the median of its eligible group, ties by run id — and the rule and the
resulting id are both written into ``figures/captions.md``.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import code_stamp                      # noqa: E402

FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "docs", "tables")
GRID = [e for e in range(120) if (e + 1) % 5 == 0]
DELTAS = [0.05, 0.10, 0.20]
AXES = ["ID", "WC", "OOD"]

OI = {"ID": "#0072B2", "WC": "#E69F00", "OOD": "#009E73",
      "incompatible": "#D55E00", "compatible-solved": "#009E73",
      "compatible-unsolved": "#0072B2", "indeterminate": "#999999",
      "accent": "#CC79A7", "grey": "#666666", "empty": "#D55E00"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
W1, W2 = 3.4, 7.0

captions: List[str] = []


def save(fig, name: str, caption: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300)
    plt.close(fig)
    captions.append(caption)
    print(f"  wrote figures/{name}.pdf + .png")


def median_run(runs: List[dict]) -> dict:
    s = sorted(runs, key=lambda r: (r["A2"]["rho_star_LE"], r["run_id"]))
    return s[(len(s) - 1) // 2]


def ghat(r: dict, a: str) -> np.ndarray:
    return np.asarray(r["A1"]["axes"][a]["ghat"], dtype=float)


def feasible(r: dict, a: str, d: float = 0.10) -> List[int]:
    g = ghat(r, a)
    return [e for e, v in zip(GRID, g) if v <= d]


def strip(ax, y, epochs, color, empty_note=None, w=4.2):
    """One feasibility strip drawn in EPOCH coordinates, so it aligns with a curve panel.

    Index coordinates were used first and did not survive sharex: the strips collapsed
    into a narrow band under a 120-wide curve axis. Row labels go on the y axis rather
    than as free text, which keeps them outside the data area at any width.
    """
    for e in GRID:
        on = e in epochs
        ax.add_patch(Rectangle((e - w / 2, y - 0.36), w, 0.72,
                               fc=color if on else "#EDEDED", ec="white", lw=0.4))
    if empty_note and not epochs:
        ax.text(np.mean(GRID), y, empty_note, ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold",
                bbox=dict(fc=OI["empty"], ec="none", pad=1.8))


# ------------------------------------------------------------------ F1 (redesigned)

def fig_F1(T: List[dict]) -> None:
    inc = [r for r in T if r["A2"]["per_delta"]["delta_0.1"]["taxonomy"] == "incompatible"]
    r = median_run(inc)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W1, 3.3), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.5, 1.25], hspace=0.14))
    for a in AXES:
        g = ghat(r, a)
        ax.plot(GRID, g, color=OI[a], lw=1.3, label=a, zorder=3)
        t = r["A1"]["axes"][a]["t_star_grid_index"]
        ax.plot(GRID[t], g[t], "o", color=OI[a], ms=4.5, mec="white", mew=0.7, zorder=4)
        ax.vlines(GRID[t], 0, g[t], color=OI[a], lw=0.7, ls=":", zorder=2)
    ax.axhline(0.10, color=OI["grey"], lw=0.9)
    ax.text(122, 0.10, "δ=0.10", fontsize=6.5, color=OI["grey"], va="bottom", ha="right")
    ax.set_ylabel("ĝ$_a$(t)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", frameon=False, ncol=3, columnspacing=1.2,
              handlelength=1.4)
    ax.set_title("Each axis has its own best epoch — and no epoch serves all three",
                 loc="left", fontsize=7.5)

    for i, a in enumerate(AXES):
        strip(bx, -i, feasible(r, a), OI[a])
    joint = r["A2"]["per_delta"]["delta_0.1"]["F_delta"]
    strip(bx, -3.4, joint, OI["compatible-solved"],
          empty_note="EMPTY — no jointly adequate checkpoint")
    bx.set_ylim(-4.1, 0.6)
    bx.set_yticks([0, -1, -2, -3.4])
    bx.set_yticklabels(["ID ≤ δ", "WC ≤ δ", "OOD ≤ δ", "joint"], fontsize=7)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("retained checkpoint (epoch)")
    for sp in bx.spines.values():
        sp.set_visible(False)
    ax.set_xlim(GRID[0] - 6, GRID[-1] + 6)
    save(fig, "mechanism_windows",
         f"**F1 `mechanism_windows.pdf` [MAIN]** — payload: *the joint feasible set is "
         f"empty*. Tier-1 run `{r['run_id']}`, chosen by the deterministic rule (median "
         f"ρ̂*_LE among the {len(inc)} runs classified incompatible at δ=0.10, ties by run "
         f"id); ρ̂*_LE = {r['A2']['rho_star_LE']:.4f}. Top panel: the three ĝ_a(t) curves "
         f"with per-axis argmins marked and dropped to the axis. Bottom panel: each axis's "
         f"feasible set {{t : ĝ_a(t) ≤ 0.10}} as a strip, and their intersection as a "
         f"fourth strip, which is empty. Sources: `A1.axes.<a>.ghat` and "
         f"`.t_star_grid_index`; the three per-axis strips are a threshold read of those "
         f"stored curves at 0.10; the joint strip is "
         f"`A2.per_delta.delta_0.1.F_delta` as stored.")


# ------------------------------------------------------------------ F3

def fig_F3(G: List[dict], T: List[dict]) -> None:
    fig, ax = plt.subplots(figsize=(W2, 3.1))
    x0 = 0
    counts = {}
    for frame, runs in (("Phase II (15)", G), ("Tier 1 (36)", T)):
        s = sorted(runs, key=lambda r: r["A2"]["rho_star_LE"])
        xs = np.arange(len(s)) + x0
        for x, r in zip(xs, s):
            cls = r["A2"]["per_delta"]["delta_0.1"]["taxonomy"]
            ax.plot(x, max(r["A2"]["rho_star_LE"], 1e-3), "o", ms=4.5,
                    color=OI.get(cls, OI["grey"]), mec="white", mew=0.5, zorder=3)
            if r["A2"]["rho_star_LE"] == 0.0:
                ax.annotate(r["run_id"], (x, 1e-3), textcoords="offset points",
                            xytext=(5, 12), fontsize=5.5, color=OI["grey"],
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["grey"]))
        ax.text(xs.mean(), 3.2, frame, ha="center", fontsize=8)
        counts[frame] = {f"{d:g}": sum(
            1 for r in runs
            if r["A2"]["per_delta"][f"delta_{d:g}"]["taxonomy"] == "incompatible")
            for d in DELTAS}
        x0 += len(s) + 3
    ax.axvline(len(G) + 1, color=OI["grey"], lw=0.6)
    for d, ls in zip(DELTAS, [":", "-", "--"]):
        ax.axhline(d, color="black", lw=0.9, ls=ls, zorder=1)
        ax.text(-1.2, d, f"δ={d:g}", fontsize=6.5, ha="right", va="center")
    tier = counts["Tier 1 (36)"]
    ax.text(0.995, 0.02,
            "Tier-1 runs above each line:  "
            + "   ".join(f"δ={d:g} → {tier[f'{d:g}']}/36" for d in DELTAS),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            bbox=dict(fc="white", ec=OI["grey"], lw=0.5, pad=2.5))
    ax.set_yscale("log"); ax.set_ylim(8e-4, 4.0); ax.set_xlim(-7, x0 - 2)
    ax.set_xticks([])
    ax.set_xlabel("runs, sorted by ρ̂*_LE within frame")
    ax.set_ylabel("ρ̂*_LE   (log scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=OI[c], label=c)
               for c in ("incompatible", "compatible-solved", "compatible-unsolved",
                         "indeterminate")]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2)
    save(fig, "rho_dotplot",
         "**F3 `rho_dotplot.pdf` [MAIN]** — payload: *most runs sit well above δ, not "
         "marginally above it*. All 51 audited runs, frames separated, sorted ascending "
         "within frame, coloured by δ=0.10 class. All three δ lines are drawn and the "
         "Tier-1 counts above each are boxed, so the sensitivity shift (29 → 27 → 19 "
         "incompatible at δ = 0.05 / 0.10 / 0.20) reads off one figure. **Log y**: ρ̂*_LE "
         "spans about three orders of magnitude and a linear axis collapses the sub-δ runs "
         "onto the baseline. The two runs at ρ̂*_LE = 0.0000 — "
         "`cifar100_symmetric0.6_ce_seed2` (Phase II) and `c100n_elr_seed2` (Tier 1) — are "
         "drawn at the axis floor and named, since zero has no position on a log scale. "
         "Sources: `A2.rho_star_LE`, `A2.per_delta.delta_<d>.taxonomy`, both battery files.")


# ------------------------------------------------------------------ F4

def _budget_panel(ax, T, key, title, annotate=None):
    ns = [50, 100, 200, 500, 1000, 2000]
    M = np.array([[r["A11"]["curves"][key]["q"][str(n)] for n in ns]
                  for r in T if r["A11"].get("curves")])
    med = np.median(M, axis=0)
    ax.fill_between(ns, np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0),
                    color=OI["ID"], alpha=0.18, lw=0)
    ax.plot(ns, med, "-o", color=OI["ID"], lw=1.4, ms=3.5)
    ax.axhline(0.9, color=OI["incompatible"], lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_xlim(45, 2400); ax.set_ylim(0, 1.02)
    ax.set_title(title, loc="left")
    if annotate:
        ax.annotate(annotate, (0.96, 0.5), xycoords="axes fraction", ha="right",
                    fontsize=7, color=OI["incompatible"], fontweight="bold",
                    bbox=dict(fc="white", ec=OI["incompatible"], lw=0.6, pad=2.5))
    return med


def fig_F4(T: List[dict]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.0), sharex=True, sharey=True)
    spec = [("ID->ID", "ID → ID", None), ("OOD->OOD", "OOD → OOD", None),
            ("ID->OOD", "ID → OOD", "median max q = 0.216"),
            ("mixed->OOD", "mixed → OOD", "median max q = 0.301")]
    for ax, (k, t, ann) in zip(axs.ravel(), spec):
        _budget_panel(ax, T, k, t, ann)
    axs[0, 0].text(52, 0.93, "q = 0.9 target", fontsize=6.5, color=OI["incompatible"])
    for ax in axs[1]:
        ax.set_xlabel("clean labels n  (log)")
    for ax in axs[:, 0]:
        ax.set_ylabel("q(n)")
    save(fig, "budget_curves",
         "**F4 `budget_curves.pdf` [MAIN]** — payload: *the on-axis pairs saturate toward "
         "the target; the OOD-target-from-ID pairs hit a ceiling far below it*. Median "
         "q(n) across the 36 Tier-1 runs with an IQR band, n on a log axis, q = 0.9 "
         "dashed. The two off-diagonal panels carry the verified figures inline. Note the "
         "annotated statistic is the **median of each run's own maximum q**, which is what "
         "the adjudication record reports; the maximum of the plotted median curve is a "
         "different number (0.144 and 0.270) and is not what is annotated. Source: "
         "`A11.curves.<pair>.q` (six n values, B = 1000, assessment on D_assess only). "
         "Display transform: median and IQR across runs.")


# ------------------------------------------------------------------ F5

def fig_F5(G: List[dict]) -> None:
    r = next(x for x in G if x["run_id"] == "cifar100_symmetric0.6_ce_seed2")
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for a in AXES:
        g = ghat(r, a)
        ax.plot(GRID, g, color=OI[a], lw=1.3, label=a, zorder=3)
    joint = r["A2"]["per_delta"]["delta_0.1"]["F_delta"]
    for e in joint:
        ax.axvspan(e - 2.5, e + 2.5, color=OI["compatible-solved"], alpha=0.18, zorder=1)
        ax.plot(e, 0.0, "*", ms=13, color=OI["compatible-solved"], mec="white", mew=0.6,
                zorder=5, clip_on=False)
    ax.axhline(0.10, color=OI["grey"], lw=0.9)
    ax.text(122, 0.10, "δ=0.10", fontsize=6.5, color=OI["grey"], va="bottom", ha="right")
    pretty = {"E_tau1": "E(τ=1)", "NA": "NA", "ER_argmax": "ER",
              "LW_N_reported_not_gating": "LW-N"}
    top = ax.get_ylim()[1]
    for s, lab in pretty.items():
        if s not in r["A4"]["selectors"]:
            continue
        e = GRID[r["A4"]["selectors"][s]["grid_index"]]
        ax.axvline(e, color=OI["accent"], lw=1.0, alpha=0.85, zorder=2)
        ax.text(e, top, f" {lab}", rotation=90, fontsize=6.5, va="top", ha="left",
                color=OI["accent"])
    ax.annotate("", xy=(joint[0], top * 0.55),
                xytext=(GRID[r["A4"]["selectors"]["E_tau1"]["grid_index"]], top * 0.55),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=OI["grey"]))
    ax.text((joint[0] + GRID[r["A4"]["selectors"]["E_tau1"]["grid_index"]]) / 2,
            top * 0.58, "all selectors miss it", ha="center", fontsize=6.5,
            color=OI["grey"])
    ax.set_xlabel("retained checkpoint (epoch)")
    ax.set_ylabel("ĝ$_a$(t)")
    ax.set_xlim(0, 124); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.1, handlelength=1.4)
    ax.set_title("One checkpoint works — and nothing points to it", loc="left",
                 fontsize=7.5)
    w = r["A2"]["per_delta"]["delta_0.1"]["w_delta"]
    p = r["A5"]["per_delta"]["delta_0.1"]["p_unif"]
    save(fig, "case_study",
         f"**F5 `case_study.pdf` [MAIN]** — payload: *the distance between the star and the "
         f"marker cluster*. Phase-II run `{r['run_id']}`, the one compatible run in that "
         f"frame (ρ̂*_LE = {r['A2']['rho_star_LE']:.4f}). All three ĝ_a(t) bottom out at the "
         f"same epoch, starred; w = 1/24 = {w:.3f} and the uniform-random baseline is "
         f"p_unif = {p:.3f}. Vertical rules mark where E(τ=1), NA, ER-argmax and LW-N each "
         f"selected; none is the starred epoch. Sources: `A1.axes.<a>.ghat`, "
         f"`A2.per_delta.delta_0.1.{{F_delta,w_delta}}`, `A4.selectors.<s>.grid_index`, "
         f"`A5.per_delta.delta_0.1.p_unif`. LW-N is drawn and is non-gating.")


# ------------------------------------------------------------------ F6, F7 (appendix)

def fig_F6(T: List[dict]) -> None:
    cert = [r for r in T if r["A9"]["two_world_certificate"]["certificate_bearing"]]
    r = median_run(cert)
    keys = list(r["A9"]["per_score_pool"].keys())
    dj = [p for p in r["A9"]["two_world_certificate"]["pairs"] if p["disjoint"]]
    hot = set(sum([p["pair"] for p in dj], []))
    fig, ax = plt.subplots(figsize=(W2, 0.34 * (len(keys) + 2) + 0.9))
    rows = [(f"{a} axis (context)", set(feasible(r, a)), False) for a in ("ID", "WC")]
    rows += [(k, set(r["A9"]["per_score_pool"][k]["F_delta"]), k in hot) for k in keys]
    for y, (label, feas, is_hot) in enumerate(rows):
        for i, e in enumerate(GRID):
            on = e in feas
            ax.add_patch(Rectangle((i, -y - 0.4), 0.92, 0.8,
                                   fc=(OI["accent"] if (on and is_hot) else
                                       OI["compatible-solved"] if on else "#EDEDED"),
                                   ec="white", lw=0.4))
        ax.text(-0.6, -y, label, ha="right", va="center", fontsize=7,
                color=OI["accent"] if is_hot else "black",
                fontweight="bold" if is_hot else "normal")
    ax.set_xlim(-9, 24); ax.set_ylim(-len(rows), 1)
    ax.set_xticks(np.arange(24) + 0.46)
    ax.set_xticklabels(GRID, fontsize=5.5, rotation=90)
    ax.set_yticks([]); ax.set_xlabel("retained checkpoint (epoch)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Two pools, disjoint feasible sets — {len(dj)} disjoint pair(s)",
                 loc="left", fontsize=8)
    save(fig, "certificate",
         f"**F6 `certificate.pdf` [APPENDIX]** — payload: *two admissible OOD pools whose "
         f"feasible sets do not overlap*. Tier-1 run `{r['run_id']}`, the median ρ̂*_LE "
         f"among the {len(cert)} certificate-bearing runs. One strip per (score | pool) "
         f"OOD variant showing its δ=0.10 feasible set; keys participating in a disjoint "
         f"pair are highlighted. ID and WC strips above for context. Sources: "
         f"`A9.per_score_pool.<u|o>.F_delta` and "
         f"`A9.two_world_certificate.pairs[].{{pair,disjoint}}`; the two context strips are "
         f"a threshold read of `A1.axes.{{ID,WC}}.ghat` at 0.10.")


def fig_F7(G: List[dict], T: List[dict]) -> None:
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    marks = {"Phase II": "o", "Tier 1": "^"}
    same = {}
    for frame, runs in (("Phase II", G), ("Tier 1", T)):
        for a in ("ID", "OOD"):
            xs = [r["A1"]["axes"][a]["frozen_historical"]["t_star_120"] for r in runs]
            ys = [r["A1"]["axes"][a]["t_star_epoch"] for r in runs]
            ax.scatter(xs, ys, s=16, marker=marks[frame], facecolor="none",
                       edgecolor=OI[a], lw=0.9, label=f"{frame} · {a}")
            same[(frame, a)] = sum(1 for x, y in zip(xs, ys) if x == y)
    ax.plot([0, 119], [0, 119], color=OI["grey"], lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("frozen-historical oracle epoch (120-grid, smoothed)")
    ax.set_ylabel("corrected oracle epoch (24-grid, raw)")
    ax.set_xlim(-4, 124); ax.set_ylim(-4, 124)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.text(0.98, 0.03, "\n".join(
        f"{f} {a}: identical {same[(f, a)]}/{15 if f == 'Phase II' else 36}"
        for f in ("Phase II", "Tier 1") for a in ("ID", "OOD")),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=OI["grey"])
    save(fig, "oracle_shift",
         "**F7 `oracle_shift.pdf` [APPENDIX]** — payload: *the grid and definition change "
         "moves the oracle*. Corrected (24-grid raw argmin) against frozen-historical "
         "(120-grid smoothed argmin), both frames, identity line, identical-epoch counts "
         "annotated. **ID and OOD only**: the WC axis has no frozen-historical "
         "counterpart, because the frozen frame's `tail` axis was R_tail_static over 120 "
         "points — a different object, not a second reading of the same one. Sources: "
         "`A1.axes.<a>.t_star_epoch`, `A1.axes.<a>.frozen_historical.t_star_120`.")


# ------------------------------------------------------------------ GA

def fig_GA(T: List[dict]) -> None:
    fig = plt.figure(figsize=(5.1, 2.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    inc = [r for r in T if r["A2"]["per_delta"]["delta_0.1"]["taxonomy"] == "incompatible"]
    r = median_run(inc)
    ax = fig.add_subplot(gs[0])
    for i, a in enumerate(AXES):
        strip(ax, -i, feasible(r, a), OI[a])
    strip(ax, -3.3, r["A2"]["per_delta"]["delta_0.1"]["F_delta"],
          OI["compatible-solved"], empty_note="EMPTY")
    ax.set_xlim(GRID[0] - 6, GRID[-1] + 6); ax.set_ylim(-4.0, 0.6)
    ax.set_xticks([])
    ax.set_yticks([0, -1, -2, -3.3])
    ax.set_yticklabels(["ID", "WC", "OOD", "joint"], fontsize=6)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("feasible checkpoints", loc="left", fontsize=7)
    bx = fig.add_subplot(gs[1])
    med = _budget_panel(bx, T, "ID->OOD", "clean ID labels → OOD",
                        "ceiling  q ≈ 0.216")
    bx.tick_params(labelsize=6)
    bx.set_xlabel("n (log)", fontsize=6.5); bx.set_ylabel("q(n)", fontsize=6.5)
    fig.text(0.5, -0.09,
             "Most noisy-label trajectories contain no jointly deployable checkpoint — "
             "and clean ID labels cannot buy OOD selection.",
             ha="center", fontsize=6.5)
    save(fig, "graphical_abstract",
         f"**GA `graphical_abstract.pdf`** — 5.1×2.0in (≈13×5cm). Left: the F1 strip panel "
         f"for `{r['run_id']}`, joint row empty. Right: the F4 ID→OOD panel, flat below the "
         f"target. Banner text as supplied. Sources are those of F1 and F4; no additional "
         f"field is read.")


# ------------------------------------------------------------------ tables

def tables(G: List[dict], T: List[dict]) -> List[str]:
    os.makedirs(TAB, exist_ok=True)
    notes = []
    lines = ["# A-T1 — per-run compatibility summary, all 51 audited runs", "",
             "Sources: `per_run[].A2.rho_star_LE`, `per_run[].A2.per_delta.delta_<d>."
             "{w_delta,taxonomy}` from `battery_g2.json` and `battery_tier1.json`. "
             "Class is at δ = 0.10. No value is recomputed.", ""]
    for frame, runs in (("Phase II (G2-15)", G), ("Tier 1 (36)", T)):
        lines += [f"## {frame}", "",
                  "| run | ρ̂*_LE | w(0.05) | w(0.10) | w(0.20) | class (δ=0.10) |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(runs, key=lambda x: x["A2"]["rho_star_LE"]):
            p = r["A2"]["per_delta"]
            lines.append(f"| `{r['run_id']}` | {r['A2']['rho_star_LE']:.4f} | "
                         f"{p['delta_0.05']['w_delta']:.3f} | {p['delta_0.1']['w_delta']:.3f} | "
                         f"{p['delta_0.2']['w_delta']:.3f} | {p['delta_0.1']['taxonomy']} |")
        lines.append("")
    open(os.path.join(TAB, "A-T1_per_run.md"), "w").write("\n".join(lines) + "\n")
    print("  wrote docs/tables/A-T1_per_run.md")

    keys = list(T[0]["A9"]["per_score_pool"].keys())
    l2 = ["# A-T3 — OOD decomposition per (score | pool), Tier 1", "",
          "Sources: `per_run[].A9.per_score_pool.<u|o>.{rho_star_LE,w_delta,"
          "ood_t_star_epoch,ood_zero_iqr_24}` over the 36 Tier-1 runs. Medians across runs "
          "are display aggregation; `runs with ρ̂* > δ` is a threshold count of the stored "
          "per-run values.", "",
          "| score \\| pool | median ρ̂*_LE | median w(0.10) | runs with ρ̂*_LE > 0.10 | "
          "median OOD oracle epoch | runs with zero 24-grid OOD IQR |",
          "|---|---|---|---|---|---|"]
    for k in keys:
        rho = [r["A9"]["per_score_pool"][k]["rho_star_LE"] for r in T]
        w = [r["A9"]["per_score_pool"][k]["w_delta"] for r in T]
        ep = [r["A9"]["per_score_pool"][k]["ood_t_star_epoch"] for r in T]
        z = sum(1 for r in T if r["A9"]["per_score_pool"][k]["ood_zero_iqr_24"])
        l2.append(f"| `{k}` | {np.median(rho):.4f} | {np.median(w):.3f} | "
                  f"{sum(1 for x in rho if x > 0.10)}/36 | {np.median(ep):.0f} | {z}/36 |")
    l2 += ["", "## Aggregation-variant rows — NOT EMITTED", "",
           "The brief asks for one row per aggregation variant (mean / min / max) "
           "confirming taxonomy invariance. Those rows are **not** produced here, and the "
           "reason is the one that removed F8 from the figure set: the aggregated OOD "
           "curves are stored (`A9.aggregation_sensitivity.<u>.{mean,min,max}`, 24 points "
           "each), but their taxonomy is not. Deriving it means giving each aggregated "
           "curve its own corrected oracle and 24-epoch IQR denominator, re-forming "
           "`max_a ĝ_a(t)` against the unchanged ID and WC axes, and thresholding at δ — "
           "a new quantity, not a display transform, so it is reported rather than "
           "computed.", "",
           "One row of that block is available without any new computation and is stated "
           "here for completeness: the **energy score averaged over the semantic pools "
           "{svhn, cross_cifar}** *is* the primary R_OOD by definition (protocol §4), so "
           "its taxonomy is the primary taxonomy already reported — 27 incompatible / "
           "3 compatible-solved / 4 compatible-unsolved / 2 indeterminate at δ = 0.10. "
           "The remaining variants (min, max, and the msp-scored equivalents) need the "
           "derivation above.", ""]
    open(os.path.join(TAB, "A-T3_ood_decomposition.md"), "w").write("\n".join(l2) + "\n")
    print("  wrote docs/tables/A-T3_ood_decomposition.md")
    notes.append("A-T3 aggregation-variant rows withheld (new quantity); energy/mean row "
                 "supplied from the primary taxonomy by definition")
    return notes


def main() -> int:
    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to produce figures from an unattested or dirty tree")
    G = json.load(open(os.path.join(ROOT, "results", "corrected",
                                    "battery_g2.json")))["per_run"]
    T = json.load(open(os.path.join(ROOT, "results", "corrected",
                                    "battery_tier1.json")))["per_run"]
    print(f"[figures v2] Phase II {len(G)} runs, Tier 1 {len(T)} runs")
    fig_F1(T); fig_F3(G, T); fig_F4(T); fig_F5(G)          # main
    fig_F6(T); fig_F7(G, T)                                 # appendix
    fig_GA(T)
    tables(G, T)

    head = [
        "# Figure captions and source fields (v2)",
        "",
        "Four main figures, two appendix figures, one graphical abstract. F2 and F8 were "
        "cut in the design review; their content is carried by text and by "
        "`docs/tables/`.",
        "",
        "Presentation only. Every plotted number is read from "
        "`results/corrected/battery_g2.json` or `battery_tier1.json`; JSON paths are "
        "relative to `per_run[]` unless stated. Where one run is shown the selection rule "
        "is deterministic — the run whose ρ̂*_LE is the median of its eligible group, ties "
        "by run id — so the choice is checkable rather than a matter of trust. Each entry "
        "names the figure's single payload.",
        "",
        f"Produced at `git_head` {stamp['git_head_short']}, `git_tree_dirty` "
        f"{stamp['git_tree_dirty']}.",
        "",
    ]
    with open(os.path.join(FIG, "captions.md"), "w") as fh:
        fh.write("\n".join(head) + "\n\n" + "\n\n".join(captions) + "\n")
    print(f"  wrote figures/captions.md ({len(captions)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
