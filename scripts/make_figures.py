"""Manuscript figures. PRESENTATION ONLY — every plotted number is read from an artifact.

The rule this script is written to obey: nothing is computed here that is not already in
``battery_g2.json`` / ``battery_tier1.json``. The only transforms applied are ones a reader
would call display work — sorting runs for a dot plot, thresholding a stored curve to shade
a region, taking a median and IQR across runs for a band the instruction asked for, and
mapping stored epochs onto a strip. Where a requested panel needed a quantity that does
not exist in the artifacts, the panel is omitted and the omission is reported rather than
quietly filled in by recomputation.

Run selection is deterministic wherever a single run is shown: the run whose rho-hat*_LE is
the median of its eligible group. That rule is stated in the caption file next to the run
id, so the choice can be checked rather than trusted.

Palette is Okabe-Ito (colorblind-safe). Single-column width is 3.4in with 8pt base type.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from provenance import code_stamp                      # noqa: E402

FIG = os.path.join(ROOT, "figures")
GRID = [e for e in range(120) if (e + 1) % 5 == 0]
DELTAS = [0.05, 0.10, 0.20]
AXES = ["ID", "WC", "OOD"]

OI = {"ID": "#0072B2", "WC": "#E69F00", "OOD": "#009E73",
      "incompatible": "#D55E00", "compatible-solved": "#009E73",
      "compatible-unsolved": "#0072B2", "indeterminate": "#999999",
      "accent": "#CC79A7", "grey": "#666666"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
W1, W2 = 3.4, 7.0            # single- and double-column widths, inches

captions: List[str] = []


def save(fig, name: str, caption: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300)
    plt.close(fig)
    captions.append(caption)
    print(f"  wrote figures/{name}.pdf + .png")


def delta_lines(ax, horizontal=True, labels=True):
    for d, ls in zip(DELTAS, [":", "-", "--"]):
        f = ax.axhline if horizontal else ax.axvline
        f(d, color=OI["grey"], lw=0.7, ls=ls, zorder=0)
        if labels:
            (ax.text(0.995, d, f"δ={d:g}", transform=ax.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=6, color=OI["grey"])
             if horizontal else
             ax.text(d, 0.99, f"δ={d:g}", transform=ax.get_xaxis_transform(),
                     ha="left", va="top", rotation=90, fontsize=6, color=OI["grey"]))


def median_run(runs: List[dict]) -> dict:
    """Deterministic pick: the run whose rho-hat*_LE is the median of the group."""
    s = sorted(runs, key=lambda r: (r["A2"]["rho_star_LE"], r["run_id"]))
    return s[(len(s) - 1) // 2]


def ghat(r: dict, a: str) -> np.ndarray:
    return np.asarray(r["A1"]["axes"][a]["ghat"], dtype=float)


# ---------------------------------------------------------------- F1, F5, F6 shared

def _curves_panel(ax, r, title, mark_selectors=None):
    for a in AXES:
        ax.plot(GRID, ghat(r, a), color=OI[a], lw=1.2, label=f"ĝ_{a}", zorder=3)
        t = r["A1"]["axes"][a]["t_star_grid_index"]
        ax.plot(GRID[t], ghat(r, a)[t], "o", color=OI[a], ms=4, mec="white", mew=0.6,
                zorder=4)
    delta_lines(ax)
    F = r["A2"]["per_delta"]["delta_0.1"]["F_delta"]
    if F:
        for e in F:
            ax.axvspan(e - 2.5, e + 2.5, color=OI["compatible-solved"], alpha=0.18, zorder=1)
    else:
        ax.text(0.5, 0.94, "F$_{0.10}$ = ∅  (no jointly δ-adequate checkpoint)",
                transform=ax.transAxes, ha="center", va="top", fontsize=7,
                color=OI["incompatible"],
                bbox=dict(fc="white", ec=OI["incompatible"], lw=0.6, pad=2))
    if mark_selectors:
        for i, (s, gi) in enumerate(mark_selectors.items()):
            ax.axvline(GRID[gi], color=OI["accent"], lw=0.8, ls="-", alpha=0.8, zorder=2)
            ax.text(GRID[gi], ax.get_ylim()[1], f" {s}", rotation=90, fontsize=6,
                    va="top", ha="left", color=OI["accent"])
    ax.set_xlabel("retained checkpoint (epoch)")
    ax.set_ylabel("normalized regret  ĝ$_a$(t)")
    ax.set_title(title, loc="left")
    ax.set_xlim(0, 123)


def fig_F1(T: List[dict]) -> None:
    inc = [r for r in T if r["A2"]["per_delta"]["delta_0.1"]["taxonomy"] == "incompatible"]
    r = median_run(inc)
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    _curves_panel(ax, r, "Axis optima diverge; no checkpoint satisfies all three")
    ax.legend(loc="upper right", frameon=False, ncol=3, columnspacing=1.0)
    save(fig, "concept_trajectory",
         f"**F1 `concept_trajectory.pdf`** — one Tier-1 incompatible run, "
         f"`{r['run_id']}`. Selection rule: the median ρ̂*_LE among the "
         f"{len(inc)} Tier-1 runs classified incompatible at δ=0.10 "
         f"(ties broken by run id); ρ̂*_LE = {r['A2']['rho_star_LE']:.4f}. "
         f"Sources: `per_run[].A1.axes.{{ID,WC,OOD}}.ghat` (24-point curves) and "
         f"`.t_star_grid_index` (markers); `per_run[].A2.per_delta.delta_0.1.F_delta` "
         f"(empty, hence the annotation). Display transform: none — curves are plotted "
         f"as stored against the retained-epoch grid.")


def fig_F5(G: List[dict]) -> None:
    r = next(x for x in G if x["run_id"] == "cifar100_symmetric0.6_ce_seed2")
    sels = {s: r["A4"]["selectors"][s]["grid_index"]
            for s in ("E_tau1", "NA", "ER_argmax", "LW_N_reported_not_gating")
            if s in r["A4"]["selectors"]}
    pretty = {"E_tau1": "E(τ=1)", "NA": "NA", "ER_argmax": "ER-argmax",
              "LW_N_reported_not_gating": "LW-N"}
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    _curves_panel(ax, r, "A feasible checkpoint exists — and is not found",
                  {pretty[k]: v for k, v in sels.items()})
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.0)
    w = r["A2"]["per_delta"]["delta_0.1"]["w_delta"]
    p = r["A5"]["per_delta"]["delta_0.1"]["p_unif"]
    save(fig, "case_study",
         f"**F5 `case_study.pdf`** — the Phase-II compatible run "
         f"`{r['run_id']}` (ρ̂*_LE = {r['A2']['rho_star_LE']:.4f}). The shaded band is the "
         f"single feasible checkpoint, w = 1/24 = {w:.3f}; uniform-random baseline "
         f"p_unif = {p:.3f}. Vertical rules mark where each selector chose. Sources: "
         f"`A1.axes.*.ghat`, `A2.per_delta.delta_0.1.{{F_delta,w_delta}}`, "
         f"`A4.selectors.*.grid_index`, `A5.per_delta.delta_0.1.p_unif`. "
         f"LW-N is drawn but is non-gating.")


def fig_F3(G: List[dict], T: List[dict]) -> None:
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    x0 = 0
    for frame, runs in (("Phase II (15)", G), ("Tier 1 (36)", T)):
        s = sorted(runs, key=lambda r: r["A2"]["rho_star_LE"])
        xs = np.arange(len(s)) + x0
        for x, r in zip(xs, s):
            cls = r["A2"]["per_delta"]["delta_0.1"]["taxonomy"]
            ax.plot(x, max(r["A2"]["rho_star_LE"], 1e-3), "o", ms=4.5,
                    color=OI.get(cls, OI["grey"]), mec="white", mew=0.5, zorder=3)
            if r["A2"]["rho_star_LE"] == 0.0:
                ax.annotate(f"ρ̂*=0.0000\n{r['run_id']}", (x, 1e-3),
                            textcoords="offset points", xytext=(6, 14), fontsize=6,
                            color=OI["grey"],
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["grey"]))
        ax.text(xs.mean(), 3.4, frame, ha="center", fontsize=8)
        x0 += len(s) + 3
    ax.axvline(len(G) + 1, color=OI["grey"], lw=0.6, ls="-")
    ax.set_yscale("log")
    delta_lines(ax)
    ax.set_ylim(8e-4, 4.0)
    ax.set_xticks([])
    ax.set_xlabel("runs, sorted by ρ̂*_LE within frame")
    ax.set_ylabel("ρ̂*_LE  (log scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=OI[c], label=c)
               for c in ("incompatible", "compatible-solved", "compatible-unsolved",
                         "indeterminate")]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)
    save(fig, "rho_dotplot",
         "**F3 `rho_dotplot.pdf` [HEADLINE]** — all 51 audited runs, ρ̂*_LE sorted "
         "ascending within frame, coloured by the δ=0.10 class. **Log y axis**, chosen "
         "because ρ̂*_LE spans roughly three orders of magnitude and a linear axis "
         "collapses everything below δ into the baseline; the single ρ̂*=0.0000 run is "
         "drawn at the axis floor (1e-3) and annotated, since zero has no position on a "
         "log scale. Sources: `per_run[].A2.rho_star_LE` and "
         "`per_run[].A2.per_delta.delta_0.1.taxonomy` from both battery files. Display "
         "transform: sorting, and the floor substitution just described.")


def fig_F4(T: List[dict]) -> None:
    conds = [("ID->ID", "ID → ID"), ("OOD->OOD", "OOD → OOD"),
             ("ID->OOD", "ID → OOD"), ("mixed->OOD", "mixed → OOD")]
    ns = [50, 100, 200, 500, 1000, 2000]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.0), sharex=True, sharey=True)
    for ax, (key, title) in zip(axs.ravel(), conds):
        M = np.array([[r["A11"]["curves"][key]["q"][str(n)] for n in ns]
                      for r in T if r["A11"].get("curves")])
        med = np.median(M, axis=0)
        lo, hi = np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0)
        ax.fill_between(ns, lo, hi, color=OI["ID"], alpha=0.18, lw=0)
        ax.plot(ns, med, "-o", color=OI["ID"], lw=1.3, ms=3.5)
        ax.axhline(0.9, color=OI["incompatible"], lw=0.8, ls="--")
        ax.text(52, 0.915, "q = 0.9 target", fontsize=6, color=OI["incompatible"])
        ax.set_xscale("log"); ax.set_xlim(45, 2400); ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left")
        # The record's "median max q" is the median of each run's own maximum, NOT the
        # maximum of the median curve. They differ (0.216 vs 0.144 on ID->OOD), and the
        # verified figure is the former, so that is what is annotated.
        med_max = float(np.median(M.max(axis=1)))
        ax.annotate(f"median max q = {med_max:.3f}", (0.97, 0.92),
                    xycoords="axes fraction", ha="right", va="top", fontsize=6.5,
                    color=OI["grey"])
    for ax in axs[1]:
        ax.set_xlabel("clean labels n  (log)")
    for ax in axs[:, 0]:
        ax.set_ylabel("q(n)")
    save(fig, "budget_curves",
         "**F4 `budget_curves.pdf` [HEADLINE]** — clean-label supply curves, 2×2 by "
         "(selection objective → evaluated axis). Line is the median q(n) across the 36 "
         "Tier-1 runs, band is the interquartile range; n on a log axis; the q=0.9 target "
         "is dashed. The two OOD-target-from-ID panels never approach the target — median "
         "max q 0.216 (ID→OOD) and 0.301 (mixed→OOD) — while OOD→OOD reaches it. Source: "
         "`per_run[].A11.curves.<pair>.q` (six n values, B=1000, assessment on D_assess "
         "only). Display transform: median and IQR across runs, as specified.")


def fig_F6(T: List[dict]) -> None:
    cert = [r for r in T if r["A9"]["two_world_certificate"]["certificate_bearing"]]
    r = median_run(cert)
    keys = list(r["A9"]["per_score_pool"].keys())
    dj = [p for p in r["A9"]["two_world_certificate"]["pairs"] if p["disjoint"]]
    hot = set(sum([p["pair"] for p in dj], []))
    fig, ax = plt.subplots(figsize=(W2, 0.34 * (len(keys) + 3) + 0.9))
    rows = []
    for a in ("ID", "WC"):
        g = ghat(r, a)
        rows.append((f"{a} axis (context)", set(np.asarray(GRID)[g <= 0.10].tolist()), False))
    for k in keys:
        rows.append((k, set(r["A9"]["per_score_pool"][k]["F_delta"]), k in hot))
    for y, (label, feas, is_hot) in enumerate(rows):
        for i, e in enumerate(GRID):
            inF = e in feas
            ax.add_patch(Rectangle((i, -y - 0.4), 0.92, 0.8,
                                   fc=(OI["accent"] if (inF and is_hot) else
                                       OI["compatible-solved"] if inF else "#EEEEEE"),
                                   ec="white", lw=0.4))
        ax.text(-0.6, -y, label, ha="right", va="center",
                fontsize=7, color=OI["accent"] if is_hot else "black",
                fontweight="bold" if is_hot else "normal")
    ax.set_xlim(-9, 24); ax.set_ylim(-len(rows), 1)
    ax.set_xticks(np.arange(24) + 0.46); ax.set_xticklabels(GRID, fontsize=5.5, rotation=90)
    ax.set_yticks([]); ax.set_xlabel("retained checkpoint (epoch)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Feasible checkpoints per OOD (score | pool) — {len(dj)} disjoint pair(s) "
                 f"highlighted", loc="left", fontsize=8)
    save(fig, "certificate",
         f"**F6 `certificate.pdf`** — Tier-1 run `{r['run_id']}`, selected as the median "
         f"ρ̂*_LE among the {len(cert)} certificate-bearing runs. Each strip is the "
         f"δ=0.10 feasible set for one (score | pool) OOD variant; the pair whose feasible "
         f"sets are disjoint is highlighted. The top two strips are the ID and WC axes for "
         f"context. Sources: `A9.per_score_pool.<u|o>.F_delta`, "
         f"`A9.two_world_certificate.pairs[].{{pair,disjoint}}`, and — for the two context "
         f"strips only — a threshold read of the stored `A1.axes.{{ID,WC}}.ghat` at 0.10. "
         f"No quantity is recomputed; thresholding a stored curve is display binning.")


def fig_F7(G: List[dict], T: List[dict]) -> None:
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    marks = {"Phase II": "o", "Tier 1": "^"}
    same = {}
    for frame, runs in (("Phase II", G), ("Tier 1", T)):
        for a in ("ID", "OOD"):
            xs = [r["A1"]["axes"][a]["frozen_historical"]["t_star_120"] for r in runs]
            ys = [r["A1"]["axes"][a]["t_star_epoch"] for r in runs]
            ax.scatter(xs, ys, s=16, marker=marks[frame], facecolor="none",
                       edgecolor=OI[a], lw=0.9,
                       label=f"{frame} · {a}")
            same[(frame, a)] = sum(1 for x, y in zip(xs, ys) if x == y)
    ax.plot([0, 119], [0, 119], color=OI["grey"], lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("frozen-historical oracle epoch (120-grid, smoothed)")
    ax.set_ylabel("corrected oracle epoch (24-grid, raw)")
    ax.set_xlim(-4, 124); ax.set_ylim(-4, 124)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.text(0.98, 0.03, "\n".join(f"{f} {a}: identical {same[(f,a)]}"
                                  f"/{15 if f=='Phase II' else 36}"
                                  for f in ("Phase II", "Tier 1") for a in ("ID", "OOD")),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=OI["grey"])
    save(fig, "oracle_shift",
         "**F7 `oracle_shift.pdf` [appendix]** — A1 oracle-epoch scatter, corrected "
         "(24-grid raw argmin) against frozen-historical (120-grid smoothed argmin), with "
         "the identity line. **ID and OOD only**: the WC axis has no frozen-historical "
         "counterpart, because the frozen frame's `tail` axis was R_tail_static over 120 "
         "points — a different object rather than a second reading of the same one, so "
         "plotting it against the corrected WC axis would be a category error. Counts of "
         "identical epochs are annotated (Phase II: ID 1/15, OOD 2/15). Sources: "
         "`A1.axes.<a>.t_star_epoch` and `A1.axes.<a>.frozen_historical.t_star_120`.")


def fig_F8(T: List[dict]) -> None:
    keys = list(T[0]["A9"]["per_score_pool"].keys())
    ep = np.array([[r["A9"]["per_score_pool"][k]["ood_t_star_epoch"] for r in T]
                   for k in keys])
    wd = np.array([[r["A9"]["per_score_pool"][k]["w_delta"] for r in T] for k in keys])
    order = np.argsort([r["A2"]["rho_star_LE"] for r in T])
    fig, axs = plt.subplots(2, 1, figsize=(W2, 3.2), sharex=True)
    for ax, M, cmap, lab in ((axs[0], ep[:, order], "viridis", "corrected OOD oracle epoch"),
                             (axs[1], wd[:, order], "magma", "w$_{0.10}$")):
        im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=6.5)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03).set_label(lab, fontsize=6.5)
    axs[1].set_xlabel("Tier-1 runs, sorted by ρ̂*_LE")
    axs[1].set_xticks([])
    save(fig, "aggregation_heatmap",
         "**F8 `aggregation_heatmap.pdf` [appendix, PARTIAL]** — per-(score | pool) "
         "corrected OOD oracle epoch (top) and w$_{0.10}$ (bottom) across the 36 Tier-1 "
         "runs, columns sorted by ρ̂*_LE. Sources: "
         "`A9.per_score_pool.<u|o>.{ood_t_star_epoch,w_delta}`. "
         "**The requested mean/min/max aggregation row groups are NOT drawn.** The "
         "aggregated OOD curves are stored (`A9.aggregation_sensitivity.<u>.{mean,min,"
         "max}`), but their w_δ is not, and deriving it would mean re-forming the joint "
         "frame — maximizing over the ID, WC and aggregated-OOD normalized regrets and "
         "thresholding — which is a new quantity, not a display transform. Reported "
         "rather than computed.")


def fig_F2() -> None:
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.axhspan(0.10 - 0.025, 0.10 + 0.025, color=OI["indeterminate"], alpha=0.28, lw=0)
    ax.axhline(0.10, color="black", lw=1.0)
    ax.text(0.02, 0.10, " δ", va="bottom", fontsize=8)
    ax.text(0.98, 0.10, "±0.025 indeterminate band ", ha="right", va="center", fontsize=6.5,
            color=OI["grey"])
    ax.annotate("", xy=(0.5, 0.235), xytext=(0.5, 0.128),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=OI["incompatible"]))
    ax.text(0.53, 0.19, "incompatible\nρ̂*_LE > δ  (F$_δ$ = ∅)", fontsize=7,
            color=OI["incompatible"], va="center")
    ax.annotate("", xy=(0.5, -0.03), xytext=(0.5, 0.072),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=OI["compatible-solved"]))
    ax.text(0.53, 0.028, "compatible  ρ̂*_LE ≤ δ\n  solved: ∃ registered selector with Ĵ ≤ δ\n"
                         "  unsolved: otherwise", fontsize=7, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 0.26); ax.set_xticks([])
    ax.set_ylabel("ρ̂*_LE")
    ax.set_title("Corrected run taxonomy (schematic)", loc="left")
    save(fig, "taxonomy_diagram",
         "**F2 `taxonomy_diagram.pdf`** — schematic, no data. The ρ̂*_LE axis against δ "
         "with the ±0.025 indeterminate band and the four classes; the solved/unsolved "
         "split is annotated as “∃ registered selector with Ĵ ≤ δ”, the registered set "
         "being {E(τ=1), NA, ER-argmax}. Post-registered selectors never gate.")


def fig_GA(G: List[dict], T: List[dict]) -> None:
    fig = plt.figure(figsize=(5.1, 2.0))          # ≤ 13cm × 5cm
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.35)
    inc = [r for r in T if r["A2"]["per_delta"]["delta_0.1"]["taxonomy"] == "incompatible"]
    r = median_run(inc)
    ax = fig.add_subplot(gs[0])
    for a in AXES:
        ax.plot(GRID, ghat(r, a), color=OI[a], lw=1.0)
        t = r["A1"]["axes"][a]["t_star_grid_index"]
        ax.plot(GRID[t], ghat(r, a)[t], "o", color=OI[a], ms=3, mec="white", mew=0.5)
    ax.axhline(0.10, color=OI["grey"], lw=0.7)
    ax.text(2, 0.13, "δ", fontsize=6, color=OI["grey"])
    ax.text(0.5, 0.97, "F$_{0.10}$ = ∅", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color=OI["incompatible"])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("one trajectory", loc="left", fontsize=7)
    ax2 = fig.add_subplot(gs[1])
    allr = sorted(G + T, key=lambda x: x["A2"]["rho_star_LE"])
    for i, x in enumerate(allr):
        cls = x["A2"]["per_delta"]["delta_0.1"]["taxonomy"]
        ax2.plot(i, max(x["A2"]["rho_star_LE"], 1e-3), "o", ms=2.6,
                 color=OI.get(cls, OI["grey"]))
    ax2.axhline(0.10, color=OI["grey"], lw=0.7)
    ax2.set_yscale("log"); ax2.set_ylim(8e-4, 4.0)
    ax2.set_xticks([]); ax2.tick_params(labelsize=5.5)
    ax2.set_title("all 51 runs", loc="left", fontsize=7)
    fig.text(0.5, -0.10,
             "Most noisy-label trajectories contain no jointly deployable checkpoint — "
             "and clean ID labels cannot buy OOD selection.",
             ha="center", fontsize=6.5)
    save(fig, "graphical_abstract",
         f"**GA `graphical_abstract.pdf`** — composite, 5.1×2.0in (≈13×5cm). Left: "
         f"miniature of F1, same run `{r['run_id']}` under the same median rule. Right: "
         f"miniature of F3, all 51 runs pooled and sorted, log y, δ=0.10 rule drawn. "
         f"Banner text as supplied. Sources are those of F1 and F3; no additional field is "
         f"read.")


def main() -> int:
    stamp = code_stamp()
    if not stamp.get("git_available") or stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to produce figures from an unattested or dirty tree")
    G = json.load(open(os.path.join(ROOT, "results", "corrected", "battery_g2.json")))["per_run"]
    T = json.load(open(os.path.join(ROOT, "results", "corrected", "battery_tier1.json")))["per_run"]
    print(f"[figures] Phase II {len(G)} runs, Tier 1 {len(T)} runs")
    fig_F1(T); fig_F2(); fig_F3(G, T); fig_F4(T); fig_F5(G); fig_F6(T)
    fig_F7(G, T); fig_F8(T); fig_GA(G, T)

    head = [
        "# Figure captions and source fields",
        "",
        "Presentation only. Every plotted number is read from "
        "`results/corrected/battery_g2.json` or `battery_tier1.json`; JSON paths below are "
        "relative to `per_run[]` unless stated. Where one run is shown, the selection rule "
        "is deterministic — the run whose ρ̂*_LE is the median of its eligible group, ties "
        "broken by run id — so the choice can be checked rather than trusted.",
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
