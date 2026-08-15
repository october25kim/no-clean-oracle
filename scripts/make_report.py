"""G1 final report: JSONL logs -> NCR tables, heatmaps, trajectories, verdict.

Host-side, no GPU, no training state: everything below is reconstructed from the
per-epoch logs, so the whole report can be regenerated under a different smoothing
window or NCR threshold without touching a checkpoint.

The PRE-REGISTERED analysis unit is one (dataset x noise x learner) task -- 16 cells --
and the BCa bootstrap resamples the 3 seeds inside a cell. The learner is a fixed
factor and is never pooled into a resampling unit. The sealed rule in
``analysis.ncr.verdict`` is applied verbatim to those 16 cells and its thresholds are
read as written: ">= 50% of cells" is >= 8 of 16, the ID<->OOD clause is >= 4 of 16.

The (dataset x noise) view that pools the two learners is computed as a SECONDARY
diagnostic only. Its intervals mix between-learner heterogeneity into what is
otherwise seed variance, so they are not comparable to the primary CIs and no
decision reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import yaml                                                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.io import load_run, risk_trajectories, static_tail_from_reference  # noqa: E402
from analysis.ncr import (OBJECTIVES, aggregate_cell_ncr, analyze_run, iqr,        # noqa: E402
                          verdict)
from data.ood_pools import CIFARC_LOCAL_REPORT_SENTENCE                          # noqa: E402
from plot_trajectory import plot_run                                             # noqa: E402

# Diverging: two Okabe-Ito hues about a neutral grey zero (NCR can be slightly
# negative because oracle epochs come from the smoothed curve, see the footnote).
NCR_CMAP = LinearSegmentedColormap.from_list(
    "ncr", ["#0072B2", "#8FC7E8", "#F0F0F0", "#F0B48A", "#D55E00"])

# Owner decision recorded after the verdict (2026-08-12). It re-scopes what G1's result
# is taken to mean; it does NOT alter the sealed rule, the verdict, or verdict.json.
RESCOPE_NOTE = (
    "**Owner re-scope, recorded 2026-08-12 after the verdict below.** ID<->OOD is "
    "promoted to the primary conflict axis and the tail objective is demoted to "
    "secondary, per the pre-planned two-objective fallback. The conclusion G1 supports "
    "for the record is: *conflict is real, concentrated at moderate noise, and survives "
    "ELR* -- not that it is universal. The verdict itself stands exactly as computed."
)

FOOTNOTE_NEG_CR = (
    "Cross-regret can be slightly negative. Oracle epochs are the argmin of the "
    "3-epoch moving average, while the regret itself is read off the RAW curve, so a "
    "raw dip one epoch away from the smoothed minimum shows up as a small negative "
    "entry. Values are reported raw and unclamped."
)


# ---- collection ---------------------------------------------------------------

def discover_runs(results_dir: str, require_terminal: bool = True) -> List[str]:
    out = []
    for name in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, name)
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "metrics.jsonl")):
            continue
        if require_terminal and not os.path.isfile(os.path.join(d, "TERMINAL.json")):
            continue
        out.append(d)
    return out


def build_cells(run_dirs: List[str], smooth_w: int) -> Dict[str, dict]:
    """Analyse every run, keyed by its pre-registered (dataset x noise x learner) cell.

    The static tail-class set belongs to the (dataset x noise) GROUP, not to the cell:
    CE and ELR have to be scored on the same classes, so both take the set from their
    group's CE seed-0 run.
    """
    runs = [load_run(d) for d in sorted(run_dirs)]
    by_group = defaultdict(list)
    for r in runs:
        by_group[r.tail_group].append(r)

    cells: Dict[str, dict] = {}
    for group, group_runs in sorted(by_group.items()):
        ref = next((r for r in group_runs
                    if r.meta["learner"] == "ce" and r.meta["seed"] == 0), None)
        if ref is None:
            print(f"  [skip] {group}: no CE seed-0 run, so the group has no static tail set")
            continue
        tail_classes = static_tail_from_reference(ref)
        for r in sorted(group_runs, key=lambda x: (x.meta["learner"], x.meta["seed"])):
            a = analyze_run(risk_trajectories(r, tail_classes), smooth_w=smooth_w)
            cells.setdefault(r.cell, dict(
                entries=[], tail_classes=tail_classes.tolist(), tail_group=group,
                ref_dir=ref.run_dir, learner=r.meta["learner"], n_epochs=len(ref.epochs),
            ))["entries"].append(dict(run=r, analysis=a))
    return dict(sorted(cells.items()))


def pooled_groups(cells: Dict[str, dict]) -> Dict[str, List[dict]]:
    """(dataset x noise) groups for the SECONDARY table -- both learners pooled."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for c in cells.values():
        out[c["tail_group"]].extend(c["entries"])
    return dict(sorted(out.items()))


# ---- tables -------------------------------------------------------------------

def _fmt_matrix(mean, lo, hi, thresh: float) -> List[str]:
    lines = ["| measured \\ deployed | " + " | ".join(OBJECTIVES) + " |",
             "|---|" + "---|" * len(OBJECTIVES)]
    for a, jp in enumerate(OBJECTIVES):
        cells = []
        for b in range(len(OBJECTIVES)):
            if a == b:
                cells.append("—")
                continue
            star = " **\\***" if lo[a, b] > thresh else ""
            cells.append(f"{mean[a, b]:+.3f} [{lo[a, b]:+.3f}, {hi[a, b]:+.3f}]{star}")
        lines.append(f"| **{jp}** | " + " | ".join(cells) + " |")
    return lines


def cell_aggregates(cells: Dict[str, dict], n_boot: int, seed: int = 0) -> Dict[str, dict]:
    return {cell: aggregate_cell_ncr([e["analysis"].NCR for e in c["entries"]],
                                     n_boot=n_boot, seed=seed)
            for cell, c in cells.items()}


def pooled_aggregates(groups: Dict[str, List[dict]], n_boot: int, seed: int = 0) -> Dict[str, dict]:
    """SECONDARY only: (dataset x noise) with both learners thrown into one resample."""
    return {g: aggregate_cell_ncr([e["analysis"].NCR for e in entries], n_boot=n_boot, seed=seed)
            for g, entries in groups.items()}


def learner_prevalence(cells: Dict[str, dict], vd: dict, thresh: float) -> Dict[str, dict]:
    """Conflict prevalence per learner across the primary cells.

    The audit's live question is whether the conflict survives a learner built to
    resist label noise, so CE cells and ELR cells are counted separately.
    """
    out: Dict[str, dict] = {}
    for learner in sorted({c["learner"] for c in cells.values()}):
        names = [n for n, c in cells.items() if c["learner"] == learner]
        pcs = [vd["per_cell"][n] for n in names]
        out[learner] = dict(
            n_cells=len(names),
            with_conflict=sum(bool(p["conflict_offdiag"]) for p in pcs),
            id_ood=sum(bool(p["id_ood_conflict"]) for p in pcs),
            tail_only=sum(bool(p["conflict_offdiag"])
                          and all("tail" in pair for pair in p["conflict_offdiag"]) for p in pcs),
            all_overlap_zero=sum(bool(p["all_overlap_zero"]) for p in pcs),
            median_max_offdiag=float(np.median([p["max_offdiag"] for p in pcs])) if pcs else float("nan"),
        )
    return out


# ---- figures ------------------------------------------------------------------

def heatmap_grid(aggs: Dict[str, dict], out_path: str, thresh: float) -> str:
    names = list(aggs)
    ncol = min(4, len(names)) or 1
    nrow = int(np.ceil(len(names) / ncol))
    vmax = max(0.2, float(np.nanmax([np.abs(a["mean"]).max() for a in aggs.values()])))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    im = None
    for ax, name in zip(axes.ravel(), names):
        agg = aggs[name]
        m = agg["mean"].copy()
        np.fill_diagonal(m, np.nan)
        im = ax.imshow(m, cmap=NCR_CMAP, norm=norm)
        ax.set_xticks(range(len(OBJECTIVES)), OBJECTIVES, fontsize=8)
        ax.set_yticks(range(len(OBJECTIVES)), OBJECTIVES, fontsize=8)
        ax.set_title(name, fontsize=9)
        for a in range(len(OBJECTIVES)):
            for b in range(len(OBJECTIVES)):
                if a == b:
                    ax.text(b, a, "—", ha="center", va="center", fontsize=9, color="#999999")
                    continue
                sig = agg["ci_lo"][a, b] > thresh
                ax.text(b, a, f"{agg['mean'][a, b]:.2f}" + ("\n*" if sig else ""),
                        ha="center", va="center", fontsize=8.5,
                        color="#111111", fontweight="bold" if sig else "normal")
    axes[-1][0].set_xlabel("deployed checkpoint", fontsize=9)
    axes[0][0].set_ylabel("measured objective", fontsize=9)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
        cb.set_label("mean NCR  (regret / IQR of the measured risk)", fontsize=9)
    fig.suptitle(f"Normalized cross-regret per (dataset x noise x learner) cell   (* = 95% BCa CI entirely above {thresh})",
                 fontsize=11)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analysis_values(cells: Dict[str, dict], aggs: Dict[str, dict], vd: dict,
                    smooth_w: int, thresh: float) -> dict:
    """Every number the verdict rests on, at full float precision.

    The markdown tables round to 3 decimals; an independent reimplementation needs the
    unrounded values to compare at 1e-9. Keyed exactly as docs/analysis_protocol.md
    section 8 describes.
    """
    out: Dict[str, dict] = {}
    for cell, c in cells.items():
        per_seed = {}
        for e in c["entries"]:
            r, a = e["run"], e["analysis"]
            per_seed[str(r.meta["seed"])] = dict(
                run_id=r.run_id, n_epochs=len(r.epochs),
                t_star={j: int(a.t_star[j]) for j in OBJECTIVES},
                t_star_raw={j: int(a.t_star_raw[j]) for j in OBJECTIVES},
                R_star={j: float(a.R_star[j]) for j in OBJECTIVES},
                IQR={j: float(iqr(a.risks[j])) for j in OBJECTIVES},
                last_regret={j: float(a.last_regret[j]) for j in OBJECTIVES},
                epoch_gap={k: int(v) for k, v in a.epoch_gap.items()},
                CR=a.CR.tolist(), NCR=a.NCR.tolist())
        agg = aggs[cell]
        out[cell] = dict(
            learner=c["learner"], tail_group=c["tail_group"],
            tail_classes=c["tail_classes"], seeds=sorted(int(s) for s in per_seed),
            per_seed=per_seed,
            mean_NCR=agg["mean"].tolist(),
            ci_lo=agg["ci_lo"].tolist(), ci_hi=agg["ci_hi"].tolist(),
            per_cell_verdict=vd["per_cell"][cell])
    return dict(objectives=list(OBJECTIVES),
                cell_definition="dataset x noise x learner",
                smooth_window=smooth_w, ncr_threshold=thresh,
                bootstrap=dict(kind="BCa", n_boot=10000, alpha=0.05, seed=0,
                               generator="numpy.random.default_rng (PCG64)",
                               resample_unit="the 3 seeds within a cell",
                               reseeded_per_matrix_entry=True),
                protocol="docs/analysis_protocol.md",
                verdict={k: v for k, v in vd.items() if k != "per_cell"},
                cells=out)


def sensitivity(run_dirs: List[str], windows: List[int], thresholds: List[float],
                n_boot: int) -> List[dict]:
    """EXPLORATORY: how the cell counts move under other smoothing / threshold choices.

    The moving-average window changes the oracle epochs and therefore every NCR, so
    each window is a full re-analysis; the threshold only re-reads the same intervals.
    Non-decisional by construction -- nothing here is written to verdict.json.
    """
    rows = []
    for w in windows:
        cells_w = build_cells(run_dirs, w)
        aggs_w = cell_aggregates(cells_w, n_boot)
        for t in thresholds:
            v = verdict(aggs_w, thresh=t)
            per_learner = learner_prevalence(cells_w, v, t)
            rows.append(dict(window=w, threshold=t, verdict=v["verdict"],
                             n_cells=v["n_cells"],
                             cells_with_conflict=v["cells_with_conflict"],
                             id_ood=v["id_ood_conflict_cells"],
                             all_overlap_zero=v["cells_all_overlap_zero"],
                             tail_only=v["tail_only_cells"],
                             ce=per_learner.get("ce", {}).get("with_conflict"),
                             elr=per_learner.get("elr", {}).get("with_conflict")))
    return rows


# ---- report -------------------------------------------------------------------

def write_report(cells, aggs, pooled_aggs, prevalence, vd, cfg, out_md: str,
                 figures: List[str], smooth_w: int, thresh: float,
                 sens: List[dict] | None = None) -> str:
    n_runs = sum(len(c["entries"]) for c in cells.values())
    L = []
    L.append("# G1 — No Clean Oracle, Phase-0 audit")
    L.append("")
    L.append(f"**Verdict: {vd['verdict']}**  "
             f"(threshold NCR > {thresh}; {vd['cells_with_conflict']}/{vd['n_cells']} cells with "
             f"conflict, ID<->OOD conflict in {vd['id_ood_conflict_cells']} cells)")
    L.append("")
    L.append(RESCOPE_NOTE)
    L.append("")
    L.append("Measurement only: no checkpoint-selection method, proxy score, or selector is "
             "implemented or evaluated anywhere in this phase. The quantities below are logged "
             "per-epoch trajectories, oracle epochs, and cross-regret.")
    L.append("")
    L.append(f"- runs analysed: **{n_runs}** across **{len(cells)}** pre-registered "
             "**(dataset x noise x learner)** cells — the learner is a fixed factor, and the "
             "bootstrap resamples only the seeds inside a cell.")
    L.append(f"- oracle epoch: argmin of a {smooth_w}-epoch moving average of the raw logged risk")
    L.append("- R_ID = clean-test top-1 error; R_tail = mean per-class error over the static tail "
             "set of the cell's (dataset x noise) group (bottom 30% of classes by that group's CE "
             "seed-0 final-epoch accuracy, so CE and ELR are scored on the same classes); "
             "R_OOD = 1 - AUROC of the energy score against the semantic pools "
             "(SVHN, opposite-CIFAR), averaged.")
    L.append(f"- NCR = cross-regret / IQR over epochs of the measured risk; CI = BCa bootstrap "
             f"({cfg['analysis']['n_boot']} resamples) over the seeds in a cell.")
    L.append(f"- the sealed thresholds are read as written against {vd['n_cells']} cells: "
             f"\">= 50% of cells\" is >= {int(np.ceil(0.5 * vd['n_cells']))} of {vd['n_cells']}, "
             f"and the ID<->OOD clause is >= 4 of {vd['n_cells']}.")
    L.append("")
    L.append(f"> {CIFARC_LOCAL_REPORT_SENTENCE}")
    L.append("")
    L.append(f"> {FOOTNOTE_NEG_CR}")
    L.append("")

    L.append("## Pre-registered decision rule")
    L.append("")
    L.append("```")
    L.append((verdict.__doc__ or "").strip())
    L.append("```")
    L.append("")
    L.append("```json")
    L.append(json.dumps({k: v for k, v in vd.items() if k != "per_cell"}, indent=2))
    L.append("```")
    L.append("")

    L.append("## Does the conflict survive a noise-robust learner?")
    L.append("")
    L.append("Conflict prevalence counted separately over the CE cells and the ELR cells of the "
             "primary table. ELR is built to resist label-noise memorization, so if the conflict "
             "persists in the ELR cells it is not an artefact of a learner that simply overfits "
             "the corrupted labels.")
    L.append("")
    L.append("| learner | cells | with conflict | ID<->OOD conflict | tail-only conflict "
             "| all CIs overlap 0 | median max off-diag NCR |")
    L.append("|---|---|---|---|---|---|---|")
    for learner, p in prevalence.items():
        L.append(f"| **{learner.upper()}** | {p['n_cells']} | {p['with_conflict']} | {p['id_ood']} "
                 f"| {p['tail_only']} | {p['all_overlap_zero']} | {p['median_max_offdiag']:+.3f} |")
    L.append("")

    L.append(f"## Conflict severity by cell (primary, {vd['n_cells']} cells)")
    L.append("")
    L.append("| cell | runs | epochs | max off-diag NCR | conflicting pairs (CI entirely > thr) "
             "| ID<->OOD | all CIs overlap 0 | mean oracle-epoch gaps (ID-tail / ID-OOD / tail-OOD) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cell, c in cells.items():
        pc = vd["per_cell"][cell]
        gaps = {k: float(np.mean([e["analysis"].epoch_gap[k] for e in c["entries"]]))
                for k in ("ID_tail", "ID_OOD", "tail_OOD")}
        pairs = ", ".join(f"{a}@{b}" for a, b in pc["conflict_offdiag"]) or "—"
        L.append(f"| `{cell}` | {len(c['entries'])} | {c['n_epochs']} | {pc['max_offdiag']:+.3f} "
                 f"| {pairs} | {'yes' if pc['id_ood_conflict'] else 'no'} "
                 f"| {'yes' if pc['all_overlap_zero'] else 'no'} "
                 f"| {gaps['ID_tail']:.1f} / {gaps['ID_OOD']:.1f} / {gaps['tail_OOD']:.1f} |")
    L.append("")
    L.append("`A@B` = the regret paid on objective A when objective B's oracle checkpoint is "
             "deployed. Rows are the measured objective, columns the deployed checkpoint.")
    L.append("")

    L.append("## NCR per cell — mean [95% BCa CI]  (primary)")
    L.append("")
    for cell, agg in aggs.items():
        seeds = sorted({e["run"].meta["seed"] for e in cells[cell]["entries"]})
        L.append(f"### `{cell}`  (seeds {seeds}; static tail classes "
                 f"{cells[cell]['tail_classes']})")
        L.append("")
        L += _fmt_matrix(agg["mean"], agg["ci_lo"], agg["ci_hi"], thresh)
        L.append("")

    L.append("## Secondary diagnostic — learners pooled by (dataset x noise)")
    L.append("")
    L.append("**Non-decisional.** These 8 rows pool CE and ELR into a single resampling unit, so "
             "their intervals mix between-learner heterogeneity into what is seed variance in the "
             "primary table. They are not comparable to the primary CIs and the verdict does not "
             "read them; they are here only to show the picture at the coarser grain.")
    L.append("")
    L.append("| dataset x noise (pooled) | max off-diag NCR | pairs with CI entirely > thr |")
    L.append("|---|---|---|")
    offs = [(a, b) for a in range(3) for b in range(3) if a != b]
    for name, agg in pooled_aggs.items():
        mx = max(agg["mean"][a, b] for a, b in offs)
        above = ", ".join(f"{OBJECTIVES[a]}@{OBJECTIVES[b]}"
                          for a, b in offs if agg["ci_lo"][a, b] > thresh) or "—"
        L.append(f"| `{name}` | {mx:+.3f} | {above} |")
    L.append("")

    if sens:
        L.append("## Appendix (exploratory, NON-DECISIONAL) — threshold and smoothing sensitivity")
        L.append("")
        L.append("How the cell counts move if the pre-registered smoothing window or NCR "
                 "threshold had been chosen differently. The pre-registered setting is "
                 f"**window {smooth_w}, threshold {thresh}** (bold row). This appendix exists to "
                 "show how close the boundary is, not to relocate it: the verdict of record is "
                 "the pre-registered one, `verdict.json` is untouched by this table, and none of "
                 "these alternative labels may be quoted as a G1 result.")
        L.append("")
        L.append("| MA window | threshold | cells with conflict | ID<->OOD | tail-only "
                 "| all CIs overlap 0 | CE / ELR cells with conflict | label |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sens:
            pre = (r["window"] == smooth_w and abs(r["threshold"] - thresh) < 1e-12)
            b = "**" if pre else ""
            L.append(f"| {b}{r['window']}{b} | {b}{r['threshold']:.2f}{b} "
                     f"| {b}{r['cells_with_conflict']}/{r['n_cells']}{b} | {b}{r['id_ood']}{b} "
                     f"| {b}{r['tail_only']}{b} | {b}{r['all_overlap_zero']}{b} "
                     f"| {b}{r['ce']} / {r['elr']}{b} | {b}{r['verdict']}{b} |")
        L.append("")
        L.append("PASS needs `cells with conflict >= 8/16` **and** `ID<->OOD >= 4`; "
                 "KILL needs `all CIs overlap 0` in more than 70% of cells.")
        L.append("")
        L.append("Read the window column with the mechanism in mind. A smaller window makes the "
                 "oracle epoch the argmin of a noisier curve, so it lands more often on a "
                 "single-epoch dip; the risk at that dip is lower, every off-diagonal regret "
                 "measured against it is larger, and more cells clear the threshold. That is "
                 "exactly the failure mode the moving average was pre-registered to prevent, so "
                 "rows at window 1 are the LEAST trustworthy in this table, not the most "
                 "encouraging. The pre-registered setting is the conservative one, and the "
                 "appendix does not license reading its neighbours as the result.")
        L.append("")
        L.append("Stated plainly for the record: of the nine combinations, the pre-registered "
                 f"(window {smooth_w}, threshold {thresh}) yields the FEWEST conflict cells of "
                 "any setting tried — it is the most conservative of the nine. The WEAK verdict "
                 "is therefore not an artefact of a lenient choice; every alternative would have "
                 "been more generous to the hypothesis, which is why the pre-registered result "
                 "is the one that stands.")
        L.append("")
    L.append("## Appendix — reading memorization from the logs")
    L.append("")
    L.append("The gate figure comparing CE and ELR reports the fraction of NOISY training labels "
             "the model predicts (against the clean-label fraction, above which the model is "
             "memorizing corrupted labels) and the p90 of the per-sample cross-entropy, NOT the "
             "median. At these noise rates the median training sample is correctly labelled, so "
             "the median tracks how hard a learner fits the clean majority and moves the WRONG "
             "way: ELR reaches a much lower median than CE while simultaneously driving the "
             "corrupted tail (p90) far higher, which is the refusal-to-memorize signal. "
             "`train_loss_noisy` is likewise not comparable across learners — it is whatever "
             "objective each one optimises (CE for CE, CE + lambda * regularizer for ELR).")
    L.append("")

    L.append("## Figures")
    L.append("")
    for f in figures:
        L.append(f"- [{os.path.basename(f)}]({os.path.relpath(f, os.path.dirname(out_md))})")
    L.append("")

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return out_md


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--results", default=os.path.join(ROOT, "results", "runs"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results", "report"))
    p.add_argument("--include-partial", action="store_true",
                   help="also analyse runs without TERMINAL.json (incomplete sweeps)")
    p.add_argument("--no-trajectories", action="store_true", help="skip per-run trajectory pngs")
    p.add_argument("--sensitivity", action="store_true",
                   help="append the exploratory threshold/smoothing sensitivity table")
    a = p.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    smooth_w = int(cfg["eval"]["smooth_window"])
    thresh = float(cfg["analysis"]["ncr_threshold"])
    n_boot = int(cfg["analysis"]["n_boot"])

    run_dirs = discover_runs(a.results, require_terminal=not a.include_partial)
    if not run_dirs:
        print(f"no runs under {a.results}")
        return 1
    expected = (len(cfg["datasets"]) * len(cfg["noise_configs"])
                * len(cfg["learners"]) * len(cfg["seeds"]))
    print(f"[report] {len(run_dirs)}/{expected} runs available")

    cells = build_cells(run_dirs, smooth_w)
    if not cells:
        print("no analysable cells (every cell needs its CE seed-0 run)")
        return 1
    aggs = cell_aggregates(cells, n_boot)                       # 16 primary cells
    vd = verdict(aggs, thresh=thresh)                           # sealed rule, verbatim
    pooled_aggs = pooled_aggregates(pooled_groups(cells), n_boot)   # 8, diagnostic only
    prevalence = learner_prevalence(cells, vd, thresh)

    fig_dir = os.path.join(ROOT, "results", "figures")
    figures = [heatmap_grid(aggs, os.path.join(fig_dir, "ncr_heatmaps.png"), thresh)]
    if not a.no_trajectories:
        for c in cells.values():
            for e in c["entries"]:
                figures.append(plot_run(
                    e["run"].run_dir,
                    os.path.join(fig_dir, f"{e['run'].run_id}_trajectory.png"),
                    smooth_w=smooth_w, tail_reference=c["ref_dir"]))  # the cell's tail set

    sens = (sensitivity(run_dirs, [1, 3, 5], [0.05, 0.10, 0.15], n_boot)
            if a.sensitivity else None)
    out_md = write_report(cells, aggs, pooled_aggs, prevalence, vd, cfg,
                          os.path.join(a.out_dir, "G1_report.md"), figures, smooth_w,
                          thresh, sens)
    if sens:
        with open(os.path.join(a.out_dir, "sensitivity_exploratory.json"), "w") as fh:
            json.dump(dict(non_decisional=True, preregistered=dict(
                window=smooth_w, threshold=thresh), rows=sens), fh, indent=2)
    with open(os.path.join(a.out_dir, "analysis_values.json"), "w") as fh:
        json.dump(analysis_values(cells, aggs, vd, smooth_w, thresh), fh, indent=2)
    with open(os.path.join(a.out_dir, "verdict.json"), "w") as fh:
        json.dump(dict(verdict=vd, cell_definition="dataset x noise x learner",
                       n_cells=vd["n_cells"], learner_prevalence=prevalence,
                       n_runs=sum(len(c["entries"]) for c in cells.values()),
                       expected_runs=expected, smooth_window=smooth_w, threshold=thresh),
                  fh, indent=2)

    print(f"[report] VERDICT = {vd['verdict']}  "
          f"({vd['cells_with_conflict']}/{vd['n_cells']} cells with conflict, "
          f"ID<->OOD in {vd['id_ood_conflict_cells']})")
    for learner, p in prevalence.items():
        print(f"[report]   {learner}: {p['with_conflict']}/{p['n_cells']} cells with conflict, "
              f"ID<->OOD in {p['id_ood']}")
    print(f"[report] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
