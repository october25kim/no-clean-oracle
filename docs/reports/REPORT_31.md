# REPORT 31 — Manuscript figures F1–F8 and the graphical abstract

**Project** G1-NoCleanOracle · **Session** ext_realnoise · **Issued** 2026-08-15
**Scope** presentation only — every plotted number already existed in
`results/corrected/battery_g2.json` or `battery_tier1.json`
**Code** `scripts/make_figures.py`, committed before it ran (R2); refuses to draw from an
unattested or dirty tree, and did refuse once, correctly, on the first regeneration

Nine vector PDFs with 300-dpi PNG previews under `figures/`, tracked and anchored.
`figures/captions.md` carries, per figure, the exact JSON paths read, the run-selection
rule where one run is shown, and any display transform.

## What each figure shows

**F1 `concept_trajectory.pdf`** — one Tier-1 incompatible run, `c10n_worst_gce_seed1`,
picked by the deterministic rule (median ρ̂*_LE among the 27 incompatible runs, ties by run
id). Three ĝ_a(t) curves with per-axis argmins marked, the three δ lines, and an annotation
that F_0.10 is empty. The concept in one panel: the optima sit at different epochs and no
epoch satisfies all three.

**F2 `taxonomy_diagram.pdf`** — schematic, no data. The ρ̂*_LE axis against δ with the
±0.025 indeterminate band and the four classes; the solved/unsolved split annotated as
"∃ registered selector with Ĵ ≤ δ", the registered set being {E(τ=1), NA, ER-argmax}.

**F3 `rho_dotplot.pdf` [HEADLINE]** — all 51 runs, Phase II and Tier 1 visually separated,
sorted ascending within frame, one dot per run coloured by δ=0.10 class, three δ reference
lines. **Log y**, stated in the caption: ρ̂*_LE spans about three orders of magnitude and a
linear axis flattens everything under δ into the baseline.

One correction to the brief: there are **two** runs at ρ̂*_LE = 0.0000, not one —
`cifar100_symmetric0.6_ce_seed2` in Phase II (the compatible-unsolved case) and
`c100n_elr_seed2` in Tier 1 (compatible-solved). Both are drawn at the axis floor and
annotated by name, since zero has no position on a log scale.

**F4 `budget_curves.pdf` [HEADLINE]** — 2×2 panels, median q(n) across the 36 Tier-1 runs
with an IQR band, n on a log axis, the q=0.9 target dashed. The two OOD-target-from-ID
panels sit far below the target throughout.

**F5 `case_study.pdf`** — `cifar100_symmetric0.6_ce_seed2`: three ĝ_a(t) curves, the single
feasible checkpoint shaded, vertical rules where E(τ=1), NA, ER-argmax and LW-N each chose.
Caption carries w = 1/24 = 0.042 and p_unif = 0.042.

**F6 `certificate.pdf`** — `c100n_gce_seed2`, the median ρ̂*_LE among the 3
certificate-bearing runs. A 24-cell strip per (score | pool) showing that variant's δ=0.10
feasible set, with the keys involved in a disjoint pair highlighted; ID and WC strips above
for context. This run carries 4 disjoint pairs.

**F7 `oracle_shift.pdf` [appendix]** — corrected against frozen-historical oracle epochs,
both frames, identity line, identical-epoch counts annotated (Phase II: ID 1/15, OOD 2/15).

**F8 `aggregation_heatmap.pdf` [appendix]** — per-(score | pool) corrected OOD oracle epoch
and w_0.10 across the 36 Tier-1 runs, columns sorted by ρ̂*_LE.

**GA `graphical_abstract.pdf`** — 5.1 × 2.0 in (≈13 × 5 cm): miniature F1 left, miniature
F3 right, banner text as supplied.

## Three things I did not do silently

**F8 is partial, and the omission is the point.** The requested mean/min/max aggregation
row groups are not drawn. The aggregated OOD curves *are* stored
(`A9.aggregation_sensitivity.<u>.{mean,min,max}`), but their w_δ is not: obtaining it means
re-forming the joint frame — maximizing over the ID, WC and aggregated-OOD normalized
regrets, then thresholding at δ. That is a new quantity, not a display transform, so per
the instruction it is reported rather than computed. The stored half of F8 is drawn in
full. Authorizing the recomputation would complete the panel in one short run.

**F7 covers ID and OOD, not all three axes.** The WC axis has no frozen-historical
counterpart: the frozen frame's `tail` axis was `R_tail_static` over 120 points, which is a
different object rather than a second reading of the same one. Plotting it against the
corrected WC axis would look like a comparison and would not be one.

**F4's annotation was wrong on first draw and was fixed before release.** The record's
"median max q" is the median over runs of each run's own maximum; the panel had annotated
the maximum of the median curve. On ID→OOD those differ — 0.144 against the verified
0.216 — so the figure would have contradicted the adjudication record it illustrates. The
annotation now uses the verified statistic, and the reason sits in a comment beside it.

## Discipline notes

Run selection is deterministic everywhere a single run appears — median ρ̂*_LE of the
eligible group, ties by run id — and both the rule and the resulting run id are in the
caption file, so the choice is checkable rather than something to take on trust. Palette is
Okabe-Ito; base type is 8pt at 3.4in single-column width; δ ∈ {0.05, 0.10, 0.20} appear as
labelled reference lines wherever a normalized-regret or ρ̂* scale is drawn.

No analysis was run. The battery artifacts and the seals are untouched, and the canonical
checks are unchanged from REPORT_28.
