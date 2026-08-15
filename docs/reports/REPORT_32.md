# REPORT 32 — Figures v2 and the data tables

**Project** G1-NoCleanOracle · **Session** ext_realnoise · **Issued** 2026-08-15
**Scope** presentation only — every plotted number already existed in the battery artifacts
**Code** `scripts/make_figures.py`, committed before each run (R2)

The amendment supersedes the first figure set. F2 and F8 are deleted, F1 is redesigned,
F6/F7 move to the appendix, and two data tables are emitted. Seven PDFs with PNG previews
under `figures/`, tracked; `figures/captions.md` carries per-figure source paths, the
run-selection rule, and the single payload each figure is built to deliver.

## Main figures

**F1 `mechanism_windows.pdf`** — payload: *the joint feasible set is empty*. Tier-1
`c10n_worst_gce_seed1` (median ρ̂*_LE among the 27 incompatible runs, ties by run id;
ρ̂*_LE = 0.3346). Top panel: three ĝ_a(t) curves, per-axis argmins marked and dropped to the
axis. Bottom panel, on the same epoch axis: one feasibility strip per axis, then a joint
strip carrying the banner. ID is feasible at a single epoch, WC at a different single
epoch, OOD at four — and the intersection is empty. `concept_trajectory.pdf` is superseded
and deleted.

**F3 `rho_dotplot.pdf`** — payload: *runs sit well above δ, not marginally above it*. All
51 runs, frames separated, sorted within frame, coloured by δ=0.10 class. All three δ lines
drawn, with the Tier-1 counts above each boxed inline (29 / 27 / 19 at δ = 0.05 / 0.10 /
0.20) so the sensitivity shift reads off one figure. Log y, stated in the caption. Both
ρ̂* = 0.0000 runs are named at the axis floor.

**F4 `budget_curves.pdf`** — payload: *on-axis pairs saturate toward the target; the
OOD-from-ID pairs hit a ceiling far below it*. Median q(n) with IQR band, log n, q = 0.9
dashed, and the two off-diagonal panels annotated inline with the verified 0.216 and 0.301.

**F5 `case_study.pdf`** — payload: *the distance between the star and the marker cluster*.
`cifar100_symmetric0.6_ce_seed2`: three curves bottoming at the same epoch, starred,
w = 1/24 = 0.042, p_unif = 0.042; vertical rules where E(τ=1), NA, ER-argmax and LW-N each
selected, with a span annotated between the star and the cluster.

## Appendix figures

**F6 `certificate.pdf`** — `c100n_gce_seed2` (median ρ̂*_LE among the 3 certificate-bearing
runs), per-(score | pool) feasibility strips with the disjoint-pair keys highlighted, ID
and WC strips above for context. This run carries 4 disjoint pairs.

**F7 `oracle_shift.pdf`** — corrected against frozen-historical oracle epochs, both frames,
identity line, identical-epoch counts annotated. ID and OOD only: the WC axis has no
frozen-historical counterpart, since the frozen frame's `tail` axis was `R_tail_static`
over 120 points — a different object, not a second reading of the same one.

**GA `graphical_abstract.pdf`** — 5.1 × 2.0 in (≈13 × 5 cm): the F1 strip panel with the
empty joint row on the left, the F4 ID→OOD panel with its flat ceiling on the right, banner
text as supplied.

## Tables

**`docs/tables/A-T1_per_run.md`** — all 51 runs, both frames: run id, ρ̂*_LE, w at the three
δ, and the δ=0.10 class, sorted ascending by ρ̂*_LE within frame.

**`docs/tables/A-T3_ood_decomposition.md`** — per-(score | pool) medians of ρ̂*_LE, w(0.10),
the corrected OOD oracle epoch, the count of runs with ρ̂*_LE > δ, and the count of runs
with a zero 24-grid OOD IQR.

## One requested element withheld, for the reason F8 was cut

A-T3 asks for one row per aggregation variant confirming taxonomy invariance. **Those rows
are not produced.** The aggregated OOD curves are stored
(`A9.aggregation_sensitivity.<u>.{mean,min,max}`, 24 points each), but their taxonomy is
not. Obtaining it means giving each aggregated curve its own corrected oracle and 24-epoch
IQR denominator, re-forming `max_a ĝ_a(t)` against the unchanged ID and WC axes, and
thresholding at δ — a new quantity, not a display transform. The omission is stated inside
the table itself, not only here.

**One row of that block is available with no new computation, and is given.** The energy
score averaged over the semantic pools {svhn, cross_cifar} *is* the primary R_OOD by
definition (protocol §4), so its taxonomy is the primary taxonomy already reported:
27 / 3 / 4 / 2 at δ = 0.10. The remaining variants — min, max, and the msp-scored
equivalents — need the derivation above. Authorizing it would complete both the table block
and the cut F8 in one short run.

## Two defects caught during production

**F4's annotation, first draft.** The record's "median max q" is the median over runs of
each run's own maximum; the panel had annotated the maximum of the median curve. On ID→OOD
those differ — 0.144 against the verified 0.216 — so the figure would have contradicted the
adjudication record it illustrates. Fixed before release, with the distinction noted in
both the code and the caption.

**F1's strip panel, first draft of v2.** The strips were drawn in index coordinates while
the curve panel used epochs; under `sharex` they collapsed into a narrow band with
overlapping tick labels. Strips now use epoch coordinates and the row labels moved to the y
axis. Caught by looking at the rendered PNG rather than by trusting that the code was
right.

The R2 guard also did its job once: the script refused to draw while regenerated outputs
left the tree dirty, which is the intended behaviour and not a fault.

## State

No analysis was run; the battery artifacts, the seals and the canonical checks are
unchanged from REPORT_28. Standing items unchanged: A10b open and non-blocking, Tier 2
gated.
