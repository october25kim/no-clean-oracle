# Timestamped Remediation Analysis Plan v2 — FINAL
Honest status: a post-hoc analysis plan committed after Phase
I–II frozen results existed. Outcome-informed scope, disclosed:
EXP-001 (oracle-gap magnitudes, 45 cells), EXP-002 (per-pool
IQR distributions, 105 values; zero-IQR guard firing status),
EXP-003 (G2 full-D medians) were exposed pre-sealing and are in
the exposure ledger; this plan's G2-frame rules were written
with those exposures on record. Tier-1 selector outcomes remain
fully sealed (never computed until Task 4 under commitment);
Tier-1 analyses are therefore prospective. Governing theory:
theory_spec_v2 (213fddf7…). Effective only once committed AND
anchored (R6).

## 0. FACT resolutions (from docs/FACT_REDACTED.md, anchored)
F1 → LW-N applies DIRECTLY to the logged one-epoch
pred_flip_count; 24-grid recomputation rejected.
F2 → historical oracle = 120-grid smoothed-argmin with raw-curve
read; A1 activated.
F3 → analysis axis = static worst-class (WC), non-circular;
"tail" renamed WC everywhere in corrected outputs; sealed G1
field names unchanged with a mapping note.
F4 → R_OOD_primary = mean over {svhn, cross_cifar} of
1−AUROC(energy); msp and CIFAR-C-local logged, non-primary.
DEFECT (new ledger entry D-7): selection.py:31 maps zero IQR to
normalized regret 0.0 — fail-open; corrected rule is fail-closed
(below). Whether it ever fired is adjudicated from unsealed
EXP-002 data in A9.

## 1. Fixed conventions
δ = 0.10 primary; {0.05, 0.20} sensitivity. Grid: the 24
retained checkpoints for EVERY oracle, selector, and
compatibility quantity [CORRECTED-PRIMARY]; the 120-grid
smoothed oracle is [FROZEN-HISTORICAL]. Corrected oracle =
argmin of the RAW logged risk restricted to the 24 retained
epochs; ties → earliest. Denominator d_a = IQR of the raw risk
over the 24 retained epochs; exact-zero IQR → run excluded-with-
flag on that axis (fail-closed) [CORRECTED-PRIMARY]; sensitivity
floors ε ∈ {1e-4, 1e-3, 1e-2} absolute. RNG seed 20260814;
K_CV = 5 stratified.

## 2. Analysis battery (order fixed; CPU; G2-15 then Tier1-36)
A1 Grid+definition coherence: corrected oracles per §1; per-run
deltas vs historical (epoch and regret); all downstream uses
corrected. A2 Compatibility: ρ̂*_LE, F_δ, w_δ at all δ; raw Δ
vectors saved alongside. A3 Taxonomy: incompatible /
compatible-solved / compatible-unsolved / indeterminate
(ρ̂*_LE within ±0.025 of δ under any sensitivity); historical
0/45 re-expressed in this decomposition. A4 Selector accounting:
Ĵ_s,LE, η̂_s,LE within-layer for E(τ=1 primary, τ-grid dual
robustness), NA, ER(argmax primary); adjudicated on compatible
runs; best-achievable gaps on incompatible. A5 Exact chance
baselines: p_unif = w_δ; expected uniform joint regret =
mean_t max_a g_a(t); no binomial p-values. A6 CF selection
benchmarks (per-axis and minimax; procedure risks; sign-free).
A7 Set-valued oracles: ε ∈ {0.005, 0.01, 0.02}; CR ranges;
verdict-flip table. A8 Normalization sensitivity: raw Δ;
range-normalized; oracle-relative where R̂* > 0.01; A3 stability
under each. A9 OOD decomposition per (u ∈ {msp, energy},
o ∈ {svhn, cross_cifar, cifar_c_local}): per-(u,o) corrected
oracles and feasibility; aggregation sensitivity {primary mean,
min, max, per-pool}; zero-IQR guard firing audit from unsealed
EXP-002 (D-7 impact statement). A10 Post-registered selectors:
LW-N = retrospective first local minimum of PC′ (k = 3, paper's
value) on the logged 120-epoch pred_flip_count, plateau/tie →
earliest, mapped to the NEAREST retained checkpoint (ties →
earlier); registered deviations: train-mode augmented per-sample-
timing statistic (F1a), grid mapping, k and warm-up convention
NEEDS-VERIFICATION per F1c; evaluated on both G2-15 and Tier1-36.
TGS-N-torsion on all runs; TGS-N-full on G2-15 only, timing
pilot 1 ckpt/dataset-scale, abort variant if >20 min/ckpt;
3 deviations recorded verbatim; no pooling. A11 Clean-label
budget: D_eval 50/50 → D_sel/D_assess seeded stratified; n ∈
{50,100,200,500,1000,2000}; B = 1000; q for (ID→ID), (ID→OOD),
(OOD→OOD), (mixed→OOD); n* at q ≥ 0.9, n_max = 2000;
non-monotone → no n*. A12 LOSO verdict stability. A13 Phase-I
equivalence-style reanalysis [SENSITIVITY]; frozen WEAK verdict
untouched. A14 E prediction re-adjudication on Tier 1 at both
granularities and both metric scales.

## 3. Branch language (pre-written; binds after A3 on Tier 1)
Incompatible-majority: "trajectory incompatibility dominates;
selector failure there is necessary; information content is
adjudicated on the compatible minority." Compatible-unsolved-
majority: "identifiability failure dominates; jointly adequate
checkpoints exist but no registered source-only selector locates
them; w_δ grades difficulty." Mixed: per-learner×dataset split
reported with no reframing beyond these sentences. Frozen
R1/R2/R3 adjudication executed in parallel as
[FROZEN-HISTORICAL]; corrected taxonomy is paper-primary.

## 4. Unsealing protocol (strict order)
(1) this plan committed AND anchored → (2) unseal
results/sealed/* (EXP-linked FACT quantities): reveal salt,
verify commitment, then read → (3) A1–A14 on G2-15 → (4) unseal
results/sealed_ext/* (oracle epochs, E/NA/ER selector outputs,
counts; commitment-verified) → (5) A1–A14 on Tier1-36 →
(6) branch adjudication (frozen + corrected) — REVIEW-SIDE →
(7) publish both. Every unsealed file's commitment verified
before reading; verification transcript reported.

## 5. Reporting rules
Layer tags on every number; frozen/corrected disagreements
published side by side; spec G.2 forbidden phrases bind; defect
ledger updated (incl. D-7) in the same commit as any analysis
exercising a ledgered defect.

---

## Follow-up correction (2026-08-15, R-2)

Appended as a follow-up rather than an edit, per R1.

§4(2) correction: results/sealed/ never existed; EXP-001/002
magnitudes were exposed pre-sealing and live in the exposure
ledger. Step (2) is replaced by: cite EXP-001/002 from the
ledger as already-exposed inputs where A1/A9 require them. The
outcome-informed scope disclosure in the header already covers
this.

## Follow-up restorations (2026-08-15, pin answers 14 and 20)

Two values were dropped when the plan was compressed. Restored
here as follow-up text, not edited into §1/§2.

**A9 / D-7 audit — grid separation (pin 14).** The D-7 firing
audit concerns the FROZEN pipeline and is therefore adjudicated
on the 120-point IQRs as historically computed: did
`selection.py:31` ever map a zero scale to a normalized regret
of 0.0? The corrected analyses use ONLY the 24-epoch IQR with
fail-closed exclusion, and 24-grid zero-IQR incidence is
reported separately. The two grids never mix in this audit.

**A13 — equivalence margin (pin 20).** Margin = 0.10 on NCR;
criterion = BCa 95% upper bound below the margin. Descriptive
only, per the [SENSITIVITY] tag A13 already carries; the frozen
WEAK verdict is untouched.

**A10b — TGS-N deferred (pin 15).** TGS-N-torsion and TGS-N-full
are split out of A10 into a deferred registered addendum. The
battery proceeds WITHOUT them and they gate nothing. The A10b
path is: fetch arXiv 2605.08870, produce
`docs/TGS_TRANSCRIPTION.md` (statistic, sampling, weights,
direction, verbatim quotes and hashes) plus proposed pins,
review side confirms, then it executes. A10 as executed
therefore covers LW-N only.

**Taxonomy gating (pin 3).** compatible-solved is decided by the
registered selector set {E(τ=1), NA, ER-argmax} alone.
Post-registered selectors — LW-N now, TGS-N later — are reported
but never gate the taxonomy.

## A9-ADDENDUM (post-adjudication sensitivity)

Registered 2026-08-15 at anchored public HEAD c7d98e17.

A9-ADDENDUM (post-adjudication sensitivity, registered
2026-08-15 / c7d98e17): for each aggregation variant v ∈
{energy-min, energy-max, msp-mean, msp-min, msp-max} over the
semantic pools, construct R_OOD^v, assign it its own corrected
24-grid oracle and 24-epoch IQR denominator (fail-closed), hold
ID and WC fixed, re-derive max_a ĝ_a(t) and the A3 taxonomy at
δ ∈ {0.05, 0.10, 0.20}, both frames. This addendum cannot alter
the anchored step-6 adjudication, which stands on the primary
aggregation; its output annotates robustness and is published
either way. Executed after this registration is anchored.
