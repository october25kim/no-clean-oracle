# Step-6 Adjudication — Corrected Primary + Frozen Historical
Executed review-side against remediation_plan_v2 (anchored) and
theory_spec_v2 (77cfae00… post-transport / 213fddf7…
pre-transport). Inputs: battery_g2.json, battery_tier1.json,
REPORT_27/28 (anchored). Adjudicator outcome-blind until these
anchored reports; Tier-1 seals held until plan anchoring →
per spec E.1, Tier-1 results are PROSPECTIVE confirmation.

## 1. Corrected taxonomy ruling [CORRECTED-PRIMARY]
G2-15 (δ=0.10): incompatible 14 · compatible-solved 0 ·
compatible-unsolved 1 · indeterminate 0.
Tier1-36 (δ=0.10): incompatible 27 · compatible-solved 3 ·
compatible-unsolved 4 · indeterminate 2.
Branch sentence selected (plan §3, verbatim, binding):
"trajectory incompatibility dominates; selector failure there is
necessary; information content is adjudicated on the compatible
minority." Mixed/unsolved-majority sentences: not applicable.
Robustness: taxonomy unchanged across the full floor grid (A8);
no A3 flip under the ε grid (A7, G2); indeterminate runs (2)
withheld from all claims.

## 2. Reinterpretation of the frozen 0/45 [both published]
FROZEN-HISTORICAL: 0/45 OOD-robust selections stands as
registered. CORRECTED: in 14 of those 15 runs no jointly
δ-adequate checkpoint existed on the grid (ρ̂*_LE > 0.10);
failure there was necessary. The selector-insufficiency claim
survives ONLY on the compatible minority: 8 compatible runs
across frames, 3 solved by some registered selector; the G2
compatible run (ρ̂*_LE = 0.0000, w = 1/24) was missed by all
three registered selectors against a uniform baseline of 0.042.

## 3. Frozen branch table [FROZEN-HISTORICAL]
R1 rejected (joint successes at δ=0.10 imply per-axis OOD
successes > 0; compatible-solved 8 at δ=0.20). R2 rejected
(max joint reliability 2/36 < 29/36). RULING: R3 — reported
against the table with no reframing, per registration. Exact
per-axis counts reside in ext_counts.json (unsealed, commitment
83d4b10c…).

## 4. E predictions on Tier 1 [LE]
P1 (late-epoch bias): supported — LW-N median mapped epoch 119;
E per-run selections concentrate late (per artifact). P2
(largest gap on OOD): aggregate CONFIRMED on both scales
(normalized regret and epoch gap); per-run 26/36 and 28/36;
all four cells reported, none cited alone.

## 5. Constructive results
Budget curves: n*(ID→ID) reached 18/36 (median 2000);
n*(OOD→OOD) 21/36 (median 1000); n*(ID→OOD) 1/36 with median
max q 0.216; n*(mixed→OOD) 2/36, median max q 0.301. Reading
licensed by spec F.3: the binding constraint on OOD-adequate
selection is sample TYPE (OOD-representative validation data),
not clean-label quantity. Two-world certificates: 3/36 runs,
7 disjoint pool pairs — empirical instantiations of theorem G.1
on real trajectories.

## 6. Honesty entries
(a) CF benchmark excess negative in 9/36 runs — selectors beat
the CF minimax procedure there; reported sign-free. (b) LOSO:
5/12 cells unstable (all 2–1 splits) — all claims are run-level
counts; cell-level generalizations are not licensed. (c) D-7:
real defect, zero firings and zero exclusions on both frames —
frozen successes are not artifacts of the fail-open guard.
(d) A13 descriptive: mean ρ̂*_LE CI [0.2422, 0.4084] lies
entirely above δ = 0.10.

## 7. Claim licensing instantiation (spec G.2 table)
C1′: empirical joint incompatibility in 27/36 Tier-1 and 14/15
G2 runs, robust to registered sensitivities. C2 (reformulated):
on conflict-enriched trajectories, selector failure was
necessary in 14/15; on the compatible minority across frames,
registered selectors solved 3/8. C3: GRANTED as prospective —
the qualitative pattern (incompatibility dominance, OOD-largest
gaps, selector insufficiency on compatible runs) reproduced on
human-annotation noise across four learner mechanisms under
seals held until plan anchoring. Budget corollary: as §5.
Forbidden phrases remain in force.

---

## Verification record (session-side, added on commit)

Committed only after `scripts/verify_step6.py` — itself committed before it ran, per R2 —
recomputed every figure above from `results/corrected/battery_g2.json`,
`results/corrected/battery_tier1.json` and the unsealed
`results/sealed_ext/ext_counts.json`. **45 of 45 checks pass**; the transcript is
`results/corrected/step6_verification.json` (`code_stamp`, `git_tree_dirty: false`).

Verified figures: both taxonomy tables at all three δ; compatible-run counts and
per-selector successes on both frames; the cross-frame totals of 8 compatible and 3
solved; the G2 compatible run's ρ̂*_LE and w; the frozen-branch derivations including max
joint reliability 2/36 against the 29/36 bar; both A14 tables; the certificate counts;
all four budget n*-reached counts and the two median max-q values; the A6 median and its
9/36 negative excesses; the A13 interval; LOSO 7/12; and zero D-7 firings and exclusions
on both frames.
