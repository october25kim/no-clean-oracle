# Classification Adjudication Record
Executed against: docs/framing_prereg.md (commit 3366144),
δ = 0.10 binding, sensitivity δ ∈ {0.05, 0.20}, π = 12/15.
Inputs: E_tgrid.json (sha256 a4d8d299…), C1_noisyval.json
(97f64497…), B_representation.json (cd2d7ae2…).
Adjudicator: review side (local analysis reviewer).
Status at adjudication: all three confirmatory selector outputs
existed and were inspected; classification rule predates all
outputs (commit 3366144, 07:57:56, pre-computation evidence on
record). Mechanical counts below are subject to independent
recomputation (classification_verification.json); this record
is committed only after that verification passes.

## 1. Mechanical classification

### 1.1 S2 test — per-axis success counts (r ≤ δ = 0.10)
Selector      | ID    | tail  | OOD
E (T=1)       | 1/15  | 3/15  | 0/15
C1 (primary)  | 2/15  | 3/15  | 0/15
(b) argmax    | 1/15  | 3/15  | 0/15
Required for reliable success: ≥ 12/15 per axis, all three axes
simultaneously, for at least one selector.
Result: no selector reaches 12/15 on any single axis. S2 REJECTED.

### 1.2 S1 test — axis-disagreement count
Convention: per run, the per-axis best selector is the argmin of
regret over the three selectors on that axis; a run counts as a
disagreement iff the best selector differs across axes; exact
regret ties (selectors choosing the same epoch) do NOT constitute
disagreement.
Count: 6/15 runs show axis disagreement.
Required: ≥ 12/15. S1 disagreement condition NOT met.

### 1.3 Classification
S2 rejected, S1 not met → class S3 (mixed), per the registered
mechanical rule.

### 1.4 Sensitivity (non-binding, registered)
δ = 0.05: S2 rejected (all counts ≤ their δ=0.10 values).
δ = 0.20: S2 rejected — OOD success remains 0/15 for every
selector (minimum OOD regret across all 45 selector×run cells
is +0.289 > 0.20). The disagreement count is δ-independent.
Classification is robust to the registered threshold range.

## 2. S3 default-clause ruling (interpretive)
Registered clause: S3 → conditional-guideline framing ONLY if a
compact rule (≤ 2 factors) explains the pattern; otherwise
default to S1-style audit with reduced claims.
Candidate compact rule: learner × noise level. All δ=0.10
successes on ID and tail occur in ELR cells, predominantly
sym0.2. The rule explains the ID/tail pattern.
Ruling: the clause's condition is NOT satisfied in the sense
required, because the compact rule yields no deployment-robust
guideline: OOD success is 0/45 unconditionally, including every
ELR cell. A rule that cannot name any condition under which all
three axes are jointly selectable does not support a
conditional-guideline paper. DEFAULT CLAUSE APPLIES:
framing = S1-STYLE AUDIT with reduced claims.
Headline claim licensed by the data: no clean-validation-free
selector, at any registered threshold up to δ = 0.20, selected
an OOD-robust checkpoint in any of 45 selector×run cells.

## 3. E prediction adjudication
Registered predictions (G2_design_memo.md, frozen pre-analysis):
  P1 late-epoch selection bias.
  P2 largest oracle gap on OOD.
P1: CONFIRMED. E(T=1) median selected epoch = 119 (grid maximum);
15/15 runs select epoch ≥ 104 (minimum selected epoch = 104).
[Correction note: an earlier unissued draft stated 13/15; the
figure failed independent recomputation (verification session,
fd4c908) and was corrected to 15/15 before this record was
committed. The error direction was conservative — the corrected
figure supports P1 more strongly.]
P2: neither the comparison granularity nor the gap metric was
registered; all levels are recorded and none may be cited alone:
  - Aggregate (median normalized regret): CONFIRMED
    (ID +0.570 / tail +0.439 / OOD +2.242).
  - Per-run, normalized-regret metric: PARTIAL — OOD largest in
    9/15 runs; cifar100 high-noise cells place ID or tail above.
  - Per-run, epoch-gap metric: 12/15 runs place OOD largest.
  The regret-metric figure (9/15) is treated as primary for
  narrative purposes because regret is the registered success
  measure elsewhere in this prereg; the epoch-gap figure is
  reported alongside as sensitivity.

## 4. Post-hoc statistic pin (E degree of freedom)
Primary statistic: mean max-softmax. Entropy variant: non-binding
sensitivity. Justification: resolved AFTER results exist, but the
two variants' T-robust/T-sensitive verdicts (4/15 vs 3/15) were
concordant and on record BEFORE this resolution; classification
sensitivity to the choice is accordingly low. Recorded per R1 as
a follow-up commit, not an amendment.

## 5. Downstream commitments triggered
- Audit framing fixed; target venues per prereg S1 clause.
- Real-noise dataset extension: scheduled as required robustness
  addition before submission (S1 clause applies to the S1-style
  ruling).
- Tabled items unchanged: disambiguation sweep, B-iv, fixed-LR
  study.
