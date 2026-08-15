# Framing preregistration — Phase-0 → paper

## Status: THRESHOLDS FINALIZED — **STILL NOT BINDING** (B3–B6 open, plus S1-null below)

Update 2026-08-13: `δ = 0.10` and `π = 0.8 (12/15)` received and recorded verbatim in
"Thresholds as finalized" below, closing B1 and B2's *value* question. The document
still does not bind, because B3 (best-selector tie rule), B4 (E's confirmatory variant
under T-sensitivity), B5 (missing-output handling) and B6 remain open, and because the
verification requested with the thresholds surfaced a defect in the S1 condition —
recorded under "Verification of the finalized thresholds". Still no confirmatory
selector output exists.

## Original status: RECEIVED AND TIMESTAMPED — **NOT YET BINDING**

Received 2026-08-13, before any confirmatory baseline or E output was computed or
inspected. Recorded here so the arrival time is on the record.

It is **not registered as binding**, because the classification rule it fixes cannot be
applied as written: the two thresholds the rule turns on, `δ` and `π`, are still
`<PIN>`. A rule whose thresholds are chosen later is not preregistered — the choice
would land after results exist, which is the exact failure the document is written to
prevent. **The blanks were not filled in on this side**; picking them would move the
decision to whoever wrote the code.

Confirmed at the moment of recording: no selector result exists. No E statistic (raw,
T-grid, or normalized), no C1 output, no representation-based output, nothing under
`results/report/` matching a confirmatory selector. The stored logits and features are
shared **inputs** to four consumers, not results.

Section "Blocking gaps" below lists what must be pinned before this becomes binding.
Nothing in the document is edited; the received text is reproduced verbatim.

---

## As received (verbatim)

> # Framing Preregistration — Phase-0 → Paper
> Status: registered BEFORE any confirmatory baseline or E output
> was computed or inspected. Stored logits/features exist on disk as
> shared inputs; no selector results exist at registration time.
>
> ## Definitions
> - Axes: ID, tail-dynamic, OOD-primary (as pinned by the G1 protocol).
> - Confirmatory selector set: {E (T-grid primary variant),
>   C1 noisy-validation, (b) representation-based}.
>   full-D is EXCLUDED from scenario classification
>   (analysis_class: exploratory; scope caveat on record).
> - r(s, a, k): normalized regret of selector s on axis a in run k,
>   relative to the per-axis oracle epoch (G1 frame).
> - Success on an axis: r ≤ δ, with δ = &lt;PIN&gt;.
> - Reliable success: success in ≥ π fraction of the 15 runs,
>   with π = &lt;PIN&gt;.
>
> ## Mechanical classification rule (fixed now)
> - S2 (sufficiency): at least one selector in the confirmatory set
>   achieves reliable success on ALL three axes simultaneously.
> - S1 (conflict / no free lunch): no selector achieves S2, AND the
>   per-axis best selector differs across axes in ≥ π of runs.
> - S3 (mixed): neither S1 nor S2 holds.
> Ties and boundary cases: classified by the rule as written;
> no post-hoc threshold adjustment. Any deviation requires a
> timestamped amendment committed BEFORE further analysis.
>
> ## E prediction adjudication (separate from classification)
> The frozen predictions (late-epoch selection bias; largest
> oracle gap on OOD) are adjudicated exactly as registered in
> G2_design_memo.md. Their outcome strengthens or weakens the
> narrative but does NOT gate the S1/S2/S3 classification.
>
> ## Framing commitments (binding)
> - S1 → audit-paper framing ("no clean oracle, no free lunch across
>   deployment axes"); target analysis-track venues (TMLR / NeurIPS
>   D&B / analysis tracks); a real-noise dataset extension is
>   scheduled as a required robustness addition before submission.
> - S2 → sufficiency framing; positioning obligation: explicit
>   boundary against existing noisy-validation literature (criterion
>   vs. evaluation-axes contribution); method track becomes eligible.
> - S3 → conditional-guideline framing ONLY if a compact rule
>   (≤ 2 factors, e.g., noise type × learner) explains the pattern;
>   otherwise default to S1-style audit with reduced claims.
> - No reframing after results are seen except via timestamped
>   amendment with stated rationale.

---

## Blocking gaps — must be pinned before this binds

**B1. `δ` is unpinned.** Success on an axis is `r ≤ δ`. Every downstream count depends
on it. For calibration, G1's own conflict threshold on a comparable normalized scale is
NCR > 0.10, and the exploratory selectors measured so far sit at normalized regrets of
roughly 0.06–1.4 — so δ is discriminating over that whole range, not a formality.

**B2. `π` is unpinned**, and is used for **two different quantities**:

- in *reliable success* — the fraction of runs in which one selector clears δ on one
  axis (a per-selector, per-axis success rate);
- in *S1* — the fraction of runs in which the per-axis **best** selector differs across
  axes (a cross-axis disagreement rate).

These are different constructs and there is no reason one number should serve both. If
sharing the symbol is deliberate, say so; otherwise the second needs its own threshold.

**B3. "per-axis best selector" has no tie rule.** With three selectors and three axes,
exact or near-exact ties are likely, and S1 turns on whether the argmin differs across
axes. "Ties classified by the rule as written" does not resolve this, because the rule
does not define a best-selector tie-break. Options that must be chosen now, not later:
strict argmin with a fixed ordering; a margin below which two selectors count as tied
(and how a tie is then scored for the "differs" test); or exclusion of tied runs with
the count reported.

**B4. E's confirmatory variant is ambiguous.** The set names "E (T-grid primary
variant)", but the registered T-grid procedure produces **four** selections
(`T ∈ {0.5, 1, 2, 5}`) plus a `T-robust` / `T-sensitive` label — it does not designate
one `T` as primary. If E is T-robust the four agree and the ambiguity is moot; if E is
T-sensitive, which selection enters classification is currently undefined. Pin it now
(e.g. `T = 1` as the canonical untempered case, with the grid reported alongside).

**B5. Missing or undefined selector output is unhandled.** If a selector yields no
selection for some run, the rule does not say whether that run counts as a failure, is
dropped from the denominator, or blocks classification.

**B6 (lower severity). S3's "compact rule ... explains the pattern"** has no fit
criterion. This gates a framing branch rather than the mechanical classification, but
"explains" will need a stated bar.

## Scope caveat that should be settled at the same time

The classification is defined over "the 15 runs" — which is not a neutral sample. The
15 runs with checkpoints are exactly the **5 cells in which the G1 verdict found
ID↔OOD conflict**, selected for that reason when B2 was scoped; the other 11 of the 16
cells have no checkpoints and cannot be evaluated:

| in the confirmatory universe (5 cells × 3 seeds) | excluded (11 cells) |
|---|---|
| `cifar100_symmetric_0.4_ce` | the 11 non-ID↔OOD-conflict cells |
| `cifar100_symmetric_0.6_ce` | |
| `cifar10_asymmetric_0.4_ce` | |
| `cifar10_asymmetric_0.4_elr` | |
| `cifar10_symmetric_0.2_elr` | |

This matters most for **S1**, whose claim is that axes conflict: adjudicating it on
cells chosen because they exhibited conflict makes the sample enriched for the
conclusion. The finding may still hold, but the claim it licenses is narrower than
"no free lunch across deployment axes" unqualified — at minimum it is *conditional on
cells where G1 already established ID↔OOD conflict*.

Three ways to settle it, all of which should be decided before results:
1. **State the conditioning** in the S1 framing and in the paper claim.
2. **Extend the checkpoint universe** to some non-conflict cells so the classification
   sample is not conflict-selected (costs a further B2-style rerun; those cells have
   only their final checkpoint today).
3. **Re-scope the classification** to a question the sample can answer without
   enrichment.

S2 is less affected — a selector succeeding on all three axes *in the hardest cells* is
a stronger result than in a neutral sample, not a weaker one.

---

# Thresholds as finalized (received 2026-08-13, verbatim)

> ## Definitions (thresholds finalized)
> - Success on an axis: r ≤ δ, with δ = 0.10.
>   Normalized regret r is defined as in docs/analysis_protocol.md
>   [VERIFY & CITE the exact definition here before commit].
>   Rationale: (i) 10% relative-to-oracle degradation is the
>   operational boundary for "effectively oracle-free"; (ii) δ lies
>   below the best median regret observed in the EXPLORATORY full-D
>   analysis (+0.14), which is excluded from classification — the
>   threshold is informed only by excluded pilot data; no
>   confirmatory selector output existed at registration time.
> - Reliable success: success in ≥ 12 of 15 runs (π = 0.8).
>   Rationale: under H0 of coin-flip success (p = 0.5),
>   P(≥12/15) ≈ 0.018 — chance-level performance is rejected at
>   ~2%. The same π applies to the S1 axis-disagreement condition;
>   this reuse is intentional and shares the same binomial logic.
> - Sensitivity (non-binding): classification is additionally
>   reported under δ = 0.05 and δ = 0.20 for transparency;
>   the binding classification uses δ = 0.10 only.

# Verification of the finalized thresholds

## The requested citation: what `r` actually is, and where it is defined

`docs/analysis_protocol.md` **does not define selector regret** — the word "selector"
does not occur in it, because no selector existed at G1. What §5 defines is the
oracle-versus-oracle cross-regret and its normalization:

> **CR[a, b] = R_a[t*_b] − R*_a** … **IQR_a = np.percentile(R_a, 75) −
> np.percentile(R_a, 25)** (numpy default `linear` interpolation, over all 120 epochs)
> … **NCR[a, b] = CR[a, b] / IQR_a**, and exactly `0.0` when `IQR_a == 0`.

The operative definition of `r(s, a, k)` is in **`src/analysis/selection.py::regret_at_epoch`**,
and it is the same normalization with the selector's chosen epoch in place of another
axis's oracle epoch:

```
r(s, a, k) = ( R_a[ t_selected(s, k) ] − R_a[ t*_a(k) ] ) / IQR_a(k)
```

with, for run `k` and axis `a`: `R_a` the axis's risk trajectory over all 120 logged
epochs in the **G1 frame**; `t*_a` the axis's oracle epoch, i.e. the argmin of the
3-epoch moving average (first index on ties), per protocol §5; `IQR_a` the interquartile
range of `R_a` over all 120 epochs, numpy `linear` interpolation; and `r = 0.0` by
convention when `IQR_a == 0`. Note `t_selected` is restricted to the 24 retained
checkpoint epochs, whereas `t*_a` may fall on any of the 120 — so `r ≥ 0` is not
guaranteed by construction but is expected in practice.

**Recommended citation text for the preregistration**, replacing the `[VERIFY & CITE]`
placeholder: *"r(s, a, k) = (R_a[t_selected] − R_a[t\*_a]) / IQR_a, using the oracle
epoch and IQR definitions of docs/analysis_protocol.md §5; implemented in
src/analysis/selection.py::regret_at_epoch."*

## Rationale (ii) is true of full-D but not of the exploratory record as a whole

δ = 0.10 does sit below full-D's best median (+0.140, `gmm_separation` on tail).
It does **not** sit below every median observed in excluded pilot work:

| exploratory analysis (all excluded from classification) | ID | tail | OOD |
|---|---|---|---|
| full-D `gmm_separation` | +0.265 | **+0.140** | +1.104 |
| full-D `gmm_auc` | +0.260 | +0.541 | +0.397 |
| full-D `quantile_proxy` | +0.966 | +0.880 | +1.423 |
| coarse-D (4 statistics) | +0.625 … +0.989 | +0.668 … +1.239 | +0.952 … +1.239 |
| **Label Wave** (premise-violated, 48 G1 runs) | **+0.059** | **+0.091** | +0.438 |

Label Wave's medians on ID and tail are **below δ = 0.10**. The provenance claim holds —
δ is informed only by excluded pilot data — but the calibration claim as written would
mislead: it is not the case that no pilot selector clears δ. Two caveats limit how much
this says: Label Wave was run outside the regime its paper validates (that paper reports
fixed learning rates only — Appendix B "Fixed at 0.01", C.4 sweeping fixed values — and
never evaluates an annealed schedule, so our cosine run is out-of-regime; that annealing
decays churn through optimization temperature is **our** inference, not a claim of the
paper, which is silent on schedules), and its medians are
over all 48 G1 runs rather than the 15 in the classification universe. Suggested repair:
state rationale (ii) as *"δ sits below every full-D median and below all coarse-D
medians; the only exploratory statistic clearing it is the premise-violated Label Wave
on two of three axes."*

## Reliable success: the binomial check passes

P(X ≥ 12 | n = 15, p = 0.5) = **0.017578**, matching the stated ≈1.8%. π = 12/15 = 0.8.

## S1's reuse of π does **not** share the same binomial logic — and inverts the test

Under a null of no structure — for each axis the best selector drawn uniformly and
independently from the three — the axes all agree with probability 3·(1/3)³ = **0.1111**,
so they **disagree with probability 0.8889**. Disagreement is the default, not the
surprise. Consequently:

| quantity under the no-structure null | value |
|---|---|
| P(a run shows axis-disagreement) | 0.889 |
| expected disagreeing runs out of 15 | **13.3** |
| P(≥ 12 of 15 runs disagree) | **0.9235** |

The reliable-success threshold rejects chance at ~2%; the same threshold applied to
disagreement is **passed by chance ~92% of the time**, and 12/15 is *below* the null's
own expectation of 13.3/15. The shared binomial logic does not transfer because the null
rate is ≈0.889, not 0.5.

Raising π does not rescue it: even requiring all 15 runs to disagree leaves
0.889¹⁵ ≈ **0.17** under the null. The weakness is in the criterion, not the threshold.

This does not affect S2, and it does not affect S1's first clause (no selector achieves
S2), which continues to carry real content. It affects only the second, conjunctive
clause. Three repairs, to be chosen before binding:

1. **Drop the disagreement clause** and let S1 rest on S2-failure plus the framing
   caveat — simplest, and S2-failure is the substantive finding.
2. **Require substantive disagreement**: the axis-wise best selectors differ *and* the
   winner's margin over the other selectors on its axis exceeds a stated size, so ties
   and noise cannot manufacture disagreement.
3. **Test the right null**: keep the clause but set the threshold from the 0.889 null
   (i.e. an exceedance level, not a reused 0.8).

## Status of the earlier blocking gaps

| gap | status |
|---|---|
| B1 δ unpinned | **closed** — δ = 0.10, with the rationale amendment suggested above |
| B2 π unpinned / dual use | **value closed** (π = 0.8); the dual use is declared intentional but its justification fails for S1 — see above |
| B3 best-selector tie rule | **open** — and now load-bearing for whichever S1 repair is chosen |
| B4 E's confirmatory variant under T-sensitivity | **open** |
| B5 missing-output handling | **open** |
| B6 S3 "compact rule explains" criterion | **open** |
| scope: 15 runs are the 5 conflict-selected cells | **open** |

The sensitivity reporting at δ ∈ {0.05, 0.20}, non-binding, is recorded and will be
produced alongside the binding δ = 0.10 classification.

---

# Measurement-validity registration (owner decision 2026-08-13)

Registered in response to an external review of the draft (oracle optimism, baseline
fairness, axis construct validity), on the four facts established read-only from the
frozen protocol, the committed code, and the primary source. Documentation scope: no
analysis code was changed, nothing was re-scored, and no run was retrained to register
any of this.

## 1. Selector regret is measured against a grid-restricted oracle

The defect the review named is real and was confirmed in code. `oracle_epoch`
(`src/analysis/ncr.py:32-37`) takes `np.argmin` over an **unrestricted** candidate set,
and its input is the length-120 logged trajectory (§4), while a selector can only choose
among the 24 retained checkpoints. `regret_at_epoch` (`src/analysis/selection.py:28`)
then subtracts the two directly, charging the selector for epochs it was never able to
pick.

**Registered primary endpoint.** Selector regret is measured against

> `t*_grid` = argmin of the **same** 3-epoch-smoothed curve, with candidates restricted
> to the checkpoint grid {4, 9, …, 119}; `R*` read on the **raw** curve at `t*_grid`.

This is the sealed §5 machinery with the candidate set restricted — smoothing window,
tie rule (first index), and the smoothed-argmin/raw-risk pairing are all unchanged. The
primary selector statistic is `R_raw[stop] - R_raw[t*_grid]`, **unclamped**, consistent
with G1's negative-CR convention.

**The 120-point oracle is retained as the information ceiling.** The grid-induced oracle
gap `R[t*_grid] - R[t*_120]` is reported as its own column, unclamped, never folded into
selector regret.

**The gap's sign is non-monotone, by construction.** Because `t*` is the argmin of the
*smoothed* curve while `R*` is read on the *raw* curve (§5), restricting the candidate
set can land on an epoch whose raw risk is **lower** than the full-grid oracle's. In the
15-run measurement this happened in 9 of 45 cells. That is the expected consequence of
the sealed definition, not an anomaly, and the column must not be clamped to hide it.

Measured size of the gap (15 G2 runs × 3 axes = 45 cells, lookup over logged metrics):
oracle epoch identical in 7/45; |Δepoch| max 26; |ΔR/IQR| median 0.1179, ≥ 0.10 in 23/45
(14 where restriction costs, 9 where it helps).

## 2. Label Wave is evaluated only on the logged 1-epoch statistic — option (b) rejected

Recomputing prediction flips from the 24 retained checkpoints is **formally rejected** as
a Label Wave evaluation. The checkpoint grid is 5 epochs apart, so such a recomputation
measures `1[ŷ^t ≠ ŷ^(t-5)]`, a five-epoch-difference statistic, while Eq. (2) is defined
on consecutive epochs; the k = 3 moving average would likewise span 15 epochs rather than
3. It is a different quantity and must never be labelled Label Wave. The evaluation runs
only on the logged, genuinely 1-epoch `pred_flip_count`, whose set, comparison,
normalization and granularity match Eq. (2) exactly. The unresolved items are recorded as
NEEDS-VERIFICATION in `src/analysis/label_wave.py` (k, patience, window warm-up, and the
unspecified prediction-production mode).

## 3. The axis is a worst-class subset, not a frequency tail

**Terminology.** The axis is renamed **"worst-class subset"** in this registration and in
all paper-facing text. Both clean test sets are exactly class-balanced (verified from the
files that train the models: CIFAR-10 test 1000/class, CIFAR-100 test 100/class,
min = max in both), so no frequency tail exists to name. G1's sealed artifacts keep their
field names — *the field named `tail` denotes the worst-30% of classes by clean-test
accuracy*.

**Reference-point rationale.** The static set is anchored at the CE seed-0 run's **final**
epoch because that is the only fixed point the learners in a group share: the class
ranking is read once, after fitting has run its course, and then applied unchanged to
every run and every epoch in the group, so CE and ELR are scored on identical classes.
Anchoring instead at an oracle epoch would make the class set depend on the very
trajectory under evaluation, and anchoring at initialization would rank classes before
any learning had occurred. This is a design choice, not a necessity; a different fixed
anchor would be defensible and would change which classes are in the set.

**Circularity verdict.** The dynamic variant *is* circular — `r_tail_dynamic`
(`src/eval/tail.py:40-45`) re-selects the worst 30% at each epoch, so the checkpoint under
evaluation defines its own target set. It is **not** an analysis axis: protocol §4 uses
the §3 static set, `analysis/io.py:104` reads `r_tail_static_series`, and §2 lists
`R_tail_dynamic` among the logged-but-not-analyzed fields. Its only live use is the
forward-pass integrity recomputation, where reproduction against the log is the whole
question and circularity is irrelevant. **The analysis axis is not circular.**

## 4. The near-zero-IQR guard is retained and empirically inert

`normalized_regret = 0.0 if IQR == 0` (`src/analysis/selection.py:31`) stays as written.
It never fired on the measured runs: across 105 series-run values (7 series × 15 runs) the
smallest trajectory IQR is **0.01721** (`msp_CIFAR-C-local`), and the primary endpoint's
smallest is 0.03414. No series comes within a factor of ten of the guard, so any
near-zero rule fixed on these magnitudes is a safeguard against a case that has not
occurred rather than a live filter.
