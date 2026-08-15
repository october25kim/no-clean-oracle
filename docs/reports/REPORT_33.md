# REPORT 33 — A9 addendum: A3 taxonomy under the five aggregation variants

**Project** G1-NoCleanOracle · **Session** ext_realnoise · **Issued** 2026-08-15
**Status** [SENSITIVITY, post-adjudication addendum] — cannot alter the anchored step-6
adjudication, which stands on the primary aggregation

## Order of operations

Registered first, computed second. The A9-ADDENDUM text was appended to
`docs/remediation_plan_v2.md` (per R1, appended not edited), the producing script
`scripts/a3_variant_addendum.py` was committed alongside it, and both were anchored at
public HEAD `d02a4727911f94d550b55e2a45d85b41b7d9ffb8` **before a single number was
computed**. R2 and the addendum's own "executed after this registration is anchored" clause
are satisfied by that one cycle.

Method as registered: for each variant the OOD axis is replaced by that aggregation over
the semantic pools {svhn, cross_cifar}, given its own corrected 24-grid oracle and 24-epoch
IQR denominator with the same fail-closed rule the primary uses; ID and WC keep their
stored ĝ curves untouched; `max_a ĝ_a(t)` and the A3 taxonomy are re-derived at all three δ
on both frames. Solved/unsolved is decided by the same registered selector set.

## Result: the taxonomy is NOT invariant

**140 (run × variant × δ) cells change class** relative to the primary energy-mean
aggregation, involving **34 of the 51 runs**. Every flip is listed individually in
`docs/tables/A-T3_ood_decomposition.md` with its run, frame, variant, δ, primary class and
variant class. Counts only below; no interpretation is offered.

### Flips by variant

| variant | flips |
|---|---|
| energy-min | 22 |
| energy-max | 33 |
| msp-mean | 24 |
| msp-min | 24 |
| msp-max | 37 |

### Flips by δ and by frame

| δ | flips | | frame | flips |
|---|---|---|---|---|
| 0.05 | 31 | | Phase II (G2-15) | 30 |
| 0.10 | 40 | | Tier 1 (36) | 110 |
| 0.20 | 69 | | | |

### Flip directions

| primary class | → variant class | count |
|---|---|---|
| indeterminate | incompatible | 30 |
| incompatible | compatible-unsolved | 23 |
| incompatible | indeterminate | 20 |
| compatible-unsolved | incompatible | 18 |
| indeterminate | compatible-unsolved | 10 |
| indeterminate | compatible-solved | 8 |
| compatible-solved | incompatible | 6 |
| incompatible | compatible-solved | 6 |
| compatible-unsolved | indeterminate | 5 |
| compatible-solved | compatible-unsolved | 5 |
| compatible-unsolved | compatible-solved | 5 |
| compatible-solved | indeterminate | 4 |

### Marginal counts, Tier 1, δ = 0.10

| aggregation | incompatible | compatible-solved | compatible-unsolved | indeterminate |
|---|---|---|---|---|
| **energy-mean (primary)** | **27** | **3** | **4** | **2** |
| energy-min | 31 | 1 | 2 | 2 |
| energy-max | 26 | 5 | 4 | 1 |
| msp-mean | 27 | 3 | 2 | 4 |
| msp-min | 29 | 2 | 2 | 3 |
| msp-max | 23 | 6 | 4 | 3 |

Phase II δ = 0.10 marginals run from 12 to 14 incompatible across the six aggregations,
with compatible-solved 0 throughout. Full tables at all three δ, both frames, are in the
anchored table file.

No run was unscorable under any variant: the fail-closed rule excluded nothing, on either
frame.

## Scope

This is a registered sensitivity computed after adjudication. It does not touch the step-6
ruling, the frozen record, or any figure. It is published as it came out, which is what the
registration committed to before the numbers existed.

Standing items unchanged: A10b open and non-blocking; Tier 2 gated; figures phase closed.
