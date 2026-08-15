# REPORT 28 — FINAL: corrected battery A1–A14 (minus A10b) on Tier1-36

**Project** G1-NoCleanOracle · **Session** ext_realnoise (DATA-PRODUCING QUARANTINED WORKER)
**Frame** Tier1-36 · **Issued** 2026-08-15 · **Artifact** `results/corrected/battery_tier1.json`
**Code** internal `c877f62`, anchored at public `cd829f3`/`e2e82fb` before execution (safeguard b)

Layer tags as in REPORT_27: **LE** = empirical lower envelope, **CF** = cross-fitted
benchmark, **FROZEN-HISTORICAL** = registered record. `eta` is formed only from two LE
terms. Spec G.2 forbidden phrases bind this prose. **No adjudication** — step (6) is
review-side, and the plan §3 branch sentences are quoted at the end without being applied.

## §4 step (4) — unsealing transcript

R4 class **trajectory-identity**, tolerance exactly 0 (sha256 equality). Salt 32 bytes each.

| sealed file | schema | committed `sha256(salt‖F)` | recomputed | verdict |
|---|---|---|---|---|
| `ext_oracle_epochs.json` | `ext_oracle_epochs.v1` | `09462aa922a15cb4…` | identical | MATCH |
| `ext_E_tgrid.json` | `ext_E_tgrid.v1` | `352744bea1b024ac…` | identical | MATCH |
| `ext_C1_noisyval.json` | `ext_C1_noisyval.v1` | `b5f76bc8f6c3c42a…` | identical | MATCH |
| `ext_B_representation.json` | `ext_B_representation.v1` | `474c0ac37b640079…` | identical | MATCH |
| `ext_counts.json` | `ext_counts.v1` | `83d4b10c6aef79aa…` | identical | MATCH |

**5/5 verified → unseal authorised.** Each file was opened only after its commitment
verified. Per spec E.1, the Tier-1 outputs were sealed from creation until the plan was
committed *and* anchored, which is the condition under which they count as prospective
rather than as an extended audit.

## Integrity and canonical checks

| check | R4 class | observed | bar | verdict |
|---|---|---|---|---|
| recomputed vs logged `R_ID`, 864 instances | internal-consistency | **1.110e-16** | ≤ 1e-12 | PASS |
| `code_stamp` | — | `git_tree_dirty: false`, head `c877f62` | — | PASS |
| data hygiene | — | exit 0 | — | PASS |
| baselined masks | trajectory-identity | 24 unchanged | 0 | PASS |
| R5 campaign invariant | trajectory-identity | 19 modules, zero drift | 0 | PASS |
| SEALED markers · G1 · B2 | — | 2/2 · 48 runs · 15 runs | unchanged | PASS |

All 36 runs scorable; **0 of 108 axis-instances** excluded by the fail-closed rule.

## A2 / A3 — taxonomy on Tier1-36 [LE]

| δ | incompatible | compatible-solved | compatible-unsolved | indeterminate |
|---|---|---|---|---|
| 0.05 | 29 | 0 | 2 | 5 |
| **0.10** | **27** | **3** | **4** | **2** |
| 0.20 | 19 | 8 | 4 | 5 |

By split × learner at δ = 0.10 (3 seeds each):

| split | CE | ELR | GCE | SOP |
|---|---|---|---|---|
| c100n | 2 inc, 1 c-unsolved | 1 inc, 1 c-solved, 1 indet | 1 inc, 2 c-solved | 2 inc, 1 c-unsolved |
| c10n_random1 | 3 inc | 3 inc | 3 inc | 2 inc, 1 indet |
| c10n_worst | 2 inc, 1 c-unsolved | 2 inc, 1 c-unsolved | 3 inc | 3 inc |

### Selector accounting on compatible runs only [LE]

7 of 36 runs are compatible at δ = 0.10.

| selector | joint successes on compatible runs | gates taxonomy |
|---|---|---|
| E(τ=1) | 1/7 | yes |
| NA | 1/7 | yes |
| ER-argmax | 2/7 | yes |
| LW-N | 1/7 | **no — reported only** |

## A5 — exact chance baselines [LE]

`p_unif = w_δ` exactly; observed values at δ = 0.10 are {0.000, 0.042, 0.083, 0.125}.
Expected uniform joint regret `mean_t max_a ĝ_a(t)`: median **1.7630**, range
**[0.9372, 2.9060]**. No simulation, no binomial tail.

## A6 — cross-fitted benchmarks [CF], sign-free

`E(τ=1) Ĵ_LE − CF_minimax`: median **+0.8243**, and **negative in 9/36 runs** — reported
as benchmark excesses with no non-negativity claim, exactly as spec A.4 requires.

## A8 — normalization sensitivity [LE, SENSITIVITY]

The taxonomy is identical under every floor in the grid (1e-4, 1e-3, 1e-2): 27 / 3 / 4 / 2.
Expected here, since no axis was fail-closed to begin with.

## A9 — OOD decomposition and the certificate line [LE]

Per (u, o) corrected oracles, ρ̂*_LE, F_δ and w_δ with ID and WC held fixed, plus
aggregation sensitivity over {mean, min, max, per-pool}, are per-run in the artifact.

**Two-world certificate** (pin 13 addendum): **3 of 36 runs are certificate-bearing**, with
**7 disjoint pool pairs** in total.

### D-7 firing audit, Tier-1 side

| side | grid | result |
|---|---|---|
| historical guard | 120-point IQR | not applicable — the frozen pipeline was never run on Tier 1 |
| corrected exclusion | 24-epoch IQR, fail-closed | **0 exclusions** across 108 axis-instances |

Combined with the G2-15 audit (0 firings, 0 exclusions), D-7 remains a real defect that has
had no opportunity to fire on either frame. Ledger updated in this analysis's commit.

## A10 — LW-N [LE], reported and non-gating

Label Wave replayed on the logged one-epoch `pred_flip_count` (k = 3, first local minimum,
plateau/tie → earliest), then mapped to the nearest retained checkpoint, ties → earlier.
A stopping point was located on **36/36** runs; median mapped epoch **119**.

**A10b (TGS-N-torsion, TGS-N-full) was not computed** — deferred per pin 15, pending a
transcription of arXiv 2605.08870. It gates nothing.

## A11 — clean-label budget curves [LE]

`D_eval` split 50/50 into `D_sel`/`D_assess` (class-stratified, seed 20260814); pools split
50/50; assessment risk, corrected oracle and denominator all computed **exclusively on
`D_assess`**. B = 1000, n ∈ {50, 100, 200, 500, 1000, 2000}, n_max = 2000, q threshold 0.9.

| pair | runs reaching n* | median n* where reached | monotone curves | median max q |
|---|---|---|---|---|
| ID→ID | 18/36 | 2000 | 32/36 | 0.879 |
| ID→OOD | **1/36** | 2000 | 8/36 | **0.216** |
| OOD→OOD | 21/36 | 1000 | 36/36 | 0.939 |
| mixed→OOD | **2/36** | 1500 | 19/36 | **0.301** |

Per the plan's operationalization, a non-monotone curve never defines n* by first crossing;
those runs are recorded as **not reached** with n_max stated. Monte-Carlo variation is
conditional on the fixed evaluation dataset.

## A12 — LOSO verdict stability [LE], δ = 0.10

12 cells; **7 stable, 5 unstable** under leave-one-seed-out:

| cell | majority | seed classes |
|---|---|---|
| c100n_ce | incompatible | inc, c-unsolved, inc |
| c100n_elr | compatible-solved | indet, inc, c-solved |
| c100n_sop | incompatible | inc, inc, c-unsolved |
| c10n_worst_ce | incompatible | c-unsolved, inc, inc |
| c10n_worst_elr | incompatible | inc, c-unsolved, inc |

Every unstable cell is a 2–1 split, so dropping the minority seed leaves a tie. Frozen
Phase-I cell verdicts are not recomputed.

## A13 — equivalence-style reanalysis [LE, SENSITIVITY]

Margin 0.10 on the mean ρ̂*_LE, BCa 95%: point **0.3161**, CI **[0.2422, 0.4084]**, upper
bound below margin: **no**. Descriptive; the frozen WEAK verdict is untouched.

## A14 — E prediction re-adjudication [LE], both granularities × both metric scales

| scale | per-run largest-regret axis (ID / WC / OOD) | aggregate medians (ID / WC / OOD) | aggregate largest |
|---|---|---|---|
| normalized regret | 0 / 10 / 26 | 0.0836 / 0.3089 / 1.2803 | OOD |
| epoch gap | 5 / 3 / 28 | 4.0 / 6.5 / 15.0 | OOD |

Both granularities and both scales are reported; neither is designated primary here.

## The pre-written branch sentences, quoted not applied

Plan §3 binds after A3 on Tier 1. The three sentences are reproduced verbatim so the
review side can apply the one its adjudication selects:

> **Incompatible-majority:** "trajectory incompatibility dominates; selector failure there
> is necessary; information content is adjudicated on the compatible minority."
> **Compatible-unsolved-majority:** "identifiability failure dominates; jointly adequate
> checkpoints exist but no registered source-only selector locates them; w_δ grades
> difficulty."
> **Mixed:** per-learner×dataset split reported with no reframing beyond these sentences.

Selecting among them is step (6) and is not done here.

## STOP

Steps (2) through (5) are complete. Step (6) adjudication — frozen and corrected — is
review-side. Tier 2 remains gated on the OOD-far pin, officially obtained Clothing1M, and a
separate instruction. A10b remains open and non-blocking.
