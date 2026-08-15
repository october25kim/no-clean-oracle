# REPORT 27 — MILESTONE-G2: corrected battery A1–A9 + A12 on the 15 audited runs

**Project** G1-NoCleanOracle · **Session** ext_realnoise (DATA-PRODUCING QUARANTINED WORKER)
**Frame** G2-15 · **Issued** 2026-08-15 · **Artifact** `results/corrected/battery_g2.json`
**Code** internal `2d9c43b`, anchored at public `cd829f3` *before* this execution (safeguard b)

Every number below is tagged with its estimand layer. **LE** = empirical lower envelope
(spec A.4), **CF** = cross-fitted selection benchmark, **FROZEN-HISTORICAL** = the
registered Phase-I/II record, untouched. `eta` is only ever formed from two LE terms.

Spec G.2 forbidden phrases bind this prose. **No adjudication is offered**: the §3 branch
sentences bind after A3 *on Tier 1*, not here, and step (6) is review-side.

## Integrity

| check | R4 class | observed | bar | verdict |
|---|---|---|---|---|
| recomputed vs logged `R_ID`, 360 checkpoint-instances | **internal-consistency** | **1.110e-16** | ≤ 1e-12 | PASS |
| `code_stamp` | — | `git_tree_dirty: false`, head `2d9c43b` | — | PASS |

The instruction anticipated trajectory-identity with an exact match. It is
internal-consistency, and the reason is visible in one line of arithmetic: the battery
forms `1 − mean(correct)` while the logged value was `mean(pred != y)`. Those are the same
quantity by algebra and different by float summation order. The forward pass, which walks
the *same* path as the training-time evaluation, did return exactly 0 on all 864
checkpoints; this check does not, because it deliberately takes a different aggregation
route in order to be able to restrict to folds and subsamples.

**Canonical integrity at this milestone:** data hygiene exit 0 · 24 baselined masks
unchanged · R5 holds across 19 modules, zero drift · SEALED markers 2/2 · G1 48 runs and
B2 15 runs unchanged.

## A1 — grid and definition coherence [LE vs FROZEN-HISTORICAL]

Each side is computed as originally defined; neither is re-denominated.

| axis | identical oracle epoch | median \|Δepoch\| | max \|Δepoch\| |
|---|---|---|---|
| ID | 1/15 | 3.0 | 22 |
| OOD | 2/15 | 1.0 | 78 |

WC has no frozen-historical counterpart: the frozen frame's `tail` axis was
`R_tail_static` over 120 points, and the corrected axis is the same class set on the
24-point grid, so the comparison would be between two different objects rather than two
readings of one.

Zero-IQR exclusions on the corrected 24-point grid: **0 of 45 axis-instances**. All 15 runs
are scorable.

## A2 / A3 — compatibility and taxonomy [LE]

| run | ρ̂*_LE | w_0.05 | w_0.10 | w_0.20 | class (δ=0.10) |
|---|---|---|---|---|---|
| cifar100_symmetric0.4_ce_seed0 | 0.4883 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar100_symmetric0.4_ce_seed1 | 0.1879 | 0.000 | 0.000 | 0.042 | incompatible |
| cifar100_symmetric0.4_ce_seed2 | 0.8750 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar100_symmetric0.6_ce_seed0 | 1.0014 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar100_symmetric0.6_ce_seed1 | 0.5475 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar100_symmetric0.6_ce_seed2 | 0.0000 | 0.042 | 0.042 | 0.042 | compatible-unsolved |
| cifar10_asymmetric0.4_ce_seed0 | 0.1599 | 0.000 | 0.000 | 0.042 | incompatible |
| cifar10_asymmetric0.4_ce_seed1 | 0.5670 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar10_asymmetric0.4_ce_seed2 | 0.4875 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar10_asymmetric0.4_elr_seed0 | 0.1384 | 0.000 | 0.000 | 0.042 | incompatible |
| cifar10_asymmetric0.4_elr_seed1 | 0.3422 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar10_asymmetric0.4_elr_seed2 | 0.1988 | 0.000 | 0.000 | 0.042 | incompatible |
| cifar10_symmetric0.2_elr_seed0 | 0.3963 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar10_symmetric0.2_elr_seed1 | 0.2534 | 0.000 | 0.000 | 0.000 | incompatible |
| cifar10_symmetric0.2_elr_seed2 | 0.2522 | 0.000 | 0.000 | 0.000 | incompatible |

`w_δ = 0.042` is one retained checkpoint out of 24.

### Decomposition of the historical 0/45 [FROZEN-HISTORICAL → LE]

| δ | incompatible | compatible-solved | compatible-unsolved | indeterminate |
|---|---|---|---|---|
| 0.05 | 14 | 0 | 1 | 0 |
| **0.10** | **14** | **0** | **1** | **0** |
| 0.20 | 10 | 0 | 3 | 2 |

Per-selector joint successes **on compatible runs only** (δ=0.10, 1 compatible run):

| registered selector | joint successes |
|---|---|
| E(τ=1) | 0/1 |
| NA | 0/1 |
| ER-argmax | 0/1 |

## A4 — selector accounting [LE], medians over 15 runs

| selector | median Ĵ_LE | median η̂_LE | min η̂_LE | gates taxonomy |
|---|---|---|---|---|
| E(τ=1) | 2.2602 | 1.9508 | +4.89e-01 | yes |
| NA | 2.2456 | 1.9508 | +4.82e-01 | yes |
| ER-argmax | 2.6518 | 2.4436 | +1.60e+00 | yes |
| ER-argmin (sensitivity) | 3.4371 | 3.1405 | +4.43e-01 | no |
| LW-N | 2.2602 | 1.9508 | +5.04e-01 | **no — reported only** |

η̂ is non-negative in every cell, as spec D.1 requires within a single layer.

## A5 — exact chance baselines [LE]

No simulation and no binomial tail. `p_unif = w_δ` exactly: **0.000 in 14 runs, 0.042 in
1** at δ=0.10. Expected uniform joint regret `mean_t max_a ĝ_a(t)`: median **1.9214**,
range **[1.3519, 4.9713]**.

## A6 — cross-fitted benchmarks [CF], sign-free

K_CV = 5, folds fixed once per dataset (ID/WC class-stratified, pools seeded random),
seed 20260814. Reported as benchmark excesses with no non-negativity claim:
`E(τ=1) Ĵ_LE − CF_minimax` has median **+1.9060** and is negative in **0/15** runs.

## A7 — set-valued oracles [LE, SENSITIVITY], ε in absolute raw-risk units

| ε | median \|O_ID(ε)\| | median \|O_WC(ε)\| | median \|O_OOD(ε)\| |
|---|---|---|---|
| 0.005 | 2 | 1 | 1 |
| 0.01 | 3 | 1 | 1 |
| 0.02 | 4 | 2 | 4 |

Per-run CR ranges over each ε-optimal set are in the artifact. No A3 class flipped under
any ε in the grid.

## A8 — normalization sensitivity [LE, SENSITIVITY]

| floor ε_a | incompatible | compatible-unsolved |
|---|---|---|
| 1e-4 | 14 | 1 |
| 1e-3 | 14 | 1 |
| 1e-2 | 14 | 1 |

The taxonomy is unchanged across the whole floor grid, which is expected here because no
axis was fail-closed in the first place. Oracle-relative normalization is computable on
**45/45** axes (all have R̂* > 0.01). Raw Δ and range-normalized maxima are per-run in the
artifact.

## A9 — OOD decomposition [LE]

Per (score u ∈ {msp, energy}) × (pool o ∈ {svhn, cross_cifar, CIFAR-C-local}): corrected
oracle, ρ̂*_LE, F_δ and w_δ with ID and WC held fixed, plus aggregation sensitivity over
{primary mean, min, max, per-pool}. Full tables in the artifact.

**Two-world certificate line** (pin 13 addendum): pool pairs whose OOD-axis feasible sets
are disjoint — **0 disjoint pairs, 0/15 certificate-bearing runs**.

### D-7 firing audit, both sides — the two grids never mix

| side | grid | result |
|---|---|---|
| historical guard (`selection.py:31`) | 120-point IQR, as computed at the time | **fired 0 times** across 45 axis-instances; smallest 120-point IQR observed **0.02233** |
| corrected exclusion | 24-epoch IQR, fail-closed | **0 exclusions** across 45 axis-instances |

D-7 is a real defect in the code — a zero scale would have been scored as a perfect 0.0 —
but on this frame it never had the opportunity to fire. The defect ledger entry is updated
in the same commit as this analysis, per the plan's §5 rule.

## A12 — LOSO verdict stability [LE], δ=0.10

| cell | majority class | stable under leave-one-seed-out |
|---|---|---|
| cifar100_symmetric_0.4_ce | incompatible | yes |
| cifar100_symmetric_0.6_ce | incompatible | **no** |
| cifar10_asymmetric_0.4_ce | incompatible | yes |
| cifar10_asymmetric_0.4_elr | incompatible | yes |
| cifar10_symmetric_0.2_elr | incompatible | yes |

The unstable cell is the one containing the single compatible run; dropping either of its
two incompatible seeds leaves a 1–1 split.

## A13 — equivalence-style reanalysis [LE, SENSITIVITY]

Margin 0.10 on the mean ρ̂*_LE, BCa 95%: point **0.3931**, CI **[0.2828, 0.5607]**, upper
bound below margin: **no**. Descriptive only; the frozen WEAK verdict is untouched.

## What this milestone does and does not establish

It establishes the corrected decomposition of a frozen count, the layer-tagged tables
above, and that both D-7 audit sides came back empty on this frame. It does not adjudicate
anything: the branch sentences in plan §3 bind after A3 on Tier 1, and step (6) is
review-side. The Tier-1 selector outputs remain sealed at the time of writing, which is
the condition under which spec E.1 lets them count as prospective rather than as an
extended audit.
