# T7 — Selector joint successes on compatible runs, exact intervals  [CORRECTED-DESCRIPTIVE]

Joint success = Ĵ_LE ≤ δ = 0.1 on a run classified compatible at that δ. Intervals are exact Clopper–Pearson at 95%. They are wide because the denominators are small; that is the point of showing them, since a bare 1/7 reads like a rate. Sources: `A2.per_delta.delta_0.1.taxonomy` for eligibility, `A4.selectors.<s>.J_LE` for success, `A5.per_delta.delta_0.1.p_unif` for the uniform baseline.

### Tier 1 — 7 compatible run(s)

| selector | successes | proportion | exact 95% CI |
|---|---|---|---|
| E(τ=1) | 1/7 | 0.143 | [0.004, 0.579] |
| NA | 1/7 | 0.143 | [0.004, 0.579] |
| ER-argmax | 2/7 | 0.286 | [0.037, 0.710] |
| LW-N *(reported, non-gating)* | 1/7 | 0.143 | [0.004, 0.579] |

Per-run uniform baseline on the same runs:

| run | w_δ = p_unif | E(τ=1) | NA | ER-argmax |
|---|---|---|---|---|
| `c100n_ce_seed1` | 0.042 | — | — | — |
| `c100n_elr_seed2` | 0.125 | — | — | hit |
| `c100n_gce_seed0` | 0.125 | — | hit | hit |
| `c100n_gce_seed2` | 0.042 | hit | — | — |
| `c100n_sop_seed2` | 0.083 | — | — | — |
| `c10n_worst_ce_seed0` | 0.042 | — | — | — |
| `c10n_worst_elr_seed1` | 0.042 | — | — | — |

### Phase II — 1 compatible run(s)

| selector | successes | proportion | exact 95% CI |
|---|---|---|---|
| E(τ=1) | 0/1 | 0.000 | [0.000, 0.975] |
| NA | 0/1 | 0.000 | [0.000, 0.975] |
| ER-argmax | 0/1 | 0.000 | [0.000, 0.975] |
| LW-N *(reported, non-gating)* | 0/1 | 0.000 | [0.000, 0.975] |

Per-run uniform baseline on the same runs:

| run | w_δ = p_unif | E(τ=1) | NA | ER-argmax |
|---|---|---|---|---|
| `cifar100_symmetric0.6_ce_seed2` | 0.042 | — | — | — |

