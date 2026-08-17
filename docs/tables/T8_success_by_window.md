# T8 — Compatible runs stratified by feasible-window width  [CORRECTED-DESCRIPTIVE]

w_δ at δ = 0.1 is the fraction of the 24 retained checkpoints that are jointly adequate, so a stratum is a difficulty band: a run with one adequate checkpoint out of 24 is a harder target than one with several. Sources: `A2.per_delta.delta_0.1.{w_delta,taxonomy}` and `A4.selectors.<s>.J_LE`.

### Tier 1 — 7 compatible run(s), 3 stratum/strata

| w_δ | ≈ checkpoints of 24 | runs | E(τ=1) | NA | ER-argmax | LW-N |
|---|---|---|---|---|---|---|
| 0.042 | 1 | 4 | 1/4 | 0/4 | 0/4 | 0/4 |
| 0.083 | 2 | 1 | 0/1 | 0/1 | 0/1 | 0/1 |
| 0.125 | 3 | 2 | 0/2 | 1/2 | 2/2 | 1/2 |

Run membership per stratum:

- **w_δ = 0.042** — `c100n_ce_seed1`, `c100n_gce_seed2`, `c10n_worst_ce_seed0`, `c10n_worst_elr_seed1`
- **w_δ = 0.083** — `c100n_sop_seed2`
- **w_δ = 0.125** — `c100n_elr_seed2`, `c100n_gce_seed0`

### Phase II — 1 compatible run(s), 1 stratum/strata

| w_δ | ≈ checkpoints of 24 | runs | E(τ=1) | NA | ER-argmax | LW-N |
|---|---|---|---|---|---|---|
| 0.042 | 1 | 1 | 0/1 | 0/1 | 0/1 | 0/1 |

Run membership per stratum:

- **w_δ = 0.042** — `cifar100_symmetric0.6_ce_seed2`

