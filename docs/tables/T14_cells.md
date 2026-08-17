# T14 — Cell-level summary, Tier 1 (12 cells × 3 seeds)  [CORRECTED-DESCRIPTIVE]

Class counts at δ = 0.1 within each (split × learner) cell, the LOSO stability flag from A12, and the margin of each seed's ρ̂*_LE from δ — negative means below the threshold. Sources: `A2.per_delta.delta_0.1.taxonomy`, `A2.rho_star_LE`, `A12.cells.*`.

| cell | incompatible | compat-solved | compat-unsolved | indeterminate | majority | LOSO stable | ρ̂*−δ per seed |
|---|---|---|---|---|---|---|---|
| `c100n_ce` | 2 | 0 | 1 | 0 | incompatible | **no** | +0.034, -0.040, +0.305 |
| `c100n_elr` | 1 | 1 | 0 | 1 | compatible-solved | **no** | -0.003, +0.089, -0.100 |
| `c100n_gce` | 1 | 2 | 0 | 0 | compatible-solved | yes | -0.030, +0.031, -0.065 |
| `c100n_sop` | 2 | 0 | 1 | 0 | incompatible | **no** | +0.090, +0.029, -0.032 |
| `c10n_random1_ce` | 3 | 0 | 0 | 0 | incompatible | yes | +0.165, +0.416, +0.095 |
| `c10n_random1_elr` | 3 | 0 | 0 | 0 | incompatible | yes | +0.137, +0.235, +0.285 |
| `c10n_random1_gce` | 3 | 0 | 0 | 0 | incompatible | yes | +1.015, +0.291, +0.503 |
| `c10n_random1_sop` | 2 | 0 | 0 | 1 | incompatible | yes | -0.020, +0.285, +0.615 |
| `c10n_worst_ce` | 2 | 0 | 1 | 0 | incompatible | **no** | -0.041, +0.076, +0.419 |
| `c10n_worst_elr` | 2 | 0 | 1 | 0 | incompatible | **no** | +0.280, -0.080, +0.289 |
| `c10n_worst_gce` | 3 | 0 | 0 | 0 | incompatible | yes | +0.100, +0.271, +0.544 |
| `c10n_worst_sop` | 3 | 0 | 0 | 0 | incompatible | yes | +0.228, +0.624, +0.739 |

### Margins by learner and by split

| grouping | n runs | median ρ̂*−δ | min | max |
|---|---|---|---|---|
| learner `ce` | 9 | +0.095 | -0.041 | +0.419 |
| learner `elr` | 9 | +0.137 | -0.100 | +0.289 |
| learner `gce` | 9 | +0.271 | -0.065 | +1.015 |
| learner `sop` | 9 | +0.228 | -0.032 | +0.739 |
| split `c100n` | 12 | +0.013 | -0.100 | +0.305 |
| split `c10n_random1` | 12 | +0.285 | -0.020 | +1.015 |
| split `c10n_worst` | 12 | +0.276 | -0.080 | +0.739 |

