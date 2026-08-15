# A-T3 — OOD decomposition per (score | pool), Tier 1

Sources: `per_run[].A9.per_score_pool.<u|o>.{rho_star_LE,w_delta,ood_t_star_epoch,ood_zero_iqr_24}` over the 36 Tier-1 runs. Medians across runs are display aggregation; `runs with ρ̂* > δ` is a threshold count of the stored per-run values.

| score \| pool | median ρ̂*_LE | median w(0.10) | runs with ρ̂*_LE > 0.10 | median OOD oracle epoch | runs with zero 24-grid OOD IQR |
|---|---|---|---|---|---|
| `msp|svhn` | 0.3448 | 0.000 | 29/36 | 44 | 0/36 |
| `msp|cross_cifar` | 0.2051 | 0.000 | 24/36 | 59 | 0/36 |
| `msp|CIFAR-C-local` | 0.3956 | 0.000 | 32/36 | 59 | 0/36 |
| `energy|svhn` | 0.3544 | 0.000 | 31/36 | 36 | 0/36 |
| `energy|cross_cifar` | 0.2986 | 0.000 | 26/36 | 44 | 0/36 |
| `energy|CIFAR-C-local` | 0.4615 | 0.000 | 34/36 | 42 | 0/36 |

## Aggregation-variant taxonomy  [SENSITIVITY, post-adjudication addendum]

Registered in `docs/remediation_plan_v2.md` (A9-ADDENDUM) before computation and executed after that registration was anchored. Each variant replaces the OOD axis with the named aggregation over the semantic pools {svhn, cross_cifar}, gives it its own corrected 24-grid oracle and 24-epoch IQR denominator (fail-closed), holds ID and WC fixed, and re-derives the A3 taxonomy. **This cannot alter the anchored step-6 adjudication**, which stands on the primary aggregation.

### Phase II (G2-15)

| aggregation | δ | incompatible | compatible-solved | compatible-unsolved | indeterminate | unscorable |
|---|---|---|---|---|---|---|
| **energy-mean (PRIMARY)** | 0.05 | 14 | 0 | 1 | 0 | 0 |
| **energy-mean (PRIMARY)** | 0.1 | 14 | 0 | 1 | 0 | 0 |
| **energy-mean (PRIMARY)** | 0.2 | 10 | 0 | 3 | 2 | 0 |
| energy-min | 0.05 | 14 | 0 | 1 | 0 | 0 |
| energy-min | 0.1 | 14 | 0 | 1 | 0 | 0 |
| energy-min | 0.2 | 11 | 0 | 2 | 2 | 0 |
| energy-max | 0.05 | 14 | 0 | 1 | 0 | 0 |
| energy-max | 0.1 | 13 | 0 | 1 | 1 | 0 |
| energy-max | 0.2 | 7 | 0 | 4 | 4 | 0 |
| msp-mean | 0.05 | 14 | 0 | 1 | 0 | 0 |
| msp-mean | 0.1 | 14 | 0 | 1 | 0 | 0 |
| msp-mean | 0.2 | 11 | 0 | 2 | 2 | 0 |
| msp-min | 0.05 | 14 | 0 | 1 | 0 | 0 |
| msp-min | 0.1 | 13 | 0 | 1 | 1 | 0 |
| msp-min | 0.2 | 11 | 0 | 3 | 1 | 0 |
| msp-max | 0.05 | 13 | 0 | 2 | 0 | 0 |
| msp-max | 0.1 | 12 | 0 | 2 | 1 | 0 |
| msp-max | 0.2 | 8 | 0 | 5 | 2 | 0 |

n = 15 runs.

### Tier 1 (36)

| aggregation | δ | incompatible | compatible-solved | compatible-unsolved | indeterminate | unscorable |
|---|---|---|---|---|---|---|
| **energy-mean (PRIMARY)** | 0.05 | 29 | 0 | 2 | 5 | 0 |
| **energy-mean (PRIMARY)** | 0.1 | 27 | 3 | 4 | 2 | 0 |
| **energy-mean (PRIMARY)** | 0.2 | 19 | 8 | 4 | 5 | 0 |
| energy-min | 0.05 | 33 | 0 | 1 | 2 | 0 |
| energy-min | 0.1 | 31 | 1 | 2 | 2 | 0 |
| energy-min | 0.2 | 23 | 5 | 6 | 2 | 0 |
| energy-max | 0.05 | 27 | 1 | 2 | 6 | 0 |
| energy-max | 0.1 | 26 | 5 | 4 | 1 | 0 |
| energy-max | 0.2 | 19 | 9 | 5 | 3 | 0 |
| msp-mean | 0.05 | 31 | 1 | 3 | 1 | 0 |
| msp-mean | 0.1 | 27 | 3 | 2 | 4 | 0 |
| msp-mean | 0.2 | 21 | 7 | 5 | 3 | 0 |
| msp-min | 0.05 | 32 | 1 | 2 | 1 | 0 |
| msp-min | 0.1 | 29 | 2 | 2 | 3 | 0 |
| msp-min | 0.2 | 22 | 5 | 7 | 2 | 0 |
| msp-max | 0.05 | 26 | 2 | 5 | 3 | 0 |
| msp-max | 0.1 | 23 | 6 | 4 | 3 | 0 |
| msp-max | 0.2 | 17 | 11 | 7 | 1 | 0 |

n = 36 runs.

### Class flips against the primary aggregation

140 (run × variant × δ) cells change class relative to the primary energy-mean aggregation. Counts only; no interpretation.

| run | frame | variant | δ | primary class | variant class |
|---|---|---|---|---|---|
| `cifar10_asymmetric0.4_elr_seed0` | Phase II (G2-15) | energy-min | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | energy-min | 0.2 | indeterminate | incompatible |
| `cifar10_symmetric0.2_elr_seed1` | Phase II (G2-15) | energy-min | 0.2 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | energy-max | 0.1 | incompatible | indeterminate |
| `cifar100_symmetric0.6_ce_seed1` | Phase II (G2-15) | energy-max | 0.2 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_ce_seed2` | Phase II (G2-15) | energy-max | 0.2 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_elr_seed0` | Phase II (G2-15) | energy-max | 0.2 | compatible-unsolved | indeterminate |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | energy-max | 0.2 | indeterminate | compatible-unsolved |
| `cifar10_symmetric0.2_elr_seed0` | Phase II (G2-15) | energy-max | 0.2 | incompatible | compatible-unsolved |
| `cifar100_symmetric0.6_ce_seed1` | Phase II (G2-15) | msp-mean | 0.2 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_ce_seed0` | Phase II (G2-15) | msp-mean | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed0` | Phase II (G2-15) | msp-mean | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | msp-mean | 0.2 | indeterminate | incompatible |
| `cifar10_symmetric0.2_elr_seed1` | Phase II (G2-15) | msp-mean | 0.2 | incompatible | compatible-unsolved |
| `cifar10_symmetric0.2_elr_seed2` | Phase II (G2-15) | msp-min | 0.1 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_ce_seed0` | Phase II (G2-15) | msp-min | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed0` | Phase II (G2-15) | msp-min | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | msp-min | 0.2 | indeterminate | incompatible |
| `cifar10_symmetric0.2_elr_seed1` | Phase II (G2-15) | msp-min | 0.2 | incompatible | compatible-unsolved |
| `cifar10_symmetric0.2_elr_seed2` | Phase II (G2-15) | msp-min | 0.2 | incompatible | compatible-unsolved |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | msp-max | 0.05 | incompatible | compatible-unsolved |
| `cifar100_symmetric0.6_ce_seed0` | Phase II (G2-15) | msp-max | 0.1 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | msp-max | 0.1 | incompatible | compatible-unsolved |
| `cifar100_symmetric0.4_ce_seed2` | Phase II (G2-15) | msp-max | 0.2 | incompatible | compatible-unsolved |
| `cifar100_symmetric0.6_ce_seed0` | Phase II (G2-15) | msp-max | 0.2 | incompatible | compatible-unsolved |
| `cifar100_symmetric0.6_ce_seed1` | Phase II (G2-15) | msp-max | 0.2 | incompatible | indeterminate |
| `cifar10_asymmetric0.4_ce_seed0` | Phase II (G2-15) | msp-max | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed0` | Phase II (G2-15) | msp-max | 0.2 | compatible-unsolved | incompatible |
| `cifar10_asymmetric0.4_elr_seed2` | Phase II (G2-15) | msp-max | 0.2 | indeterminate | compatible-unsolved |
| `cifar10_symmetric0.2_elr_seed0` | Phase II (G2-15) | msp-max | 0.2 | incompatible | compatible-unsolved |
| `c100n_ce_seed1` | Tier 1 (36) | energy-min | 0.05 | indeterminate | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | energy-min | 0.05 | compatible-unsolved | indeterminate |
| `c100n_gce_seed0` | Tier 1 (36) | energy-min | 0.05 | indeterminate | incompatible |
| `c100n_sop_seed2` | Tier 1 (36) | energy-min | 0.05 | indeterminate | incompatible |
| `c10n_worst_ce_seed0` | Tier 1 (36) | energy-min | 0.05 | indeterminate | incompatible |
| `c100n_ce_seed1` | Tier 1 (36) | energy-min | 0.1 | compatible-unsolved | incompatible |
| `c100n_elr_seed0` | Tier 1 (36) | energy-min | 0.1 | indeterminate | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | energy-min | 0.1 | compatible-solved | compatible-unsolved |
| `c100n_gce_seed0` | Tier 1 (36) | energy-min | 0.1 | compatible-solved | indeterminate |
| `c100n_sop_seed2` | Tier 1 (36) | energy-min | 0.1 | compatible-unsolved | incompatible |
| `c10n_worst_ce_seed0` | Tier 1 (36) | energy-min | 0.1 | compatible-unsolved | incompatible |
| `c100n_ce_seed1` | Tier 1 (36) | energy-min | 0.2 | compatible-solved | incompatible |
| `c100n_elr_seed0` | Tier 1 (36) | energy-min | 0.2 | compatible-solved | compatible-unsolved |
| `c100n_elr_seed1` | Tier 1 (36) | energy-min | 0.2 | indeterminate | incompatible |
| `c100n_sop_seed0` | Tier 1 (36) | energy-min | 0.2 | indeterminate | incompatible |
| `c100n_sop_seed1` | Tier 1 (36) | energy-min | 0.2 | compatible-solved | compatible-unsolved |
| `c10n_random1_gce_seed1` | Tier 1 (36) | energy-min | 0.2 | incompatible | compatible-unsolved |
| `c10n_worst_ce_seed0` | Tier 1 (36) | energy-min | 0.2 | compatible-unsolved | incompatible |
| `c10n_worst_ce_seed1` | Tier 1 (36) | energy-min | 0.2 | indeterminate | incompatible |
| `c100n_elr_seed0` | Tier 1 (36) | energy-max | 0.05 | incompatible | indeterminate |
| `c100n_gce_seed0` | Tier 1 (36) | energy-max | 0.05 | indeterminate | incompatible |
| `c100n_sop_seed0` | Tier 1 (36) | energy-max | 0.05 | incompatible | compatible-solved |
| `c100n_sop_seed1` | Tier 1 (36) | energy-max | 0.05 | incompatible | indeterminate |
| `c10n_worst_ce_seed0` | Tier 1 (36) | energy-max | 0.05 | indeterminate | incompatible |
| `c10n_worst_ce_seed1` | Tier 1 (36) | energy-max | 0.05 | incompatible | compatible-unsolved |
| `c10n_worst_elr_seed1` | Tier 1 (36) | energy-max | 0.05 | compatible-unsolved | indeterminate |
| `c100n_ce_seed1` | Tier 1 (36) | energy-max | 0.1 | compatible-unsolved | compatible-solved |
| `c100n_ce_seed2` | Tier 1 (36) | energy-max | 0.1 | incompatible | indeterminate |
| `c100n_elr_seed0` | Tier 1 (36) | energy-max | 0.1 | indeterminate | compatible-solved |
| `c100n_elr_seed2` | Tier 1 (36) | energy-max | 0.1 | compatible-solved | compatible-unsolved |
| `c100n_gce_seed0` | Tier 1 (36) | energy-max | 0.1 | compatible-solved | incompatible |
| `c100n_sop_seed0` | Tier 1 (36) | energy-max | 0.1 | incompatible | compatible-solved |
| `c100n_sop_seed1` | Tier 1 (36) | energy-max | 0.1 | incompatible | compatible-unsolved |
| `c100n_sop_seed2` | Tier 1 (36) | energy-max | 0.1 | compatible-unsolved | compatible-solved |
| `c10n_random1_sop_seed0` | Tier 1 (36) | energy-max | 0.1 | indeterminate | incompatible |
| `c10n_worst_ce_seed0` | Tier 1 (36) | energy-max | 0.1 | compatible-unsolved | incompatible |
| `c10n_worst_ce_seed1` | Tier 1 (36) | energy-max | 0.1 | incompatible | compatible-unsolved |
| `c100n_ce_seed2` | Tier 1 (36) | energy-max | 0.2 | incompatible | compatible-solved |
| `c100n_gce_seed0` | Tier 1 (36) | energy-max | 0.2 | compatible-solved | indeterminate |
| `c100n_sop_seed0` | Tier 1 (36) | energy-max | 0.2 | indeterminate | compatible-solved |
| `c10n_random1_ce_seed2` | Tier 1 (36) | energy-max | 0.2 | indeterminate | incompatible |
| `c10n_random1_elr_seed0` | Tier 1 (36) | energy-max | 0.2 | incompatible | compatible-unsolved |
| `c10n_random1_sop_seed0` | Tier 1 (36) | energy-max | 0.2 | compatible-unsolved | incompatible |
| `c10n_worst_ce_seed1` | Tier 1 (36) | energy-max | 0.2 | indeterminate | compatible-unsolved |
| `c10n_worst_ce_seed2` | Tier 1 (36) | energy-max | 0.2 | incompatible | indeterminate |
| `c10n_worst_gce_seed0` | Tier 1 (36) | energy-max | 0.2 | indeterminate | incompatible |
| `c100n_ce_seed1` | Tier 1 (36) | msp-mean | 0.05 | indeterminate | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | msp-mean | 0.05 | compatible-unsolved | incompatible |
| `c100n_gce_seed0` | Tier 1 (36) | msp-mean | 0.05 | indeterminate | compatible-solved |
| `c100n_sop_seed2` | Tier 1 (36) | msp-mean | 0.05 | indeterminate | compatible-unsolved |
| `c10n_worst_ce_seed0` | Tier 1 (36) | msp-mean | 0.05 | indeterminate | compatible-unsolved |
| `c100n_ce_seed1` | Tier 1 (36) | msp-mean | 0.1 | compatible-unsolved | incompatible |
| `c100n_elr_seed0` | Tier 1 (36) | msp-mean | 0.1 | indeterminate | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | msp-mean | 0.1 | compatible-solved | indeterminate |
| `c100n_sop_seed1` | Tier 1 (36) | msp-mean | 0.1 | incompatible | indeterminate |
| `c100n_sop_seed2` | Tier 1 (36) | msp-mean | 0.1 | compatible-unsolved | compatible-solved |
| `c10n_random1_gce_seed1` | Tier 1 (36) | msp-mean | 0.1 | incompatible | indeterminate |
| `c100n_ce_seed1` | Tier 1 (36) | msp-mean | 0.2 | compatible-solved | indeterminate |
| `c100n_elr_seed0` | Tier 1 (36) | msp-mean | 0.2 | compatible-solved | incompatible |
| `c100n_elr_seed1` | Tier 1 (36) | msp-mean | 0.2 | indeterminate | compatible-solved |
| `c10n_random1_ce_seed2` | Tier 1 (36) | msp-mean | 0.2 | indeterminate | incompatible |
| `c10n_random1_gce_seed1` | Tier 1 (36) | msp-mean | 0.2 | incompatible | compatible-unsolved |
| `c10n_random1_sop_seed2` | Tier 1 (36) | msp-mean | 0.2 | incompatible | indeterminate |
| `c10n_worst_ce_seed1` | Tier 1 (36) | msp-mean | 0.2 | indeterminate | incompatible |
| `c10n_worst_gce_seed0` | Tier 1 (36) | msp-mean | 0.2 | indeterminate | incompatible |
| `c100n_ce_seed1` | Tier 1 (36) | msp-min | 0.05 | indeterminate | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | msp-min | 0.05 | compatible-unsolved | incompatible |
| `c100n_gce_seed0` | Tier 1 (36) | msp-min | 0.05 | indeterminate | compatible-solved |
| `c100n_sop_seed2` | Tier 1 (36) | msp-min | 0.05 | indeterminate | compatible-unsolved |
| `c10n_worst_ce_seed0` | Tier 1 (36) | msp-min | 0.05 | indeterminate | incompatible |
| `c100n_ce_seed1` | Tier 1 (36) | msp-min | 0.1 | compatible-unsolved | incompatible |
| `c100n_elr_seed0` | Tier 1 (36) | msp-min | 0.1 | indeterminate | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | msp-min | 0.1 | compatible-solved | incompatible |
| `c10n_random1_gce_seed1` | Tier 1 (36) | msp-min | 0.1 | incompatible | indeterminate |
| `c10n_worst_ce_seed0` | Tier 1 (36) | msp-min | 0.1 | compatible-unsolved | indeterminate |
| `c100n_ce_seed1` | Tier 1 (36) | msp-min | 0.2 | compatible-solved | incompatible |
| `c100n_elr_seed0` | Tier 1 (36) | msp-min | 0.2 | compatible-solved | incompatible |
| `c100n_elr_seed2` | Tier 1 (36) | msp-min | 0.2 | compatible-solved | compatible-unsolved |
| `c100n_sop_seed0` | Tier 1 (36) | msp-min | 0.2 | indeterminate | incompatible |
| `c10n_random1_ce_seed2` | Tier 1 (36) | msp-min | 0.2 | indeterminate | incompatible |
| `c10n_random1_elr_seed1` | Tier 1 (36) | msp-min | 0.2 | incompatible | compatible-unsolved |
| `c10n_random1_gce_seed1` | Tier 1 (36) | msp-min | 0.2 | incompatible | compatible-unsolved |
| `c10n_worst_ce_seed1` | Tier 1 (36) | msp-min | 0.2 | indeterminate | incompatible |
| `c100n_ce_seed1` | Tier 1 (36) | msp-max | 0.05 | indeterminate | compatible-solved |
| `c100n_elr_seed2` | Tier 1 (36) | msp-max | 0.05 | compatible-unsolved | indeterminate |
| `c100n_gce_seed0` | Tier 1 (36) | msp-max | 0.05 | indeterminate | compatible-unsolved |
| `c100n_sop_seed0` | Tier 1 (36) | msp-max | 0.05 | incompatible | compatible-solved |
| `c100n_sop_seed1` | Tier 1 (36) | msp-max | 0.05 | incompatible | indeterminate |
| `c100n_sop_seed2` | Tier 1 (36) | msp-max | 0.05 | indeterminate | compatible-unsolved |
| `c10n_worst_ce_seed0` | Tier 1 (36) | msp-max | 0.05 | indeterminate | compatible-unsolved |
| `c10n_worst_ce_seed1` | Tier 1 (36) | msp-max | 0.05 | incompatible | compatible-unsolved |
| `c100n_ce_seed1` | Tier 1 (36) | msp-max | 0.1 | compatible-unsolved | compatible-solved |
| `c100n_ce_seed2` | Tier 1 (36) | msp-max | 0.1 | incompatible | indeterminate |
| `c100n_gce_seed1` | Tier 1 (36) | msp-max | 0.1 | incompatible | indeterminate |
| `c100n_sop_seed0` | Tier 1 (36) | msp-max | 0.1 | incompatible | compatible-solved |
| `c100n_sop_seed1` | Tier 1 (36) | msp-max | 0.1 | incompatible | compatible-unsolved |
| `c100n_sop_seed2` | Tier 1 (36) | msp-max | 0.1 | compatible-unsolved | compatible-solved |
| `c10n_random1_sop_seed0` | Tier 1 (36) | msp-max | 0.1 | indeterminate | incompatible |
| `c10n_worst_ce_seed1` | Tier 1 (36) | msp-max | 0.1 | incompatible | compatible-unsolved |
| `c100n_ce_seed2` | Tier 1 (36) | msp-max | 0.2 | incompatible | compatible-solved |
| `c100n_elr_seed1` | Tier 1 (36) | msp-max | 0.2 | indeterminate | compatible-solved |
| `c100n_sop_seed0` | Tier 1 (36) | msp-max | 0.2 | indeterminate | compatible-solved |
| `c10n_random1_ce_seed2` | Tier 1 (36) | msp-max | 0.2 | indeterminate | incompatible |
| `c10n_random1_elr_seed0` | Tier 1 (36) | msp-max | 0.2 | incompatible | compatible-unsolved |
| `c10n_random1_elr_seed2` | Tier 1 (36) | msp-max | 0.2 | incompatible | compatible-unsolved |
| `c10n_random1_sop_seed0` | Tier 1 (36) | msp-max | 0.2 | compatible-unsolved | incompatible |
| `c10n_worst_ce_seed1` | Tier 1 (36) | msp-max | 0.2 | indeterminate | compatible-unsolved |
| `c10n_worst_gce_seed0` | Tier 1 (36) | msp-max | 0.2 | indeterminate | incompatible |
| `c10n_worst_gce_seed1` | Tier 1 (36) | msp-max | 0.2 | incompatible | compatible-unsolved |
| `c10n_worst_sop_seed1` | Tier 1 (36) | msp-max | 0.2 | incompatible | indeterminate |

Produced at `git_head` 78727a1, `git_tree_dirty` False.
