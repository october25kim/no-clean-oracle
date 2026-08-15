# A-T1 — per-run compatibility summary, all 51 audited runs

Sources: `per_run[].A2.rho_star_LE`, `per_run[].A2.per_delta.delta_<d>.{w_delta,taxonomy}` from `battery_g2.json` and `battery_tier1.json`. Class is at δ = 0.10. No value is recomputed.

## Phase II (G2-15)

| run | ρ̂*_LE | w(0.05) | w(0.10) | w(0.20) | class (δ=0.10) |
|---|---|---|---|---|---|
| `cifar100_symmetric0.6_ce_seed2` | 0.0000 | 0.042 | 0.042 | 0.042 | compatible-unsolved |
| `cifar10_asymmetric0.4_elr_seed0` | 0.1384 | 0.000 | 0.000 | 0.042 | incompatible |
| `cifar10_asymmetric0.4_ce_seed0` | 0.1599 | 0.000 | 0.000 | 0.042 | incompatible |
| `cifar100_symmetric0.4_ce_seed1` | 0.1879 | 0.000 | 0.000 | 0.042 | incompatible |
| `cifar10_asymmetric0.4_elr_seed2` | 0.1988 | 0.000 | 0.000 | 0.042 | incompatible |
| `cifar10_symmetric0.2_elr_seed2` | 0.2522 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar10_symmetric0.2_elr_seed1` | 0.2534 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar10_asymmetric0.4_elr_seed1` | 0.3422 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar10_symmetric0.2_elr_seed0` | 0.3963 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar10_asymmetric0.4_ce_seed2` | 0.4875 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar100_symmetric0.4_ce_seed0` | 0.4883 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar100_symmetric0.6_ce_seed1` | 0.5475 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar10_asymmetric0.4_ce_seed1` | 0.5670 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar100_symmetric0.4_ce_seed2` | 0.8750 | 0.000 | 0.000 | 0.000 | incompatible |
| `cifar100_symmetric0.6_ce_seed0` | 1.0014 | 0.000 | 0.000 | 0.000 | incompatible |

## Tier 1 (36)

| run | ρ̂*_LE | w(0.05) | w(0.10) | w(0.20) | class (δ=0.10) |
|---|---|---|---|---|---|
| `c100n_elr_seed2` | 0.0000 | 0.042 | 0.125 | 0.208 | compatible-solved |
| `c10n_worst_elr_seed1` | 0.0205 | 0.042 | 0.042 | 0.042 | compatible-unsolved |
| `c100n_gce_seed2` | 0.0355 | 0.042 | 0.042 | 0.042 | compatible-solved |
| `c10n_worst_ce_seed0` | 0.0590 | 0.000 | 0.042 | 0.042 | compatible-unsolved |
| `c100n_ce_seed1` | 0.0596 | 0.000 | 0.042 | 0.250 | compatible-unsolved |
| `c100n_sop_seed2` | 0.0676 | 0.000 | 0.083 | 0.167 | compatible-unsolved |
| `c100n_gce_seed0` | 0.0695 | 0.000 | 0.125 | 0.167 | compatible-solved |
| `c10n_random1_sop_seed0` | 0.0804 | 0.000 | 0.042 | 0.083 | indeterminate |
| `c100n_elr_seed0` | 0.0970 | 0.000 | 0.042 | 0.167 | indeterminate |
| `c100n_sop_seed1` | 0.1286 | 0.000 | 0.000 | 0.125 | incompatible |
| `c100n_gce_seed1` | 0.1310 | 0.000 | 0.000 | 0.042 | incompatible |
| `c100n_ce_seed0` | 0.1337 | 0.000 | 0.000 | 0.125 | incompatible |
| `c10n_worst_ce_seed1` | 0.1757 | 0.000 | 0.000 | 0.042 | incompatible |
| `c100n_elr_seed1` | 0.1891 | 0.000 | 0.000 | 0.042 | incompatible |
| `c100n_sop_seed0` | 0.1901 | 0.000 | 0.000 | 0.042 | incompatible |
| `c10n_random1_ce_seed2` | 0.1949 | 0.000 | 0.000 | 0.042 | incompatible |
| `c10n_worst_gce_seed0` | 0.2005 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_elr_seed0` | 0.2371 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_ce_seed0` | 0.2645 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_sop_seed0` | 0.3281 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_elr_seed1` | 0.3353 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_gce_seed1` | 0.3714 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_elr_seed0` | 0.3797 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_elr_seed2` | 0.3848 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_sop_seed1` | 0.3850 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_elr_seed2` | 0.3893 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_gce_seed1` | 0.3915 | 0.000 | 0.000 | 0.000 | incompatible |
| `c100n_ce_seed2` | 0.4047 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_ce_seed1` | 0.5159 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_ce_seed2` | 0.5193 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_gce_seed2` | 0.6025 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_gce_seed2` | 0.6443 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_sop_seed2` | 0.7148 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_sop_seed1` | 0.7242 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_worst_sop_seed2` | 0.8387 | 0.000 | 0.000 | 0.000 | incompatible |
| `c10n_random1_gce_seed0` | 1.1149 | 0.000 | 0.000 | 0.000 | incompatible |

