# T16 — Proposition 2 empirical premise check  [CORRECTED-DESCRIPTIVE]

Per run: the **ID empirical oracle set** (argmin of ĝ_ID with ties, so every epoch
attaining the minimum) against the OOD feasible set F̂_OOD(δ = 0.10) = {t : ĝ_OOD(t) ≤ δ}.
Proposition 2 concerns what happens when selecting on ID lands outside the OOD-adequate
region; this table reports only which of the four relations each run exhibits. Pure
lookup from `A1.axes.{ID,OOD}.ghat`; no estimand is introduced.

| relation | count |
|---|---|
| disjoint | 29/36 |
| partial overlap | 0/36 |
| contained | 7/36 |
| F_OOD empty | 0/36 |

| run | ID oracle set (epochs) | F̂_OOD(0.10) (epochs) | ∩ | relation |
|---|---|---|---|---|
| `c100n_ce_seed0` | [109] | [99, 109, 114, 119] | 1 | contained |
| `c100n_ce_seed1` | [104] | [94] | 0 | disjoint |
| `c100n_ce_seed2` | [119] | [29] | 0 | disjoint |
| `c100n_elr_seed0` | [119] | [59, 74, 84, 109] | 0 | disjoint |
| `c100n_elr_seed1` | [94] | [44, 114, 119] | 0 | disjoint |
| `c100n_elr_seed2` | [99] | [94, 99, 114] | 1 | contained |
| `c100n_gce_seed0` | [109] | [59, 109, 114, 119] | 1 | contained |
| `c100n_gce_seed1` | [104] | [89] | 0 | disjoint |
| `c100n_gce_seed2` | [119] | [89] | 0 | disjoint |
| `c100n_sop_seed0` | [119] | [84] | 0 | disjoint |
| `c100n_sop_seed1` | [114] | [84, 104, 119] | 0 | disjoint |
| `c100n_sop_seed2` | [119] | [84, 99, 114] | 0 | disjoint |
| `c10n_random1_ce_seed0` | [34] | [19] | 0 | disjoint |
| `c10n_random1_ce_seed1` | [19] | [14, 19] | 1 | contained |
| `c10n_random1_ce_seed2` | [34] | [24] | 0 | disjoint |
| `c10n_random1_elr_seed0` | [104] | [24, 29, 59] | 0 | disjoint |
| `c10n_random1_elr_seed1` | [84] | [19, 34, 39, 44] | 0 | disjoint |
| `c10n_random1_elr_seed2` | [79] | [59, 64] | 0 | disjoint |
| `c10n_random1_gce_seed0` | [94] | [34] | 0 | disjoint |
| `c10n_random1_gce_seed1` | [104] | [49] | 0 | disjoint |
| `c10n_random1_gce_seed2` | [99] | [39] | 0 | disjoint |
| `c10n_random1_sop_seed0` | [104] | [79, 104] | 1 | contained |
| `c10n_random1_sop_seed1` | [119] | [19, 29, 49] | 0 | disjoint |
| `c10n_random1_sop_seed2` | [119] | [14] | 0 | disjoint |
| `c10n_worst_ce_seed0` | [19] | [19, 24] | 1 | contained |
| `c10n_worst_ce_seed1` | [24] | [19] | 0 | disjoint |
| `c10n_worst_ce_seed2` | [19] | [34] | 0 | disjoint |
| `c10n_worst_elr_seed0` | [49] | [29, 39, 59, 64] | 0 | disjoint |
| `c10n_worst_elr_seed1` | [59] | [59, 64] | 1 | contained |
| `c10n_worst_elr_seed2` | [79] | [59, 69] | 0 | disjoint |
| `c10n_worst_gce_seed0` | [44] | [19] | 0 | disjoint |
| `c10n_worst_gce_seed1` | [54] | [29] | 0 | disjoint |
| `c10n_worst_gce_seed2` | [44] | [29] | 0 | disjoint |
| `c10n_worst_sop_seed0` | [104] | [34, 44] | 0 | disjoint |
| `c10n_worst_sop_seed1` | [89] | [24] | 0 | disjoint |
| `c10n_worst_sop_seed2` | [119] | [29] | 0 | disjoint |

`contained` means every ID-oracle epoch is OOD-feasible; `disjoint` means none is;
`partial overlap` means some but not all. `F_OOD empty` is reported separately because
no relation to an empty set is informative.

