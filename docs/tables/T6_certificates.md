# T6 — Two-world certificate-bearing runs  [CORRECTED-DESCRIPTIVE]

Runs in which at least one pair of OOD (score | pool) variants has **disjoint** δ=0.1 feasible sets on the OOD axis, with ID and WC held fixed. Sources: `per_run[].A9.two_world_certificate.pairs[]` and `per_run[].A9.per_score_pool.<u|o>.F_delta`; the ID/WC column is a threshold read of the stored `A1.axes.{ID,WC}.ghat` at δ. Checkpoint epochs, not indices.

### `c100n_elr_seed2` — 2 disjoint pair(s)

ID ∩ WC feasible set: [89, 94, 99, 109, 114]

| pair | variant | OOD-axis feasible set |
|---|---|---|
| 1 | `msp|cross_cifar` | [99, 109, 114] |
| 1 | `msp|CIFAR-C-local` | [94] |
| 2 | `energy|svhn` | [89, 94] |
| 2 | `energy|cross_cifar` | [99, 109] |

### `c100n_gce_seed2` — 4 disjoint pair(s)

ID ∩ WC feasible set: [89, 94, 99, 104, 109, 114, 119]

| pair | variant | OOD-axis feasible set |
|---|---|---|
| 1 | `msp|svhn` | [89] |
| 1 | `msp|cross_cifar` | [99] |
| 2 | `msp|cross_cifar` | [99] |
| 2 | `msp|CIFAR-C-local` | [89] |
| 3 | `energy|svhn` | [89] |
| 3 | `energy|cross_cifar` | [94, 99] |
| 4 | `energy|cross_cifar` | [94, 99] |
| 4 | `energy|CIFAR-C-local` | [89] |

### `c10n_random1_sop_seed0` — 1 disjoint pair(s)

ID ∩ WC feasible set: [94, 104, 109, 114]

| pair | variant | OOD-axis feasible set |
|---|---|---|
| 1 | `msp|svhn` | [104] |
| 1 | `msp|CIFAR-C-local` | [109, 114] |

**3 certificate-bearing runs, 7 disjoint pairs in total.**

A disjoint pair means two admissible OOD evaluation choices whose δ-adequate checkpoint sets do not intersect: no single checkpoint is adequate under both. Counts only.

