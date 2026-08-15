# T1 — Phase-I cells, frozen record  [FROZEN-HISTORICAL]

All 16 preregistered cells (dataset × noise × learner). Transcribed from the frozen Phase-I artifacts; no value is recomputed. NCR = cross-regret / IQR over epochs; the registered threshold is 0.1. A cell is *conflicted* when at least one off-diagonal NCR has its 95% BCa CI entirely above the threshold — that is the CI-separated condition, so the two are one column, not two.

Raw cross-regret is given **per seed**, at the same axis pair as the largest off-diagonal NCR. The frozen record holds NCR at cell level and CR only per seed, so a cell-level mean CR would be a number the record never contained.

| # | dataset | noise | learner | conflicted (CI-separated) | ID↔OOD | largest off-diag NCR | axis pair | raw CR per seed at that pair | Phase II |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cifar100 | asymmetric 0.4 | ce | no | no | 0.2798 | WC (tail) ← OOD | +0.0073, +0.0303, +0.0013 | — |
| 2 | cifar100 | asymmetric 0.4 | elr | no | no | 0.4763 | WC (tail) ← OOD | +0.0067, +0.1187, +0.0117 | — |
| 3 | cifar100 | symmetric 0.2 | ce | no | no | 0.0759 | ID ← OOD | +0.0070, +0.0009, +0.0121 | — |
| 4 | cifar100 | symmetric 0.2 | elr | no | no | 0.5603 | OOD ← WC (tail) | +0.0462, +0.0057, +0.0008 | — |
| 5 | cifar100 | symmetric 0.4 | ce | yes | yes | 2.3830 | OOD ← WC (tail) | +0.0861, +0.0858, +0.1467 | **selected** |
| 6 | cifar100 | symmetric 0.4 | elr | no | no | 0.4953 | ID ← OOD | +0.0128, +0.0804, +0.0006 | — |
| 7 | cifar100 | symmetric 0.6 | ce | yes | yes | 1.2499 | WC (tail) ← OOD | +0.0400, +0.0277, +0.0237 | **selected** |
| 8 | cifar100 | symmetric 0.6 | elr | yes | no | 0.6517 | OOD ← ID | +0.0259, +0.0316, -0.0005 | — |
| 9 | cifar10 | asymmetric 0.4 | ce | yes | yes | 1.9247 | WC (tail) ← OOD | +0.1950, +0.2583, +0.2040 | **selected** |
| 10 | cifar10 | asymmetric 0.4 | elr | yes | yes | 1.2332 | OOD ← ID | +0.0800, +0.0885, +0.0877 | **selected** |
| 11 | cifar10 | symmetric 0.2 | ce | no | no | 1.0599 | OOD ← WC (tail) | +0.1559, -0.0362, +0.2378 | — |
| 12 | cifar10 | symmetric 0.2 | elr | yes | yes | 1.0290 | OOD ← ID | +0.1975, +0.0569, +0.0562 | **selected** |
| 13 | cifar10 | symmetric 0.4 | ce | no | no | 0.7803 | OOD ← WC (tail) | +0.1063, +0.0022, +0.0810 | — |
| 14 | cifar10 | symmetric 0.4 | elr | no | no | 0.3032 | ID ← OOD | +0.0105, +0.0202, +0.0028 | — |
| 15 | cifar10 | symmetric 0.6 | ce | no | no | 0.4709 | OOD ← ID | +0.0859, +0.0000, +0.0891 | — |
| 16 | cifar10 | symmetric 0.6 | elr | yes | no | 0.4791 | OOD ← WC (tail) | +0.1070, +0.0393, +0.0487 | — |

Frozen verdict: **WEAK** — 7/16 cells conflicted, ID↔OOD conflict in 5, 2 cells with all CIs overlapping 0, 2 tail-only. The 5 cells marked **selected** are the conflict-positive cells carried into Phase II (15 runs).

| source | sha256 |
|---|---|
| `results/report/analysis_values.json` | `120a93661205a1194851cf6a9f945192af07e106aeb6bd1c4c4775307dc5abc7` |
| `results/report/verdict.json` | `1b64e39108ec18ee8903b85cdb96d9698ea17386e6d44375b012791e53eaafb6` |

Field-name mapping: the frozen artifacts use `tail` for the axis renamed **WC (worst-class)** in the corrected frame (F3). Cell keys carry the underscore-separated eta used by the sealed verdict rule.

