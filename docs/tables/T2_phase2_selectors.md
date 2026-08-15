# T2 — Phase-II selector success counts, frozen record  [FROZEN-HISTORICAL]

Per-selector × per-axis success counts over the 15 Phase-II runs, at the three registered δ. Transcribed from the frozen classification record; no value is recomputed. A success is a normalized regret ≤ δ on that axis.

| selector | axis | δ=0.05 | δ=0.10 | δ=0.20 |
|---|---|---|---|---|
| E (mean max-softmax, τ=1) | ID | 0/15 | 1/15 | 4/15 |
| E (mean max-softmax, τ=1) | WC (tail) | 2/15 | 3/15 | 5/15 |
| E (mean max-softmax, τ=1) | OOD | 0/15 | 0/15 | 0/15 |
| NA (C1, accuracy) | ID | 0/15 | 2/15 | 4/15 |
| NA (C1, accuracy) | WC (tail) | 3/15 | 3/15 | 5/15 |
| NA (C1, accuracy) | OOD | 0/15 | 0/15 | 0/15 |
| ER (effective rank, argmax) | ID | 0/15 | 1/15 | 4/15 |
| ER (effective rank, argmax) | WC (tail) | 2/15 | 3/15 | 4/15 |
| ER (effective rank, argmax) | OOD | 0/15 | 0/15 | 0/15 |

**Minimum OOD normalized regret** across all 45 registered selector × run cells: **+0.2893** — NA (C1), run `cifar100_symmetric0.4_ce_seed0`. The classification record states +0.289; recomputed from the same frozen per-run artifacts the value is 0.289306, so the record's figure is confirmed. Every OOD success count above is 0 at every δ, which follows: the smallest OOD regret in the whole grid exceeds the largest δ.

Other frozen figures for cross-reference: maximum single-axis success at the binding δ = 3/15; maximum at any δ = 5/15; OOD successes at δ=0.20 = 0.

| source | sha256 |
|---|---|
| `results/report/classification_verification_record.json` | `c4c925b01655005b5e21a319f81380c65eb5c24a0abdddb094819c56241a969e` |
| `results/report/E_tgrid.json` | `a4d8d29904bdbe70ec04e28244bc61450c64249d967f21d69a266fabaf67dafd` |
| `results/report/C1_noisyval.json` | `97f64497a13fd0915f1322850a883941ef0a4421f9590815a97756b515039aad` |
| `results/report/B_representation.json` | `cd2d7ae20ee58a534099ff98e96f024d2ccc84548f455eda95b10e0fa59f0d69` |

Field-name mapping: the frozen artifacts use `tail` for the axis renamed **WC (worst-class)** in the corrected frame, and `C1` for the selector renamed **NA** (in-sample noisy agreement) in spec v2. Selector variant keys are the frozen ones; E's primary statistic is the mean max-softmax at τ=1.

