# T13 — OOD axis specification  [CORRECTED-DESCRIPTIVE]

What the OOD axis actually is, end to end. Documentation of what ran; no value is recomputed.

| aspect | specification |
|---|---|
| base metric | `1 − AUROC`, ID as the positive class |
| AUROC estimator | Mann-Whitney rank identity, ties at 0.5, unit-tested against sklearn |
| direction | higher score ⇒ more in-distribution; risk is `1 − AUROC`, so lower is better |
| scores | `msp` = max softmax; `energy` = `T·logsumexp(logits/T)` (sign flipped so higher ⇒ ID) |
| temperature | `ood_energy_T = 1.0` |
| primary pools | `svhn`, `cross_cifar` (semantic) |
| sensitivity pool | `CIFAR-C-local` (covariate) — logged, never in the primary |
| aggregation | unweighted arithmetic **macro** mean over the two semantic pools — NOT pooled samples, so the pools contribute equally despite differing n |
| per-pool n | svhn 26,032 · cross_cifar 10,000 · CIFAR-C-local 8,000 |
| ID sample | the 10,000-image clean test set, shared with the ID and WC axes |
| oracle | raw argmin on the 24 retained checkpoints, **earliest** index on ties |
| denominator | IQR of the raw risk over those same 24 epochs; exact zero is fail-closed |
| CIFAR-C-local caveat | not official CIFAR-10-C/100-C; imagecorruptions 1.1.2 over opencv-headless 4.10.0.84, severity 3, four corruptions, 2,000 per corruption, seeds 20260811 |

## Per-(score \| pool) corrected oracle epochs, Tier 1 (36 runs)

| score \| pool | median oracle epoch | min | max | runs with zero 24-grid IQR |
|---|---|---|---|---|
| `msp|svhn` | 44 | 9 | 119 | 0/36 |
| `msp|cross_cifar` | 59 | 14 | 109 | 0/36 |
| `msp|CIFAR-C-local` | 59 | 9 | 114 | 0/36 |
| `energy|svhn` | 36 | 9 | 109 | 0/36 |
| `energy|cross_cifar` | 44 | 9 | 119 | 0/36 |
| `energy|CIFAR-C-local` | 42 | 9 | 89 | 0/36 |

The primary axis is the macro mean of the two semantic `energy` rows; the other four rows are the sensitivity decomposition reported in A9.

