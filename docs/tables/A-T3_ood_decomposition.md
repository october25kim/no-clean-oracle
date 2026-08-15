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

## Aggregation-variant rows — NOT EMITTED

The brief asks for one row per aggregation variant (mean / min / max) confirming taxonomy invariance. Those rows are **not** produced here, and the reason is the one that removed F8 from the figure set: the aggregated OOD curves are stored (`A9.aggregation_sensitivity.<u>.{mean,min,max}`, 24 points each), but their taxonomy is not. Deriving it means giving each aggregated curve its own corrected oracle and 24-epoch IQR denominator, re-forming `max_a ĝ_a(t)` against the unchanged ID and WC axes, and thresholding at δ — a new quantity, not a display transform, so it is reported rather than computed.

One row of that block is available without any new computation and is stated here for completeness: the **energy score averaged over the semantic pools {svhn, cross_cifar}** *is* the primary R_OOD by definition (protocol §4), so its taxonomy is the primary taxonomy already reported — 27 incompatible / 3 compatible-solved / 4 compatible-unsolved / 2 indeterminate at δ = 0.10. The remaining variants (min, max, and the msp-scored equivalents) need the derivation above.

