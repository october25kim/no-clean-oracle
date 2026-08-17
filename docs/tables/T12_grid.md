# T12 — Grid-density sensitivity  [CORRECTED-DESCRIPTIVE]

The candidate set is thinned from the master 120-point logging grid by taking every T-th epoch (T = 1, 2, 5, 10; T = 5 is the retained checkpoint grid). **The denominator is held FIXED on the master 120-grid throughout**: recomputing the IQR on each thinned grid would move the scale and the candidate set at once and no movement could be attributed to either. Oracles are the raw argmin on the subgrid, earliest tie, and the classification is the registered rule at δ = 0.1. Source: the 120-point curves in each run's `metrics.jsonl`.

## Tier 1 — 36 scorable run(s)

| T | candidates | incompatible | compatible-* | indeterminate | mean \|F_δ\| |
|---|---|---|---|---|---|
| 1 | 120 | 31 | 1 | 4 | 0.11 |
| 2 | 60 | 27 | 1 | 8 | 0.19 |
| 5 | 24 | 25 | 7 | 4 | 0.47 |
| 10 | 12 | 21 | 10 | 5 | 0.56 |

compatible-solved/unsolved are pooled here because the selector epochs are defined on the retained grid only; the split is meaningful at T = 5 and is reported in the primary tables.

**The T = 5 row does not reproduce the primary taxonomy, and should not.** The primary normalises by the IQR over the 24 retained epochs; this table holds the denominator on the master 120-grid for every row, so that thinning the candidate set is the only thing that varies. Two quantities differ between the T = 5 row and the primary table — the candidate set agrees, the scale does not.

The direction across rows follows from the oracle, not from feasibility getting harder: a denser candidate set finds a lower R̂*, which raises every other epoch's regret against it, so more runs fall outside δ. This is the same grid-induced oracle effect recorded in FACT 2, seen here as a sensitivity rather than as a gap column.

## Phase II — 15 scorable run(s)

| T | candidates | incompatible | compatible-* | indeterminate | mean \|F_δ\| |
|---|---|---|---|---|---|
| 1 | 120 | 15 | 0 | 0 | 0.00 |
| 2 | 60 | 15 | 0 | 0 | 0.00 |
| 5 | 24 | 14 | 1 | 0 | 0.07 |
| 10 | 12 | 11 | 4 | 0 | 0.27 |

compatible-solved/unsolved are pooled here because the selector epochs are defined on the retained grid only; the split is meaningful at T = 5 and is reported in the primary tables.

**The T = 5 row does not reproduce the primary taxonomy, and should not.** The primary normalises by the IQR over the 24 retained epochs; this table holds the denominator on the master 120-grid for every row, so that thinning the candidate set is the only thing that varies. Two quantities differ between the T = 5 row and the primary table — the candidate set agrees, the scale does not.

The direction across rows follows from the oracle, not from feasibility getting harder: a denser candidate set finds a lower R̂*, which raises every other epoch's regret against it, so more runs fall outside δ. This is the same grid-induced oracle effect recorded in FACT 2, seen here as a sensitivity rather than as a gap column.

