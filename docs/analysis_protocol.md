# G1 analysis protocol — independent recomputation spec

Everything the G1 verdict rests on is derived from the per-epoch JSONL logs by
CPU-only code. This document specifies that derivation exactly enough for a second
implementation to reproduce the server's numbers **bit-for-bit on the point estimates
and exactly on the confidence intervals**, not merely within bootstrap noise.

Reference implementation on the server side:

| step | file |
|---|---|
| log reader, tail-set reconstruction, risk trajectories | `src/analysis/io.py` |
| moving average, oracle epochs, CR/NCR, BCa, verdict | `src/analysis/ncr.py` |
| tail-class construction | `src/eval/tail.py` |
| report / table / figure assembly | `scripts/make_report.py` |

Tolerances:

- **point estimates** (oracle epochs, R\*, CR, NCR, IQR, means): agreement to **1e-9**.
  Oracle epochs are integers and must match exactly.
- **BCa interval bounds**: **exact agreement** is achievable and expected, because the
  resampling is fully seeded (§6). If the two sides disagree here after following §6,
  the cause is an implementation difference, not Monte-Carlo noise.

---

## 1. Input

One directory per run, `results/runs/<run_id>/metrics.jsonl`. Line 0 is the run
metadata; every later line is one epoch, in ascending `epoch` order starting at 0.
Only these fields are read by the analysis:

**metadata** — `run_id`, `dataset`, `noise_type`, `eta`, `learner`, `seed`,
`tail_frac` (0.30), `effective_overall_noise`, `epochs` (120), `smoke` (false for all
sweep runs).

**per epoch** — `epoch`, `R_ID`, `R_OOD_primary_energy_semantic`,
`per_class_test_error` (length = number of classes).

Everything else in the log (`R_OOD` per score x pool, `R_tail_dynamic`,
`train_loss_*`, `per_class_train_acc`, timings, `gpu_uuid`, `co_tenant`, OOD
provenance) is recorded for the record and for diagnostics, and does **not** enter the
verdict path.

Note `R_tail_static` is `null` in every epoch line by design: the static tail-class set
is not knowable during training and is reconstructed here (§3).

A run whose last line is truncated (a live sweep appending) must have that line
dropped, not repaired.

## 2. Cell definition

The pre-registered analysis unit is one **(dataset x noise x learner)** cell —
**16 cells** — named

```
{dataset}_{noise_type}_{eta:g}_{learner}      e.g. cifar10_symmetric_0.6_ce
```

Each cell holds exactly the **3 seeds** {0, 1, 2}, and the bootstrap (§6) resamples
those 3 runs. The learner is a **fixed factor** and is never pooled into a resampling
unit.

The underscore before `eta` is load-bearing: the sealed verdict detects the
highest-noise cells with `"_0.6_" in name or name.endswith("_0.6")`.

A separate **(dataset x noise)** key — the **group**, 8 of them — is used for exactly
two things: it scopes the static tail set (§3), and it keys the pooled secondary table,
which is diagnostic and reads on no decision.

## 3. Static tail-class set

Fixed per **group**, so CE and ELR in the same group are scored on the same classes.

1. Take the group's `learner == "ce"`, `seed == 0` run.
2. Read `per_class_test_error` of its **final** epoch (index 119). `acc = 1 - error`.
3. `k = max(1, int(round(0.30 * C)))` — Python `round`, i.e. banker's rounding
   (C = 10 -> k = 3; C = 100 -> k = 30).
4. `order = np.lexsort((np.arange(C), acc))` — ascending accuracy, ties broken by
   ascending class index. Take the first `k`, then `np.sort` them.

## 4. Risk trajectories

For each run, three length-120 vectors:

| objective | value |
|---|---|
| `ID` | the logged `R_ID` |
| `tail` | `np.nanmean(per_class_test_error[tail_classes])`, recomputed per epoch on the §3 class set |
| `OOD` | the logged `R_OOD_primary_energy_semantic` |

`R_OOD_primary_energy_semantic` was computed at training time as the mean over the
semantic pools `[svhn, cross_cifar]` of `1 - AUROC` of the **energy** score. The
covariate pool `CIFAR-C-local` is logged but is **not** part of the primary endpoint.

Objective order is fixed: `OBJECTIVES = ("ID", "tail", "OOD")`.

## 5. Oracle epochs, CR, NCR

```python
def moving_average(x, w=3):
    k   = np.ones(w) / w
    pad = w // 2
    xp  = np.pad(x, pad, mode="edge")
    return np.convolve(xp, k, mode="same")[pad: pad + x.size]
```

- **oracle epoch** `t*_j = np.argmin(moving_average(R_j, 3))` — first index on ties.
  The raw argmin is also recorded but is **not** what CR uses.
- **R\*_j = R_j[t*_j]** — the **raw** risk at the smoothed argmin, not the smoothed
  value and not the raw minimum.
- **CR[a, b] = R_a[t*_b] - R*_a** — row `a` is the objective being **measured**,
  column `b` is the objective whose oracle checkpoint is **deployed**. The diagonal is
  identically 0.
- **IQR_a = np.percentile(R_a, 75) - np.percentile(R_a, 25)** (numpy default `linear`
  interpolation, over all 120 epochs).
- **NCR[a, b] = CR[a, b] / IQR_a**, and exactly `0.0` when `IQR_a == 0`.

**CR may be slightly negative.** `t*` comes from the smoothed curve while the regret is
read off the raw curve, so a raw dip one epoch away from the smoothed minimum produces
a small negative entry. Values are reported raw and **unclamped**; a reviewer
implementation that clamps at zero will not match.

## 6. BCa bootstrap — shared seed protocol

Per cell, the 3 seed-level NCR matrices are stacked to `(3, 3, 3)` = (runs, row, col).
The cell mean is the elementwise mean. Each of the 9 entries independently gets a BCa
interval over its 3 values.

To make the intervals reproduce exactly, both sides must match all of:

1. **Bit generator**: `np.random.default_rng(seed)` — numpy **PCG64**. Legacy
   `np.random.RandomState` / MT19937 draws a different sequence and will not match.
2. **Seed = 0**, and the generator is **re-created inside every `bca_ci` call**, so all
   9 matrix entries of a cell reuse the *same* resample index matrix. Do not thread one
   shared stream through the 9 entries.
3. **One draw call**, exactly: `rng.integers(0, n, size=(n_boot, n))` with `n = 3`,
   `n_boot = 10000`, C-order; then `boot = x[idx].mean(axis=1)`.
4. **Degenerate short-circuit before any drawing**: if `n < 2` or
   `np.allclose(x, x[0])` (numpy defaults `rtol=1e-5`, `atol=1e-8`), return
   `(theta, theta, theta)` and draw nothing. This fires on every diagonal entry.
5. **Bias correction**: `prop = mean(boot < theta)` (strict `<`), clipped to
   `[1/n_boot, 1 - 1/n_boot]`, then `z0 = scipy.stats.norm.ppf(prop)`.
6. **Acceleration**: leave-one-out jackknife means `jack[i] = mean(delete(x, i))`,
   `jbar = mean(jack)`,
   `a = sum((jbar - jack)**3) / (6 * sum((jbar - jack)**2)**1.5)`, and `a = 0` when the
   denominator is 0.
7. **Endpoints**: `alpha = 0.05`; `zl, zu = norm.ppf(0.025), norm.ppf(0.975)`;
   `adj(z) = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))`;
   `lo = np.percentile(boot, 100 * adj(zl))`, `hi = np.percentile(boot, 100 * adj(zu))`
   — numpy default `linear` interpolation.

With n = 3 the jackknife has 3 points and the intervals are wide; that is a property of
the pre-registered design, not a defect to smooth over.

## 7. Verdict

Applied **verbatim** to the 16 primary cells at `thresh = 0.10`
(`configs/base.yaml: analysis.ncr_threshold`). The sealed rule, as implemented in
`ncr.verdict`:

```
PASS : in >=50% of cells, at least one off-diagonal NCR has 95% CI entirely
       above thresh, AND the ID<->OOD direction shows conflict (CI_lo>thresh)
       in >=4 cells.
KILL : in >70% of cells, ALL off-diagonal NCR CIs overlap 0.
WEAK : conflict concentrated only in the tail objective, or only at noise 0.6.
```

Against 16 cells the thresholds read as written: **">=50% of cells" is >= 8 of 16**,
and the **ID<->OOD clause is >= 4 of 16**. Ambiguous middle ground resolves to `WEAK`.
Per-cell booleans: a cell "has conflict" iff some off-diagonal has `ci_lo > thresh`;
"all overlap 0" iff every off-diagonal satisfies `ci_lo <= 0 <= ci_hi`; "ID<->OOD" iff
`ci_lo[ID, OOD] > thresh` or `ci_lo[OOD, ID] > thresh`.

## 8. What to compare

The server ships `results/report/analysis_values.json` with full float precision (not
the 3-decimal markdown tables). Compare, per cell:

- `t_star` and `t_star_raw` per objective, per seed — exact integer match
- `R_star`, `IQR`, `CR`, `NCR` per seed — 1e-9
- cell `mean` NCR matrix — 1e-9
- `ci_lo` / `ci_hi` matrices — exact, given §6
- the verdict counters and the final label — exact

## 9. Known non-comparabilities

- `CIFAR-C-local` is a local substitution for official CIFAR-10-C/100-C: a fixed
  4-corruption x 2,000-image subsample at severity 3 built with `imagecorruptions`
  1.1.2. It is not official CIFAR-C and its numbers are not comparable to published
  CIFAR-C results. It is not in the primary endpoint.
- `train_loss_noisy` is each learner's **own** objective (CE for CE; CE + lambda *
  regularizer for ELR) and must not be compared across learners. For memorization use
  `per_class_train_acc` (fraction of noisy labels predicted, against the clean-label
  fraction) and the upper quantiles of `train_loss_quantiles`; the median moves the
  wrong way at these noise rates because the median training sample is correctly
  labelled.
- Wall-clock fields (`epoch_time_s`, `train_time_s`, `eval_time_s`) were measured
  co-located with an unrelated workload whose load varied across the sweep. They
  describe scheduling, never results.
