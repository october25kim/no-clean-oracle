# REPORT 36 — Clothing1M reconnaissance, results  [EXPLORATORY-UNVERIFIED-PROVENANCE]

```
PROJECT: G1 — No Clean Oracle
REPORT #36
hostname: user
repo: /data/workspace/sanghoon/g1_audit
git HEAD: 3314f0a
public anchor: 110baa9d29e8a0a705a15a258642cbea543c71bc (ANCHOR_CHAIN row 33)
environment: container torch 2.3.0 + CUDA 12.1, image fedcore-c400r:latest, 4x TITAN RTX
timestamp: 2026-08-18 06:05 KST
session binding: this session reports G1 only; no Fed-CORE result appears here.
```

**Classification.** Everything below is `[EXPLORATORY-UNVERIFIED-PROVENANCE]`. It is **not**
Tier 2. Nothing here may enter registered artifacts, paper claims, or be pooled with any
registered result. The dataset provenance clause stands unchanged: *third-party re-upload;
official distribution requires an agreement form; no official checksum exists; contents
unverifiable against the canonical release.*

**Numbers only.** This report stops before interpretation, as instructed.

---

## 1. Campaign

Four runs, 20 epochs each, ResNet-50 from ImageNet weights, native 64x64, SGD(0.01, 0.9,
1e-3), batch 256, cosine to zero. ELR λ=3.0 / β=0.7, still flagged NEEDS-VERIFICATION: it
equals the registered cifar10 row and was not re-derived for this dataset.

| run | host GPU | total s | mean s/epoch | min | max | exit |
|---|---|---|---|---|---|---|
| `c1m_ce_seed0` | 1 | 23025 | 1151.3 | 1143.3 | 1154.1 | 1 |
| `c1m_ce_seed1` | 1 | 23039 | 1152.0 | 1141.5 | 1155.4 | 0 |
| `c1m_elr_seed0` | 3 | 23940 | 1197.0 | 1189.5 | 1199.8 | 1 |
| `c1m_elr_seed1` | 3 | 23953 | 1197.6 | 1188.5 | 1200.0 | 0 |

The two exit-1s are defect D-13, a duplicate-key `TypeError` in the completion record raised
after all 20 epochs had run. Every checkpoint and metrics line is intact; both TERMINALs were
reconstructed post-hoc and say so in their own text. The two exit-0s are the fix confirmed.

**GPU 1 vs GPU 3.** GPU 1 mean 1151.6 s/epoch (n=2), GPU 3 mean 1197.3 s/epoch (n=2), a gap
of **3.97%**. This is **not** a device effect and may not be reported as one: CE ran only on
GPU 1 and ELR only on GPU 3, in both waves, so learner and device are fully confounded. ELR
does strictly more work per step. The design does not separate the two.

Placement is recorded in `results/exploratory_c1m/PLACEMENT.json`, read from the container
objects because no run records its own physical GPU — see the D-9 addendum. All four:
CE on host GPU 1, ELR on host GPU 3, `--shm-size` 17179869184 throughout.

## 2. Forward pass

80 checkpoints. Clean test 10,526 (ID and worst-class axes) and the NA split 5,000 (NA
selector and effective rank). Both containers exit 0, 20 OK markers per run, 3.2 GB.

The OOD axis is **absent**, not empty: there is no OOD pool, so `max_a` runs over ID and WC,
and no two-world certificate exists to look for.

## 3. A1 — per-axis oracles

Epochs are 0-indexed over the 20-point grid.

| run | ID t\* | ID R\* | ID d = IQR | WC t\* | WC R\* | WC d = IQR |
|---|---|---|---|---|---|---|
| `c1m_ce_seed0` | 19 | 0.3587 | 0.0427 | 6 | 0.5432 | 0.0511 |
| `c1m_ce_seed1` | 14 | 0.3558 | 0.0304 | 15 | 0.5389 | 0.0343 |
| `c1m_elr_seed0` | 18 | 0.3297 | 0.0521 | 17 | 0.5517 | 0.0296 |
| `c1m_elr_seed1` | 17 | 0.3298 | 0.0480 | 17 | 0.5512 | 0.0482 |

No axis was fail-closed: every IQR is strictly positive, so D-7's rule excluded 0 of 8
axis-instances. `d = max{IQR, ε}` reduced to the IQR in all eight.

Worst-class tail: classes **[0, 4, 7, 11]**, k = round(0.30 × 14) = 4, taken from the final
epoch of `c1m_ce_seed0` by the registered `ref_id` rule (the `ce_seed0` run of the condition)
and held fixed across all four runs.

## 4. A2/A3 — ρ̂\*, feasible sets, taxonomy

w_δ denominator is **20** (the recon grid), not the registered 24.

| run | ρ̂\*_LE | \|F_0.05\| | \|F_0.10\| | \|F_0.20\| | w_0.10 | taxonomy at δ=0.10 |
|---|---|---|---|---|---|---|
| `c1m_ce_seed0` | 0.1418 | 0 | 0 | 2 | 0.000 | incompatible |
| `c1m_ce_seed1` | 0.1749 | 0 | 0 | 1 | 0.000 | incompatible |
| `c1m_elr_seed0` | 0.0511 | 0 | 1 | 1 | 0.050 | compatible-unsolved |
| `c1m_elr_seed1` | 0.0000 | 1 | 2 | 4 | 0.100 | compatible-solved |

F_0.10 membership, as epochs: `c1m_ce_seed0` ∅ · `c1m_ce_seed1` ∅ · `c1m_elr_seed0` {17} ·
`c1m_elr_seed1` {17, 19}.

No run fell in the indeterminate band (|ρ̂\* − δ| ≤ 0.025): the four distances from δ=0.10
are 0.0418, 0.0749, 0.0489 and 0.1000.

## 5. A4 — registered selectors

Evaluated on the NA split, not a full training forward.

| run | selector | epoch | Ĵ_LE | η̂_LE |
|---|---|---|---|---|
| `c1m_ce_seed0` | E(τ=1) | 19 | 0.1930 | 0.0512 |
| `c1m_ce_seed0` | NA | 19 | 0.1930 | 0.0512 |
| `c1m_ce_seed0` | ER-argmax | 2 | 1.7973 | 1.6555 |
| `c1m_ce_seed1` | E(τ=1) | 19 | 0.4983 | 0.3234 |
| `c1m_ce_seed1` | NA | 19 | 0.4983 | 0.3234 |
| `c1m_ce_seed1` | ER-argmax | 3 | 2.8709 | 2.6961 |
| `c1m_elr_seed0` | E(τ=1) | 19 | 0.3032 | 0.2521 |
| `c1m_elr_seed0` | NA | 19 | 0.3032 | 0.2521 |
| `c1m_elr_seed0` | ER-argmax | 3 | 1.3084 | 1.2573 |
| `c1m_elr_seed1` | E(τ=1) | 19 | 0.0990 | 0.0990 |
| `c1m_elr_seed1` | NA | 19 | 0.0990 | 0.0990 |
| `c1m_elr_seed1` | ER-argmax | 3 | 1.7267 | 1.7267 |

E(τ=1) and NA selected the same epoch in all four runs, so their Ĵ and η̂ coincide
throughout. LW-N was not computed: it is reported and never gating in the registered frame.

## 6. A5 — uniform baseline

p_unif = w_δ exactly, so the δ=0.10 column is 0.000, 0.000, 0.050, 0.100 as in §4.

## 7. Deviations carried

Registered in `scripts/recon_forward.py` and `scripts/recon_battery.py` before either ran:

1. Grid is all 20 epochs, so w_δ has denominator 20 against the registered 24. Not
   comparable as fractions of a fixed budget.
2. Effective rank on the 5,000-row NA split, not the full training forward — a different
   estimator, not the registered one on less data. n=5000 against d=2048.
3. `draw_split`'s remainder branch executed for the first time: 14 classes gives remainder 2,
   so classes 0 and 1 contribute 358 rows and the rest 357.
4. Worst-class granularity is 1/297 = 0.337%, the smallest clean-test class.
5. No OOD axis, no frozen-historical comparison.

## 8. Stop

Per instruction, this report stops before interpretation. No claim is made here about what
these numbers mean for the registered work, and none may be carried across the
classification boundary.
