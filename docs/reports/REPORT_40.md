# REPORT 40 — T10 and T11: the manuscript's last two open numbers

```
PROJECT: G1 — No Clean Oracle
REPORT #40
hostname: user
repo: /data/workspace/sanghoon/g1_audit
git HEAD: 6fccceb
environment: numpy 1.26.4, Python 3.12.4
timestamp: 2026-08-19 KST
session binding: this session reports G1 only; no Fed-CORE result appears here.
```

Both are **new-estimand class**, registered as such in `docs/remediation_plan_v2.md`, and neither
can alter the anchored step-6 adjudication, which stands on the registered primary analysis.
Published whichever way they came out.

---

## Part 1 — T10: is the worst-class axis an artifact of its own class selection?

The registered WC axis fixes its class set from the reference run's final-epoch clean-test
per-class accuracy, then scores every checkpoint on that same clean test set. Selection and
assessment share a sample. T10 separates the two roles.

**Machinery validated before use.** The registered tail comes from the training run's logged
`per_class_test_error`; a refit must recompute per-class accuracy from forward logits, which is
a different code path. Across all **51 paired runs** the two agree at
**max|diff| = 0.000e+00** and select identical tail sets, so the only thing separating the
refit from the registered set is the index subset. Independently, T10's arm 1 (registered
classes, full sample) reproduces the adjudicated battery on **51/51 runs**, rho* to <1e-9 with
identical taxonomy.

**Three arms, because two are needed to attribute anything.** `registered` (registered classes,
full sample) · `refit` (classes proposed on the proposal half, scored on the assessment half) ·
`control` (registered classes, assessment half). Halving the evaluation sample moves the IQR
denominator and the oracle on its own, so control-vs-registered isolates sample size and
refit-vs-control isolates selection. Without the control arm every change below would have been
misread as a selection artifact.

### Class agreement

| dataset | k | overlap | mean |
|---|---|---|---|
| CIFAR-10, 1000/class | 3 | **3/3 in all 30 runs** | 3.00 |
| CIFAR-100, 100/class | 30 | 24–26/30 | 25.50 |

### Taxonomy movement, attributed (48 computable runs)

| dataset | n | sample-size effect | selection effect | combined |
|---|---|---|---|---|
| CIFAR-10 | 30 | 1 | **0** | 1 |
| CIFAR-100 | 18 | 7 | 5 | 6 |

The WC axis is **essentially deterministic at 10 classes and materially unstable at 100**. Both
effects are real at 100 classes and neither dominates; at 10 classes selection changes nothing
at all.

### The single split is not reliable at 50 per class

The registration permits a K-fold cross-fit "if class sizes forbid a single split". Rather than
settle that by assertion, both ran. 5-fold folds are unanimous in **22/30** CIFAR-10 runs but
only **5/18** CIFAR-100 runs; the 5-fold majority agrees with the single split in 38/48; median
fold-to-fold rho* range is **0.2703** (CIFAR-10) and **0.3126** (CIFAR-100). At 50 images per
class, per-class accuracy moves in 2% steps and a single split can select a tail out of
quantisation noise. The answer to the registration's conditional is empirical: it does.

**3 runs not computable** — `cifar10_symmetric0.2_elr_seed{0,1,2}`, whose registered reference
`cifar10_symmetric0.2_ce_seed0` has no stored forward pass. Recorded as such rather than
substituting a different reference, which would silently change the registered rule.

---

## Part 2 — T11: joint paired bootstrap under the fixed-cells estimand

Frame gated before any resampling: **36 runs, 12 cells, 3 distinct seeds each**.

### psi — the pinned estimand

**psi = 0.7500, 95% CI [0.6389, 0.8611], SE 0.0600, 12 cells.**

Layer (ii) only. The interval comes solely from resampling seeds *within* their cell; **cells
are never resampled**, because psi is a mean over a designed factorial grid and resampling
cells would smuggle in the superpopulation-of-conditions claim the pin forbids. Nothing here
licenses that reading.

psi is **identical across all three WC arms** by construction — it reads the registered
per-seed taxonomy, which no evaluation resample can move. That invariance is a check that the
two layers are genuinely separated rather than nominally so.

### Layer (i) — the certified window is empty everywhere

| delta | F_cert > 0 | F_poss > 0 | mean \|F_poss\| |
|---|---|---|---|
| 0.05 | **0/36** | 16/36 | 0.89 |
| 0.10 | **0/36** | 22/36 | 1.61 |
| 0.20 | 1/36 | 28/36 | 2.56 |

No checkpoint is jointly adequate across every evaluation resample, at any registered delta bar
one run at 0.20. F_poss averages 3.75 on CIFAR-100 against 0.54 on CIFAR-10. Aggregation choice
moves the count materially (min/mean/max → 19/22/22 runs with a nonempty possible window),
which is why the registered aggregation arm belongs in the report.

### WC arm comparison — the registered variant

| arm | F_cert>0 | F_poss>0 | mean \|F_poss\| | mean P(incomp\|boot) | distinct tail sets |
|---|---|---|---|---|---|
| crossfit | 0/36 | 22/36 | 1.61 | 0.7094 | 1662 |
| plain | 0/36 | 21/36 | 1.64 | 0.7375 | 334 |
| control | 0/36 | 20/36 | 1.56 | 0.7210 | 1 |

The three arms span the full range of tail-selection discipline and their tail instability
differs by **three orders of magnitude** (1 → 334 → 1662 distinct sets across replicate x fold),
yet every downstream quantity moves by a few percent and `F_cert = 0` holds in all 36 runs in
all three arms. **The cross-fit correction was a genuine fidelity requirement of the pin and was
not load-bearing for the conclusion.** Both halves of that sentence are the finding; only
running all three arms establishes it. The same-draw arm does agree slightly more tightly with
the registered classification (mean |P_boot − 1[registered incompatible]| = 0.1009 vs 0.1152
cross-fitted), the direction expected from reusing the fluctuations that chose the tail, but the
magnitude is small.

Corroboration across methods: T11's tail instability (4970–4988 of 5000 distinct on CIFAR-100
against 5–6 on CIFAR-10) reproduces T10's split from inside the bootstrap, by a
methodologically unrelated route.

### Certificate robustness — no certificate survives

**0 of 36 (run, pair) certificates stay disjoint in >=95% of replicates.** Maxima by arm:
crossfit 0.723, plain 0.737, control 0.925. 15/36 runs show at least one disjoint (score, pool)
pair in *some* replicate.

The registered A9 two-world certificates are point-estimate artifacts under evaluation
resampling. This is a negative result and it is reported as one.

### Validation

0 consistency violations across **324** run x aggregation x delta cells (`F_cert` subset of
`F_poss` throughout).

---

## Provenance note

T11 as first written contained five defects, all caught by an adversarial review **before the
registered execution** and recorded as **D-19**. One would have changed this report's headline:
psi was being computed from layer-(i) bootstrap frequencies rather than the pinned per-seed
registered taxonomy, combining the two layers the pin requires kept apart. Two of the five were
verified independently rather than accepted on report. No number was retracted because none had
been published.
