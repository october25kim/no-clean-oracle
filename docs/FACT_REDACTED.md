# FACT 1–4 — redacted record

Established read-only on 2026-08-13 from the frozen protocol, the committed code, and the
primary source, in response to an external review raising oracle optimism, baseline
fairness, and axis construct validity. This file is the redacted record: definitions,
code with line references, primary-source verbatim, and the structural findings.

**Numeric outcome values are deliberately absent.** The measured magnitudes behind FACT 2
and FACT 4 are recorded in the exposure ledger as EXP-001 and EXP-002 and are referenced
here rather than reproduced, so that reading this file does not re-expose them. Everything
below is either a definition, a quotation, or a structural statement that holds
independently of what the numbers turned out to be.

This file exists in the tree because the same content failed to reach the review side
three times as chat text. Travelling with the anchored snapshot is a transport that does
not depend on a message arriving.

---

## FACT 1a — what `pred_flip_count` actually is

Computed in `src/eval/evaluate.py:84`:

```python
flips = None if prev_pred is None else int((cur_pred != prev_pred).sum())
```

`cur_pred` is collected **inside the training loop**, `src/train/trainer.py:232-251`:

```python
self.model.train()
...
for xb, yb, idx in train_loader:      # train transform: augmentation applied
    ...
    logits = self.model(xb)
    loss.backward(); self.opt.step()  # weights update batch by batch
    with torch.no_grad():
        preds[idx_np] = logits.argmax(1).detach().cpu().numpy()
```

| property | value |
|---|---|
| set | the whole noisy training set, not a subset |
| comparison | argmax at epoch *t* vs argmax at epoch *t−1* |
| mode | `model.train()` — BatchNorm uses batch statistics |
| transform | the training transform, augmented; there is no separate eval pass |
| snapshot | the weights as of the moment each sample's batch was processed — **not** a single θ per epoch |
| granularity | once per epoch, an unnormalized integer count; epoch 0 is `null` |

Structural consequence, stated without magnitudes: each sample is predicted by whatever
parameters existed when its batch came up, so a consecutive-epoch comparison absorbs
intra-epoch weight drift on top of genuine prediction change. Relative to a synchronized
end-of-epoch pass this plausibly biases PC upward. That is a claim about direction, not a
measured quantity.

---

## FACT 1b — the Label Wave statistic, from the primary source

Yuan, Feng & Liu, *Early Stopping Against Label Noise Without Validation Data*, ICLR 2024
(arXiv:2502.07551). Verified copies:

- HTML `10fef4ee92aa6174cdd1d2de01aa78400ebb49f883073bd206ee561488ee973f`
- PDF `1d5ae931d12e713862937a43ae66e5c43d06c8c6d11f58a84817781ffd786ab3`

Eq. (2) — prediction changes = Σ<sub>i∈D</sub> 1[ŷ<sub>i</sub><sup>t</sup> ≠ ŷ<sub>i</sub><sup>(t−1)</sup>]

Eq. (3) — PC′<sub>t</sub> = (PC<sub>t</sub> + PC<sub>t−1</sub> + … + PC<sub>t−k+1</sub>) / k

Eq. (4) — Early Stopping Point = t<sub>first-min</sub>, the **first** local minimum of PC′<sub>t</sub>

Algorithm 1, verbatim:

> Let 𝜽ₒ be the initial parameters and *v* be the local minimum of PC. Let *p* be the
> "Patience", representing the number of times a worsening PC is observed before halting.
>
> `while i < p do` · `Update θ by running the training for n steps, and t ← t+n.` ·
> `PC_t ← Compute prediction changes (PC) in step t.` ·
> `PC'_t ← Moving Averages PC in recent k steps.` ·
> `if PC'_t < v then v ← PC'_t ; i ← 0, θ* ← θ, t* ← t   // Models stored at every new local minimum.` ·
> `else i ← i+1` · `Best parameters are θ*, and best number of training steps is t*.`

Appendix C.3, verbatim — the applicability limitation:

> **Label Wave method is not applicable in very low or no label noise.** There are many
> situations where validation and test errors consistently decrease even with (low level)
> noisy labels in the training data. Modern deep neural networks often exhibit benign
> overfitting (Bartlett et al. 2020), a phenomenon also describable as a memorization
> effect (Zhang et al. 2017; Arpit et al. 2017).
>
> The effectiveness of the Label Wave method in identifying an appropriate early stopping
> point is attributed to our design of a practical metric that tracks the significant
> onset of learning confusion patterns, namely prediction changes (PC). **Therefore, if
> the training process lacks a stage of learning confusion patterns, such as when training
> with perfect data or employing robust regularization approaches, the original Label Wave
> method may not identify an appropriate stopping point.**
>
> However, it is important to note that in these scenarios, applying early stopping to
> improve the model's generalization performance might not be necessary.

Appendix B, verbatim: *"Learning Rate: **Fixed at 0.01**."*
Appendix C.4, verbatim: *"Learning Rates (LR.): 0.01, 0.05, 0.001."*

Whole-paper term sweep: **cosine 0 · anneal 0 · schedul 0 · decay 0 · warmup/warm-up 0**.
Also `model.eval` 0, `eval mode` 0, `no_grad` 0 — the paper never specifies how the
predictions behind PC are produced, and gives no code link.

---

## FACT 1c — applicability verdict

| | paper Eq. (2) | logged `pred_flip_count` | match |
|---|---|---|---|
| set | 𝒟 = full noisy training set | full training set | yes |
| comparison | argmax(t) vs argmax(t−1) | same | yes |
| normalization | none (a sum) | none (integer) | yes |
| granularity | epoch (Eq. 2/3) / step (Alg. 1) | epoch | yes, with the paper's own internal inconsistency noted |
| mode / transform | **unspecified** | train-mode, augmented | undecidable |
| single-θ snapshot | ŷ<sub>i</sub><sup>t</sup> implies one θ<sub>t</sub> | per-sample timing differs | **mismatch** |

**(a) Applied directly to the logged statistic — valid, and already done.** The
transcription in `src/analysis/label_wave.py` matches the source. Four items remain
NEEDS-VERIFICATION: patience *p* (the paper names it but gives no value), *k* (the paper's
own best-supported value is used), the window warm-up convention, and the unspecified
prediction-production mode.

**(b) Recomputed from the 24 retained checkpoints — inferior, and not Label Wave.** The
checkpoint grid is 5 epochs apart, so such a recomputation measures
1[ŷ<sup>t</sup> ≠ ŷ<sup>(t−5)</sup>], not Eq. (2)'s *t−1* statistic, and a k = 3 moving
average would span 15 epochs rather than 3. Formally rejected; Label Wave is evaluated
only on the logged one-epoch statistic.

**(c) Retraining required — no.**

### The schedule-exclusion claim, checked in both directions

- **Does the source support it?** Only partly. C.3's failure conditions are (i) very low or
  absent label noise and (ii) the absence of a learning-confusion stage — perfect data
  *or robust regularization*. Learning-rate schedules are not mentioned.
- **Does the source contradict it?** No. The paper never claims operation under an annealed
  schedule and never evaluates one; every experiment uses a fixed LR.

The exclusion therefore rests on two pillars that must be kept apart:

- **(P1) Out-of-regime** — the paper's own text. Its validated regime is fixed LR
  (Appendix B "Fixed at 0.01"; C.4 sweeping three fixed values). This is an
  absence-of-support argument, not a claim the paper makes.
- **(P2) Structural argument — ours, not the paper's.** Under annealing, inter-epoch churn
  decays because the optimization temperature fell, whether or not memorization is
  underway, so a churn statistic cannot separate the two.

C.3 supports the ELR / robust-regularization exclusion, not the schedule exclusion. The
correction is committed in `015b355`.

---

## FACT 2 — oracle grid provenance

`src/analysis/ncr.py:32-37`:

```python
def oracle_epoch(risk, smooth_w=3):
    """Return (t_star_smoothed, t_star_raw), 0-indexed argmin of the risk curve."""
    risk = np.asarray(risk, dtype=np.float64)
    t_raw = int(np.argmin(risk))
    t_sm  = int(np.argmin(moving_average(risk, smooth_w)))
```

`np.argmin` is unconstrained — the whole input array is the candidate set. The input is the
length-120 vector `risk_trajectories` builds from `run.epochs`
(`src/analysis/io.py:101-108`), i.e. every logged epoch. Protocol §4 says "three
**length-120** vectors"; §5 says `t*_j = np.argmin(moving_average(R_j, 3))`. Meanwhile a
selector's `stop_epoch` can only be one of the 24 retained checkpoints, and
`src/analysis/selection.py:28` subtracts the two directly.

**Verdict: the oracle ranges over 120 points while the selector ranges over 24. The
review's objection is factually correct in code.** A grid-restricted oracle is recomputable
from the existing logs alone — restricting the argmin's candidate set is a pure lookup, no
inference required.

Structural consequence, stated without magnitudes: protocol §5 takes `t*` as the argmin of
the **smoothed** curve but reads `R*` from the **raw** curve, so restricting the candidate
set can land on an epoch whose raw risk is lower than the full-grid oracle's. The effect of
grid restriction is therefore **not sign-monotone**. Measured magnitudes: see exposure
ledger **EXP-001**.

---

## FACT 3 — the worst-class axis

Protocol `docs/analysis_protocol.md:71-96`:

- **static** — take the group's `learner == "ce"`, `seed == 0` run; read
  `per_class_test_error` at its **final** epoch (119); `acc = 1 - error`;
  `k = max(1, int(round(0.30 * C)))` (C = 10 → 3, C = 100 → 30);
  `order = np.lexsort((np.arange(C), acc))`, ascending accuracy with ties broken by
  ascending class index; take the first *k*, then `np.sort` them.
- **dynamic** — re-select that epoch's worst fraction at each epoch
  (`src/eval/tail.py:40-45`).

| question | answer |
|---|---|
| frequency- or performance-based | **performance** — bottom 30% of classes by per-class accuracy |
| which data | the **clean test set**, not the training set |
| fixed or varying | static is fixed once per group; dynamic varies per epoch |
| identical across runs in a cell | static yes (CE seed-0 anchor, shared by all learners); dynamic no |

**Circularity verdict.** The dynamic variant *is* circular — the checkpoint under evaluation
defines its own target set. It is **not an analysis axis**: protocol §4's `tail` axis uses
the §3 static set, `analysis/io.py:104` reads `r_tail_static_series`, and §2 lists
`R_tail_dynamic` among the logged-but-not-analyzed fields. Its only live use is the
forward-pass integrity recomputation, where reproduction against the log is the entire
question and circularity is irrelevant. **The analysis axis is not circular.**

**Naming.** Both clean test sets are exactly class-balanced — verified directly from the
files that train the models: CIFAR-10 test 1000 per class, CIFAR-100 test 100 per class,
min = max in both. No frequency tail exists, so "tail" is an inaccurate name; the set is a
**worst-class subset**. Renamed in the registration (`015b355`); G1's sealed artifacts keep
their field names, with the mapping note that the field named `tail` denotes the worst-30%
of classes by clean-test accuracy.

---

## FACT 4 — the OOD risk functional

Base metric is **1 − AUROC**, not FPR@95. `src/eval/ood.py` docstring: *"AUROC is the
probability a random ID sample scores above a random OOD sample; `1 - AUROC` is the R_OOD
risk logged in the audit."* Computed by the Mann-Whitney rank identity, ties at 0.5,
unit-tested against sklearn.

Per score × pool, `src/eval/evaluate.py:59-65`:

```python
id_msp, id_energy = msp_score(id_logits), energy_score(id_logits, energy_T)
for name, loader in ood_pool_loaders.items():
    ol = _logits(model, loader, device)
    ood[f"msp_{name}"]    = 1.0 - auroc(id_msp,    msp_score(ol))
    ood[f"energy_{name}"] = 1.0 - auroc(id_energy, energy_score(ol, energy_T))
```

Two scores × three pools = six values, all logged. The **aggregation operator**,
`src/eval/evaluate.py:67-69`:

```python
# primary endpoint: energy vs semantic pools (mean over semantic pools)
sem = [ood[f"energy_{p}"] for p in semantic_pools if f"energy_{p}" in ood]
r_ood_primary = float(np.mean(sem)) if sem else None
```

| aspect | value |
|---|---|
| operator | plain unweighted arithmetic mean (`np.mean`) — not min, not max, not pool-size weighted |
| score | energy only; MSP is logged but not part of the primary endpoint |
| pools | semantic only: `[svhn, cross_cifar]` |
| weighting | equal, although the pools differ in size (svhn 26,032 vs cross_cifar 10,000) |
| excluded | `CIFAR-C-local` (covariate) is logged but not in the primary endpoint |
| temperature | `ood_energy_T = 1.0` |

Protocol §4 states the same: *"the mean over the semantic pools [svhn, cross_cifar] of
1 - AUROC of the **energy** score. The covariate pool CIFAR-C-local is logged but is **not**
part of the primary endpoint."*

A near-zero-IQR guard exists at `src/analysis/selection.py:31`
(`normalized_regret = regret / scale if scale > 0 else 0.0`) and is retained by
registration. Measured IQR magnitudes and whether the guard ever fired: see exposure ledger
**EXP-002**.

`CIFAR-C-local` is **not** official CIFAR-10-C/100-C and its numbers are not comparable to
published ones: `imagecorruptions` 1.1.2 over `opencv-python-headless` 4.10.0.84, severity
3, four corruptions, 2,000 images per corruption, subsample and corruption seed 20260811.
