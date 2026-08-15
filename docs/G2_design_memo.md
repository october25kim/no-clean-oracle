# G2 feasibility memo — what the G1 artifacts can and cannot support

Analysis-only assessment, no implementation. Written against the completed G1 sweep
(48 runs x 120 epochs, 16 cells) as it exists on disk. Everything below was checked
against the artifacts rather than assumed; where an earlier assumption turned out to
be wrong, that is stated first.

**Headline:** (a) is free, (c) is cheap but biased and needs a design choice, and
(b) is blocked — the per-epoch checkpoints needed for a representation-based baseline
were never retained.

---

## Status of record for Label Wave (owner decision, 2026-08-12)

**Main-table status:** *not directly applicable under the canonical cosine schedule
(premise violation, on two pillars — see "Basis for the schedule exclusion" below;
the paper's own Appendix B fixes LR = 0.01).* The measured numbers live in an appendix, annotated as
premise-violated, with the k/patience grid shown so the conclusion's robustness is
visible and `patience` flagged needs-verification (the paper gives it no value).

**Binding:** Label Wave's degradation does **not** count toward the G2
baseline-insufficiency gate. That gate must be satisfied by *applicable* baselines
only — see "Applicability criterion" below.

**Reframe for the paper.** The LR->0 prediction freeze is not a quirk of one method;
it is a structural observation. Any selector built on prediction *churn* reads a
quantity that conflates optimization temperature with learning dynamics: as the
learning rate anneals, predictions stop moving whether or not memorization is
underway, so the churn signal decays for a reason that has nothing to do with the
phenomenon it is supposed to detect. Schedule-coupled validation-free selectors
therefore do not transfer to modern annealed schedules — which is support for the
thesis, not an embarrassment for it.

For the record: G1's cosine schedule was sealed in `configs/base.yaml` **before** Label
Wave was evaluated, so the schedule was not chosen to produce this result.

### Basis for the schedule exclusion — two pillars, correctly attributed

An earlier version of this memo cited Appendix C.3 as predicting the schedule failure
mode. A primary-source check (2026-08-13, arXiv:2502.07551 HTML, sha256
`10fef4ee92aa6174cdd1d2de01aa78400ebb49f883073bd206ee561488ee973f`) showed that it does
not, and the citation is withdrawn. C.3's stated limitations are low/absent label noise
and the absence of a learning-confusion stage; it says nothing about learning-rate
schedules. A whole-paper term sweep returns **zero** occurrences of *cosine*, *anneal*,
*schedul*, *decay*, *warmup* / *warm-up*. The exclusion rests on two pillars instead:

- **(P1) Out-of-regime — the paper's own text.** Every regime the paper validates uses a
  fixed learning rate. Appendix B, verbatim: "Learning Rate: **Fixed at 0.01**."
  Appendix C.4 sweeps fixed values, verbatim: "Learning Rates (LR.): 0.01, 0.05, 0.001."
  Cosine annealing therefore lies outside the evidence base the paper offers. This is an
  absence-of-support argument, not a claim the paper makes.
- **(P2) Structural argument — OURS, not the paper's.** Under an annealed schedule,
  inter-epoch prediction churn decays because the optimization temperature fell, whether
  or not memorization is underway, so a churn statistic cannot separate the two. This
  inference is ours; the paper neither states nor contradicts it, having never evaluated
  an annealed schedule.

Neither pillar licenses the sentence "the paper predicts this." What the paper's own
text does support directly is the ELR exclusion below, via C.3's "robust regularization
approaches" clause.

**Option (i), fixed-LR auxiliary — TABLED** as a G2-design-review agenda item. If run,
it is scoped as a separate **regime study** (a different training trajectory, not a
same-trajectory baseline, and therefore not comparable to the G1 verdict runs), and
**CE cells only**: per Appendix C.3 ELR violates the premise regardless of the learning
rate, because robust regularization removes the learning-confusion stage the method
keys on. No execution before the design review.

## Applicability criterion — adopted as the paper's baseline selection principle

> **Principle (owner-adopted 2026-08-12).** A validation-free selector's signal must
> not be a function of inter-epoch prediction churn under an annealed schedule.

This is not a local convenience for the G2 gate; it is the rule the paper uses to
decide which prior methods are in scope. The justification is the structural one above:
churn measures how far predictions moved between epochs, and under an annealed schedule
that quantity decays because the optimization temperature fell, whether or not
memorization is underway. A method resting on it cannot distinguish the two, so
evaluating it on an annealed run tests the schedule, not the method.

A baseline counts toward the gate only if it satisfies this principle. Concretely, a statistic
computed from a single checkpoint's state or outputs is applicable; a statistic
defined as a difference between consecutive (or lagged) epochs inherits the same
temperature confound Label Wave hit, and is conditional at best.

This criterion has an immediate consequence worth flagging, because it cuts against
one of the two directions suggested: **EMA / snapshot-agreement is a churn signal.**
Disagreement between the predictions of checkpoints `t` and `t-d` is exactly
inter-epoch churn with a longer lag, so under cosine annealing it decays for the same
mechanical reason. It can still be run, but it would land in the same conditional
bucket as Label Wave rather than satisfying the gate. The two proposals below are
chosen specifically to avoid that trap.

### Proposed applicable baseline D — loss-distribution separation (single-snapshot)

The small-loss family: under label noise the per-sample training loss becomes bimodal,
clean examples in the low mode and mislabeled ones in the high mode, and the
separation collapses as the model memorizes. It is read off **one** epoch at a time, so
no churn is involved.

- **From existing logs, coarse:** `train_loss_quantiles` gives p10/p25/p50/p75/p90 per
  epoch for all 48 G1 runs. A spread statistic such as `p90 - p50` or
  `(p90 - p50) / (p50 - p10)` is a usable proxy for the separation. **Free, CPU, now.**
  Its limitation is real: five quantiles cannot support the two-component mixture fit
  the literature versions (e.g. GMM-based) actually use.
- **Full version:** needs per-sample losses. Two routes — log them during any future
  run (50,000 float32 = 200 KB/epoch, 24 MB/run, negligible), or recompute them from
  the B2 checkpoints with one forward pass over the training set per checkpoint
  (24 passes/run, minutes of GPU each, shares the (b) re-evaluation).
- **Applicability:** clean. The absolute loss scale shifts as the LR anneals, but the
  *shape* separation between the two modes does not depend on predictions still moving.

**Coarse-D measured (2026-08-12, exploratory, non-decisional)** —
`results/report/coarse_d_exploratory.json`, all 48 G1 runs, peak of each spread
statistic under the same 3-epoch smoothing as the analysis. Oracle epoch medians for
reference: ID 81, tail 67, OOD 48.

| statistic | median stop | median epoch gap (ID / tail / OOD) | median normalized regret | median rank agreement |
|---|---|---|---|---|
| p90 - p50 | 52 | -17 / -8 / +14 | +0.67 / +0.76 / +1.21 | +0.12 / +0.07 / -0.03 |
| p90 - p10 | 52 | -21 / -9 / +6 | +0.63 / +0.67 / +1.24 | +0.17 / +0.11 / +0.06 |
| p75 - p25 | 24 | -25 / -15 / -6 | +0.99 / +1.24 / +0.95 | +0.05 / -0.05 / +0.16 |
| upper / lower | 93 | +25 / +40 / +53 | +0.98 / +0.85 / +1.23 | +0.14 / +0.15 / -0.11 |

Rank agreement is `-Kendall tau(statistic, risk)`: positive means the statistic orders
epochs the way that objective's risk does.

**Reading:** the coarse proxy carries essentially no ordering information — every rank
agreement is within +/-0.17 of zero, and the selected epochs cost 0.6 to 1.2 IQR
against every objective, i.e. no better than picking an epoch arbitrarily. The four
statistics also disagree with each other (median stop 24 to 93), which is what a
signal-free statistic looks like.

**What this does and does not license.** It does *not* condemn baseline D. Five
marginal quantiles cannot express two-mode structure: `p90 - p50` grows if the
mislabeled mode separated, and equally if the whole distribution merely stretched, so
the proxy is blind to the exact feature the method depends on. The honest conclusion is
that **the coarse route is not a substitute and the question is unresolved until the
full version runs** on per-sample losses. Cost of resolving it is low — the forward
passes are already required by (b), (c)/C1 and E — so D should stay in the applicable
set for the design review, with the caveat that its evidence so far is null.

### Proposed applicable baseline E — single-snapshot prediction confidence / entropy

Mean max-softmax, or mean predictive entropy, over the training set at each retained
checkpoint. Memorization drives confidence on mislabeled examples up, so the aggregate
moves characteristically without any reference to a previous epoch.

- **Cost:** one forward pass per retained checkpoint — the same pass baselines (b),
  (c)/C1 and D-full already need, so bundling them costs one sweep of inference, not
  four.
- **Applicability:** clean, single-snapshot.
- **Caveat to state if used:** confidence is calibration-sensitive, and cosine
  annealing does sharpen softmax outputs late in training. Unlike churn this is a
  monotone scale effect rather than a signal-destroying one, but it should be reported
  with a temperature-normalized variant alongside the raw one.

> **PRE-REGISTERED PREDICTION for E (recorded 2026-08-12, before E is run).**
> Training-set confidence *rises with* memorization — fitting the corrupted labels is
> precisely what makes the model confident on them — so E is predicted to **select late
> epochs**, biased past the oracle epochs in the same direction on all three
> objectives, with the largest gap on OOD (the earliest oracle). Crucially this is a
> **directional bias, not signal destruction**: the statistic still moves with the
> phenomenon, it is just offset, which is why E remains applicable under the principle
> above while churn-based selectors do not. Recording the prediction now makes the
> eventual measurement a forecast that can be wrong, rather than a story fitted after
> the fact. The temperature-normalized variant is reported alongside the raw one, and
> if normalization removes the late bias that is itself the finding.

> **AMENDMENT to the E record — 2026-08-13 06:45 KST (owner-approved).**
> The text above is unchanged; this is an addition, not an edit.
>
> **T-selection rule specified post-registration, pre-analysis; no E outputs computed
> or inspected at time of this amendment.** The original record required a
> temperature-normalized variant without saying how `T` is obtained, and the obvious
> route — fitting `T` on clean held-out labels — would break the validation-free
> premise the baseline exists to test. The rule is therefore fixed here, before any E
> statistic exists.
>
> 1. **Primary method:** fixed grid `T ∈ {0.5, 1, 2, 5}`, with a selection-epoch
>    stability check across the grid.
> 2. **Decision criterion, fixed now:** per run, if the selected epoch is identical at
>    all four `T` values, report **"T-robust"**; if it differs at any `T`, report
>    **"T-sensitive"** and publish the full epoch-by-`T` table. No other criterion may
>    be substituted once results are seen.
> 3. **Secondary method:** per-sample logit normalization, computed and reported
>    alongside. It gates no conclusion.
> 4. **Alternative, specified but NOT primary:** a label-free per-epoch `T_t` chosen so
>    that mean logit norm is constant across epochs. Registered as legitimate; not to
>    be computed without a follow-up instruction.
>
> **State of the evidence at amendment time.** The B2 forward pass was mid-flight and
> had written training-set *logits* — which are the shared **input** to four consumers,
> not an E result. No confidence, entropy, temperature-scaled, or normalized statistic
> had been computed, and none had been inspected. No E computation code existed in the
> tree. The commit carrying this amendment is the timestamp evidence that the
> specification preceded the analysis.

**Gate arithmetic after this decision.** Applicable baselines available to the gate:
(b) representation-based, (c) noisy-validation C1, plus D and E. Label Wave and any
snapshot-agreement variant sit outside the gate as conditional/regime results.

## (a) Label Wave — COMPUTABLE from existing logs, but the signal is degenerate here

**Availability: confirmed.** `pred_flip_count` is populated in **5,712 / 5,760** epoch
records. The 48 nulls are exactly epoch 0 of each run, which has no predecessor to
compare against; epochs 1-119 are complete in every run. The field is
`int((cur_pred != prev_pred).sum())` over all 50,000 training samples, i.e. the count
of training-set prediction changes between consecutive epochs — the statistic a
Label-Wave-style stopping rule is built on. No re-training and no re-evaluation is
required; cost is a few seconds of CPU.

**But the selection rule will not work as-is.** The flip count in these runs decays
monotonically toward zero as training converges, so its minimum lands at the end:

| quantity | median epoch over 48 runs |
|---|---|
| argmin of `pred_flip_count` | **118** (at epoch >= 118 in **32 / 48** runs) |
| argmin of `R_ID` | 64 |
| argmin of `R_OOD` | 42 |

A rule that stops at the minimum of the raw flip count would select the last epoch
almost always — precisely the checkpoint G1 shows to be badly degraded (CE at
`cifar10_symmetric_0.2`: `R_ID` 0.1272 at its oracle epoch 28 versus 0.1709 at the
end). So the honest statement is: **the input is available, the naive readout is
not informative on this data.** Any G2 use of Label Wave must therefore either

- reproduce the paper's actual criterion rather than a bare argmin (e.g. a turning
  point / rise-then-fall detector over a smoothed flip curve, evaluated in the early
  regime), and be validated against our oracle epochs; or
- be reported as a baseline that fails on this setup, which is a legitimate and
  cheap G2 result.

Either way the decision is a *method* decision, not a data-collection one. One
caveat to record: `pred_flip_count` is a scalar aggregate. If the criterion needs
per-sample flip histories or flips measured against the noisy label rather than
against the previous prediction, that is **not** recoverable from the logs and would
need re-instrumented training.

## (b) TopoGeoScore-style (representation-based) — BLOCKED, and not by resolution

**The premise in the request does not hold.** The assumption was ~13 checkpoints per
run (every 10 epochs plus final). In fact there is exactly **one `.pt` per run — 48
files for 48 runs**, all named `checkpoint.pt`.

`Trainer._save_ckpt` writes to a single fixed path and `os.replace`s it
(`src/train/trainer.py`). The every-10-epochs saves all overwrote each other, so what
survives is only the **final epoch (119)** state, kept for crash-resume, not a
trajectory. A single terminal checkpoint cannot support checkpoint *selection* at all:
there is nothing to select among.

So the question is not whether 13-point resolution is workable — it is that we have
1-point resolution. Options, in increasing cost:

| option | what it costs | what it buys |
|---|---|---|
| **B1. Re-run with checkpoint retention** | full sweep again: 48 runs x ~1.96 h / 4 GPUs ~= **24 h wall-clock co-located**, plus **~4.3 GB x 13 ~= 56 GB** of checkpoints (90 MB each; 48 x 13 = 624 files) | exact reproduction — seeds, masks and determinism are all fixed, so re-running reproduces the same trajectories and the existing G1 verdict remains valid |
| **B2. Re-run a subset** | e.g. the 5 ID<->OOD-conflict cells only: 15 runs, ~8 h on 4 GPUs, ~18 GB | representation baselines only where the conflict actually is, which matches the adopted ID<->OOD re-scope |
| **B3. Denser retention** | every 5 epochs = 24 points, ~112 GB for the full sweep | finer selection grid if 12 points proves too coarse |

Recommendation: **B2**, gated on the re-scope. It confines the cost to the cells the
project now cares about, and it is the only option whose output is directly comparable
to the primary axis.

Independent of which is chosen, a one-line fix must land first — `_save_ckpt` should
write `checkpoint_ep{epoch:03d}.pt` and keep the resume pointer on a separate path.
Without that change any re-run reproduces the same single-file loss.

Note also that 12 retained points would give a **10-epoch selection grid**, while G1's
measured oracle-epoch separation has a median of 10.5 epochs on the ID<->OOD axis. A
10-epoch grid is therefore at roughly the same scale as the effect being resolved —
marginal. If B2 is approved, retaining every 5 epochs for those 15 runs is the better
trade and costs no extra compute.

> **Storage correction (2026-08-12).** An earlier draft of this table put 5-epoch
> retention for the 15 B2 runs at ~9 GB. That was wrong. A checkpoint is 89.9 MB
> (model + SGD momentum + `prev_pred`), 120 epochs at every-5 is 24 per run, so B2 is
> **24 x 89.9 MB x 15 = ~32 GB**, not 9 GB. The B1 and B3 figures in the table above
> are consistent (48 runs x 24 x 89.9 MB = ~104 GB); only the B2 line was understated.
> 32 GB is still comfortable against 12 TB free, so the recommendation is unchanged.
> `latest.pt` is a hard link to the newest epoch file and costs no extra space.

## (c) Noisy-validation selection — no held-out noisy split exists

G1 trains on all 50,000 noisy-labelled training images and evaluates on the clean test
set. There is no noisy validation split, and `configs/base.yaml` defines none. Options:

| option | mechanics | bias to state |
|---|---|---|
| **C1. Carve a noisy val split from the training set (train-subset reuse)** | hold out e.g. 5,000 of the 50,000 noisy-labelled samples; score checkpoints by accuracy against their noisy labels | the samples were **trained on**, so the score is contaminated by memorization and will drift toward the late, memorized checkpoints — the same failure direction as (a). Optimistic exactly where the audit needs pessimism |
| **C2. Re-run holding out a clean-of-training noisy split** | remove 5,000 samples from training, keep their noisy labels as the val signal | unbiased as a noisy-validation baseline, but **changes the training set**, so its runs are no longer the G1 runs and its trajectories are not comparable to the verdict's. Costs a second sweep |
| **C3. Report the baseline as inapplicable** | state that no clean-of-training noisy split was collected in Phase-0 | zero cost, no false comparison |

Recommendation: **C1 for a directional read with the bias stated in the same
sentence**, and C2 only if a noisy-validation baseline has to be a headline number. C1
can reuse the existing artifacts only if per-sample scores are recomputed, which needs
model states — i.e. **C1 inherits (b)'s checkpoint problem**. Practically, C1 and B2
should be bundled into the same re-run rather than costed separately.

## Cost summary

| baseline | new training? | wall-clock | storage | blocker |
|---|---|---|---|---|
| (a) Label Wave | no | seconds (CPU) | none | criterion definition, not data |
| (b) representation-based | **yes** | 8 h (B2, 15 runs) / 24 h (B1, 48 runs) | 9-56 GB | only 1 checkpoint per run survives |
| (c) noisy-validation | C1 with (b)'s re-run; C2 needs its own sweep | shares (b) / +24 h | shares (b) | no held-out noisy split |

**Suggested sequencing:** land the `_save_ckpt` fix, do (a) on existing logs
immediately at no cost, then a single B2 re-run of the 5 ID<->OOD-conflict cells with
every-5-epoch retention serving both (b) and (c)/C1. That is one ~8 h co-located
sweep, ~9 GB, and it covers everything except a headline C2 number.

**Not affected:** the G1 verdict. Nothing proposed here re-runs or re-reads the runs
the verdict was computed from, and the frozen OOD pools and the 24 sealed noise masks
are reused unchanged, so any re-run stays comparable by construction.
