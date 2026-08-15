# Task 3 — does the B2 forward pass contain everything E needs?

Checked against E's frozen record in `docs/G2_design_memo.md` **before** the sweep
finished, so anything missing could be folded into the same pass. No E results were
computed.

## Requirement-by-requirement

| E requires | source | status |
|---|---|---|
| mean max-softmax over the training set, per checkpoint | `logits_train.npy` (50,000 x C, fp32) | **covered** |
| mean predictive entropy over the training set, per checkpoint | same array | **covered** |
| a temperature-normalized variant | logits are stored **unnormalized**, so any temperature is applied post hoc without another pass | **covered as data**; see the gap below |
| all 24 retained epochs per run, to select an epoch | 24 `ep{NNN}/` directories per run, `latest.pt` excluded | **covered** |
| per-objective oracle epochs to measure the predicted late bias against | G1 logs, already on disk; reference frame is G1 (certified) | **covered** |
| the prediction's "largest gap on OOD" test | `R_OOD_primary_energy_semantic` oracle from the G1 logs | **covered** |
| per-sample confidence, should the aggregate need decomposing | logits are per-sample, not pre-aggregated | **covered** |
| separation of clean vs mislabeled samples in the confidence signal | `labels.npz` carries both `train_noisy` and `train_clean` | **covered** |

Storing logits rather than reduced statistics is what makes this list close cleanly: any
statistic E might want — raw confidence, entropy, any temperature, a clean/mislabeled
split, a per-class breakdown — is a function of arrays already on disk.

## One gap, and it is a specification gap, not a data gap

**How is the temperature obtained?** The record says a temperature-normalized variant is
reported alongside the raw one; it does not say how `T` is fitted. This matters because
the obvious method is not available:

- Fitting `T` on the clean test set would **break the validation-free premise** — the
  whole point of these baselines is selecting without clean held-out labels. A selector
  calibrated on clean labels is not the selector being evaluated.
- Fitting `T` on the noisy training set is circular: it is the same data whose
  confidence the statistic is measuring.

Legitimate options, all computable from the stored logits with no further GPU work:

1. **Fixed grid** — report the statistic at `T` in e.g. {0.5, 1, 2, 5} and show whether
   the selected epoch is stable across it. Assumes nothing, answers "is the late bias an
   artefact of scale?" directly.
2. **Scale-free normalization** — divide each sample's logit vector by its own norm (or
   standardize it) before the softmax, removing per-epoch magnitude drift without
   fitting anything to labels.
3. **Per-epoch temperature matching** — choose `T_t` so that some label-free quantity
   (e.g. mean logit norm) is constant across epochs, isolating shape change from scale
   change.

**Recommendation:** option 1 as primary — it is assumption-free and directly tests the
caveat that motivated the variant — with option 2 as a secondary. Both are cheap and
neither needs new inference.

**No second sweep is required either way.** This is flagged now only so the choice is
made deliberately at design review rather than settled by whoever writes the code.

## Also confirmed

- **No eval-time augmentation exists.** `eval_transform` is `ToTensor` + `Normalize`
  only, with no random component, so the instruction's STOP condition does not fire and
  no transform seed is needed. The training set is passed through this eval transform in
  eval mode, which is the correct choice for a single-snapshot statistic but makes these
  per-sample losses **not comparable to the logged `train_loss_quantiles`** (produced by
  augmented, train-mode passes) — carried in `docs/forward_pass_schema.md`.
- `latest.pt` is read by nothing in the sweep.
