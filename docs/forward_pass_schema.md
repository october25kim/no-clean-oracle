# B2 forward-pass output schema

`docs/analysis_protocol.md` specifies the log-to-verdict derivation; it does not define
a forward-pass output format, because G1 never produced one. This file fills that gap
for the approved B2 inference sweep, so that its four consumers — (b) representation
analysis, (c)/C1 noisy-validation, full-D, and E — all read one artifact set and no
second sweep is needed.

## Layout

```
results/forward_b2/<run_id>/ep{NNN}/
    logits_train.npy      float32 (N_train, C)   noisy training set, eval transform
    feats_train.npy       float32 (N_train, 512) penultimate embedding, same pass
    logits_test.npy       float32 (10000,  C)    clean test set
    logits_<pool>.npy     float32 (N_pool, C)    one per frozen OOD pool
    meta.json             provenance + integrity for this checkpoint
results/forward_b2/<run_id>/labels.npz            noisy + clean train labels, test labels
results/forward_b2/MANIFEST.sha256                every file above
```

`NNN` is the checkpoint's epoch, zero-padded: `ep004 … ep119`, 24 per run.

## Why these four arrays

| consumer | needs | served by |
|---|---|---|
| (b) representation | penultimate embeddings per checkpoint | `feats_train.npy` |
| (c)/C1 noisy-validation | per-sample scores against noisy labels on a train subset | `logits_train.npy` + `labels.npz` |
| full-D loss separation | per-sample CE against the noisy label | `logits_train.npy` + `labels.npz` |
| E confidence / entropy | per-sample softmax over the train set, and a temperature-scaled variant | `logits_train.npy` (logits, not probabilities, so any temperature is applied post hoc) |
| verdict cross-check | R_ID, R_tail, R_OOD recomputed at the 24 retained epochs | `logits_test.npy`, `logits_<pool>.npy` |

Logits are stored rather than probabilities or reduced statistics on purpose: every
statistic any of the four consumers needs is a function of the logits, so storing them
means a change of statistic never requires another GPU pass.

## Inference configuration (pinned)

Identical to the B2 training environment so the outputs sit in the same numerical
frame as the G1 trajectories:

| | |
|---|---|
| mode | `model.eval()` under `torch.no_grad()` |
| precision | fp32; no AMP, no autocast |
| batch size | **512** — not unset: `eval.eval_batch_size` in the sealed `configs/base.yaml`, the same value every G1 and B2 evaluation used |
| dataloader | `shuffle=False`, deterministic order |
| eval transform | `ToTensor` + `Normalize` only — **no randomness at eval**, verified, so no seed is required for the transform |
| determinism | `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| checkpoints | `checkpoint_ep{NNN}.pt` only; `latest.pt` is a resume artifact and is never read |

The train set is passed through the **eval** transform in **eval** mode. This is a
deliberate departure from how G1's logged `train_loss_quantiles` were produced (those
came from augmented, train-mode forward passes during SGD), and it is the right choice
for a single-snapshot statistic: it removes augmentation noise and BatchNorm
batch-statistics from the measurement. The consequence must be carried wherever the two
are placed side by side — **full-D's per-sample losses are not directly comparable to
the logged `train_loss_quantiles`**, and coarse-D (computed from those logs) therefore
differs from full-D by more than resolution alone.

## meta.json per checkpoint

```
run_id, dataset, learner, seed, epoch, checkpoint_sha256,
n_classes, n_train, n_test, pools:{name: n},
batch_size, precision, torch, cuda, cudnn, gpu_uuid,
cublas_workspace_config, deterministic_algorithms, eval_transform,
reference_frame: "G1",
recomputed:{R_ID, R_tail_dynamic, R_OOD_primary_energy_semantic},
logged:{...}, max_abs_deviation_vs_log
```

`recomputed` versus `logged` is a built-in integrity check: the metrics recomputed from
these logits at epoch `NNN` must reproduce what the B2 run logged at that epoch. A
non-zero deviation would mean the forward pass is not reproducing the training-time
evaluation, and would invalidate the outputs before any consumer reads them.

---

# Revision: Tier 1 reading (registered 2026-08-14, owner-approved)

R1 follow-up revision, not a rewrite: everything above continues to describe the B2 pass
exactly as it ran. The Tier 1 real-noise pass (`scripts/forward_ext.py`, 36 runs x 24
checkpoints) writes the same layout under `results/forward_ext/<run_id>/ep{NNN}/` with
the same four arrays, the same pinned inference configuration, and the same
`meta.json` fields. Four field-level readings differ, because the campaign differs.

**`reference_frame: "ext_tier1"`**, not `"G1"`. The extension is its own self-contained
frame per the extension preregistration: its oracles, IQRs and regrets are computed
within the Tier 1 runs and are never pooled with G1 or B2. Recording `"G1"` here would
assert a comparability the preregistration explicitly declines.

**`noise_provenance`** is added, carrying `{kind: "real", split, npz_sha256,
measured_noise_rate_pct}`. The B2 rows needed no such field: their labels were
reconstructible from a transition matrix plus a seed. A sealed CIFAR-N split is an
artifact, so the artifact's digest travels with every checkpoint that depended on it, and
`forward_ext.py` re-verifies it against both the seal and the run's own metadata before
writing anything.

**`sop_raw_logit_assertions`** is added, 1 on every SOP checkpoint and 0 elsewhere
(registered expectation: 9 SOP runs x 24 = 216). The preregistration requires SOP's
stored logits to be the raw f(x) with the sparse term `u^2 - v^2` excluded;
`train.sop.assert_raw_logits` recomputes `f(x)` and demands bitwise equality with the
logits about to be written, so the property is checked rather than trusted.

**`integrity_class: "trajectory-identity (tolerance exactly 0)"`** is added, and the
tolerance is enforced rather than reported. Under R4 this recomputation walks the same
path the training-time evaluation walked, so exactly 0 is the only acceptable value —
B2 returned 0.000e+00 on all 360 of its checkpoints, which is what makes the bar
attainable rather than aspirational. A non-zero deviation writes
`ABORT_shard_<tag>.json` and stops that shard immediately, instead of spending hours
producing outputs no consumer would be permitted to read.

`dataloader_workers` is recorded per checkpoint. It is numerically inert: the eval
tensors are materialized once in the parent process and iterated with `shuffle=False`,
so a worker moves batches but never produces a value.

On completion the pass writes `results/forward_ext/MANIFEST.sha256` — every produced
file, sorted by repo-relative path, excluding the manifest itself — and reports the
manifest's own sha256, so a single value attests the whole output tree.
