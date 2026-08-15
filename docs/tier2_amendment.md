# Tier 2 Amendment — Clothing1M Axis Definitions and Launch Gate
Follow-up commit to extension_prereg.md §4 (R1: no amends).
Registered BEFORE the Tier 2 probe result is inspected and before
any Tier 2 training exists.

## 1. Launch gate (parallel-execution decision rule, fixed now)
Prerequisites for Tier 2 launch: (a) this amendment committed,
(b) probe completed. Placement rule, mechanical:
  - probe peak VRAM ≤ 11 GB → Tier 2 launches on GPUs 1/3
    CONCURRENTLY with the Tier 1 sweep (pipelined; Tier 1
    redistributes to GPUs 0/2).
  - probe peak VRAM > 11 GB → Tier 2 launches SEQUENTIALLY after
    Tier 1 completes, still on GPUs 1/3 only.
No discretionary override in-session; a change requires a
timestamped follow-up commit before launch.

## 2. Grid and trajectory
Clothing1M × {CE, SOP} × 2 seeds = 4 runs. Pre-trained ResNet-50
(ImageNet), official SOP Clothing1M optimization config for both
learners (batch 64, lr 0.001, ÷10 after epoch 5, 10 epochs, SGD
momentum 0.9; weight decay θ=0.001, u,v excluded). Checkpoint
grid: 2 per epoch (end of each half-epoch by iteration count),
20 points per run. Rationale recorded: the short pretrained
trajectory compresses selection dynamics; 2/epoch restores grid
resolution comparable to Tier 1's 24 points.

## 3. Axis definitions
- R_ID: error on the clean 10k test set (all 14 classes).
- R_tail: error on the clean test restricted to the bottom-k
  classes by NOISY-label training frequency, k = 4. Frequencies
  computed from the noisy labels only (no clean-label usage in
  the definition; validation-free premise preserved). The k=4
  membership list is computed once at data sealing and frozen.
- R_OOD: two pools, both evaluated with msp and energy scores as
  in the G1 protocol:
    OOD-near = C1M-C-local: the corruption suite applied to the
    clean test images (same generation convention as
    CIFAR-C-local; corruption types and severities pinned at
    sealing).
    OOD-far = [PIN at sealing: one disjoint-domain natural-image
    pool, resized to 224; candidate list fixed from what is
    licensable and locally available — needs verification before
    sealing; the choice is committed before any Tier 2 forward
    pass].
- Oracle epochs: per-run, from the run's own measured axes
  (self-contained frame, as Tier 1).

## 4. Registered limitations
- Clothing1M lacks full clean-label verification (only the test
  set is verified); oracle definitions rest on the 10k clean
  test. Tier 2 is therefore an existence check at scale under
  pretrained initialization, not a statistical pillar; Tier 1
  carries the aggregate claims.
- The 20-point grid and 10-epoch trajectory are not directly
  comparable to Tier 1 regrets; Tier 2 results are reported in
  their own table and never pooled with Tier 1 in classification.

## 5. Selectors
E (T-grid), C1, (b) effective rank — unchanged pins. C1 split:
N = 5,000 stratified by noisy label over 14 classes; per-class
quota = floor(5000/14) = 357, remainder 2 distributed by
ascending class index; same sampling rule as the g2 pin,
rng seed 20260813.

---

## History — probe execution note (follow-up, 2026-08-13)

The §1 probe was executed with shape-accurate synthetic batches (identical
model/batch/resolution/optimizer; fp32) because Clothing1M requires an author access
agreement and was not locally present. Peak VRAM is content-independent; the branch
ruling stands. The probe's timing figure (0.5462 s/iter, 2.37 h/epoch) is a
compute-only lower bound; wall-time calibration is deferred to first real-data epoch.

Measured: peak reserved 6.22 GiB (allocated 5.53 GiB) on GPU 3
(`GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f`, co-tenant 5.63 GB), batch 64, 224 px,
14-class head on an ImageNet-pretrained ResNet-50. §1 branch fired: **CONCURRENT**,
Tier 2 on GPUs 1/3 alongside Tier 1. Placement is unchanged by the measurement being
well under the threshold — §1's no-override clause governs.
