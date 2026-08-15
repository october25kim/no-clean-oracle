# Extension Preregistration — Real-Noise × Learner-Grid Restoration
Registered against: framing_prereg.md (3366144) and
classification_record.md (07ff003). Status: registered BEFORE any
extension run is launched; no real-noise trajectories, checkpoints,
or selector outputs exist at registration time.

## 1. Learner grid and exclusion criteria
Grid: {CE, ELR, SOP, GCE} — unprotected / implicit regularization /
explicit noise modeling / robust loss geometry.

Inclusion criteria (all required):
  K1  single network — checkpoint = one model state.
  K2  no internal sample-selection or loss-mixture machinery —
      a learner that selects samples acts as its own selector and
      contaminates the audit's subject/instrument separation.
  K2' no internalized stopping — a learner that consumes the
      stopping decision inside its training procedure (e.g. PES's
      progressive per-part early stopping) pre-empts the decision
      under audit; such methods are treated in related work as
      prior evidence of per-objective stopping divergence, not as
      comparison learners.
  K3  single continuous trajectory — no stage switching; the
      24-point grid, IQR normalization, and oracle-epoch semantics
      presuppose it.
Excluded by these criteria: DivideMix, ELR+ (K1, K2, K3),
PES/PES-semi (K2', K3), Co-teaching (K1, K2).
Scope boundary: transition-matrix / statistical-correction methods
(Forward, VolMinNet, Peer Loss/CAL) are out of scope — they
introduce the separate problem of T-estimability; recorded as a
limitation, not an exclusion by K-criteria.

## 2. Unified trajectory protocol (Decision A)
All four learners share the G1/B2 generation protocol: PreAct
ResNet-18, 120 epochs, cosine annealing, identical batch size,
augmentation, and determinism settings as the sealed configs.
Learner-specific hyperparameters are transcribed from official
sources and pinned below; official SCHEDULES are NOT transcribed.
Registered defense: the audit examines selectors on trajectories,
not learner SOTA reproduction; trajectory-generation conditions are
therefore unified and the deviation from official schedules is a
stated limitation.

## 3. Learner pins
  ELR: as in G1 (unchanged; sealed configs).
  SOP (NOT SOP+ — the consistency/KL regularizer of SOP+ is
      semi-supervised machinery, excluded for the same reason as
      K1/K3 exclusions):
    - lr_u / lr_v transcribed from the official repository
      (shengliu66/SOP), which supersedes the paper where they
      differ (asymmetric-noise lr_u discrepancy is documented by
      the authors); transcription source committed alongside.
    - Stored logits_train / logits_test are RAW f(x), excluding
      the sparse term s_i = u_i⊙u_i − v_i⊙v_i. u,v are training-
      only per-sample parameters; evaluation uses f only.
    - Registered observation frame: C1's noisy-accuracy trajectory
      under SOP is expected to differ structurally from CE/ELR
      because noise is absorbed into s rather than memorized by f.
      This is an observation target, not a defect; no C1 exclusion
      for SOP may be introduced post hoc.
  GCE: q = 0.7 (paper default); no truncation variant; loss
      replacement only, all else identical to CE.

## 4. Datasets and grid
Tier 1 (this registration launches Tier 1 only):
  {CIFAR-10N worst, CIFAR-10N random1, CIFAR-100N}
  × {CE, ELR, SOP, GCE} × 3 seeds = 36 runs.
  Noise-rate validation targets at data sealing: worst ≈ 40.21%,
  random1 ≈ 17.23%, C100N ≈ 40.20% (per the CIFAR-N release);
  measured rates must match within 0.1 percentage points or STOP.
  Ground-truth labels exist for all samples; noise masks
  (noisy ≠ clean) are derived, frozen, and sha256-sealed before
  any training.
Tier 2 (registered in scope, NOT launched by this document):
  Clothing1M × {CE, SOP} × 2 seeds, pre-trained ResNet-50,
  official SOP Clothing1M optimization config, ~10-epoch
  trajectory, checkpoint grid 2/epoch (≈20 points). Tier 2
  requires a separate pre-launch pin of axis definitions
  (tail := lowest-frequency training classes, exact k pinned;
  OOD pool selection pinned) and a timing probe; it must not
  launch before that amendment is committed.

## 5. Selectors, axes, thresholds
Selectors: E (T-grid, max-softmax primary), C1, (b) effective-rank
argmax — all as pinned in g2 pins; full-D remains exploratory and
excluded from classification. Axes and regret function unchanged
(src/analysis/selection.py::regret_at_epoch); each new run's
oracle epochs are computed from its own measured axes in the same
frame. δ = 0.10 binding, δ ∈ {0.05, 0.20} non-binding sensitivity,
reliability fraction π = 0.8 (≥ 29/36 for Tier 1).

## 6. Outcome branch table (fixed now)
  R1 — OOD success 0/N across all learners and selectors at
       δ ≤ 0.20: the audit claim generalizes from synthetic to
       human noise; headline extends to real-world label noise.
  R2 — some selector achieves reliable success (π) on ALL three
       axes within some learner×dataset cells: conditional result;
       the audit claim is scoped, the qualifying cells are
       reported exhaustively, and the paper's framing shifts to
       "no free lunch except under [registered cell description]".
  R3 — intermediate patterns: reported against this table without
       reframing; any narrative beyond R1/R2 wording requires a
       timestamped amendment.
Adjudication is mechanical against §5 thresholds; interpretive
rulings follow the same convening procedure as
classification_record.md.

## 7. Protocol rules
R1–R3 (as codified) apply. All new-process outputs must carry
code_stamp with git_tree_dirty: false.
