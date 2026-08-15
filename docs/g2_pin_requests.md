# Pin requests — C1 and (b), blocking their confirmatory runs

Raised 2026-08-13 under the g2_baselines instruction, whose Task 2 and Task 3 each carry
an explicit STOP: do not improvise a split, do not improvise a specification. Both STOPs
fire. Task 1 (E) was fully specified and has been executed.

---

## Task 2 — C1 noisy-validation: **no split or statistic is pinned anywhere**

The instruction says to use "the noisy held-out split as pinned in
`docs/analysis_protocol.md`". Checked:

| where | result |
|---|---|
| `docs/analysis_protocol.md` — "validation" | **0 occurrences** |
| `docs/analysis_protocol.md` — "split" | **0 occurrences** |
| `docs/analysis_protocol.md` — "C1" | **0 occurrences** |
| `configs/base.yaml`, `configs/b2.yaml` — any val/holdout key | **none** |

The protocol pins nothing here because nothing existed to pin: G1 trained on all 50,000
noisy-labelled images and evaluated only on the clean test set. `docs/G2_design_memo.md`
states this outright — *"There is no noisy validation split, and `configs/base.yaml`
defines none"* — and then offers **C1 / C2 / C3 as options with a recommendation**, which
is a proposal, not a specification.

What a runnable C1 needs, none of which is currently committed:

1. **Split size** — the memo says "e.g. 5,000 of the 50,000", explicitly an example.
2. **Split membership** — the seed and the sampling rule (uniform? class-stratified over
   noisy labels? over clean labels?). It must be derivable from committed configs, since
   the masks are sealed; an ad-hoc draw here would not be reproducible by the reviewer.
3. **Statistic** — accuracy against the noisy labels, or cross-entropy against them. The
   instruction says "accuracy/loss"; these are different selectors and can pick different
   epochs.
4. **Direction** — argmax accuracy / argmin loss, presumably, but it should be stated.

A further constraint worth stating before the pin is chosen: **C1 as scoped is
train-subset reuse**, so whichever split is chosen was trained on, and its score is
contaminated by memorization in the optimistic direction. That is already on record in
the memo. It bounds what a C1 result can support regardless of how the split is pinned.

**Requested:** split size, sampling rule + seed (as a committed config key), statistic,
and direction. Once pinned, C1 runs from the stored `logits_train.npy` and `labels.npz`
in minutes on CPU — no new inference.

---

## Task 3 — (b) representation baseline: **the memo specifies feasibility, not a method**

The memo's (b) section is titled *"TopoGeoScore-style (representation-based) — BLOCKED,
and not by resolution"* and is entirely about whether checkpoints exist. It contains no
metric, no statistic, no clustering choice, no selection rule. Grepping the section for
`metric|cluster|distance|method|score|statistic` returns nothing.

So this is not a small degree of freedom to fill in — the method itself is unspecified.
Inputs are ready: `feats_train.npy`, 50,000 × 512 fp32 per checkpoint, 24 per run,
attribution certified.

### Options, with what each additionally requires

| option | statistic | extra choices it forces | cost |
|---|---|---|---|
| **B-i. Spectral / effective rank** | effective rank (or participation ratio) of the embedding covariance at each checkpoint | essentially none — no metric, no k, no labels | seconds per checkpoint, CPU |
| **B-ii. Neighborhood label consistency** | fraction of each point's k nearest neighbours in embedding space carrying the same noisy label | k; distance metric (Euclidean vs cosine); whether to L2-normalize first; exact vs approximate NN | minutes per checkpoint (50k × 512 kNN), CPU-feasible |
| **B-iii. Class-conditional scatter** | ratio of within-class to between-class scatter, classes taken from the noisy labels | which label set; whether to normalize; Fisher ratio vs trace ratio | seconds per checkpoint |
| **B-iv. Persistent-homology summary** | a topological summary of the embedding point cloud | filtration, homology dimension, summary functional, subsampling scheme, plus a new dependency | hours; heaviest by far |

All four are single-snapshot and therefore satisfy the applicability principle
(no inter-epoch churn).

**Recommendation: B-i (spectral / effective rank).** It is the only option that adds
essentially no free parameters — no k, no metric, no label choice, no library — which
matters precisely because the instruction forbids improvisation: with B-ii or B-iv, most
of the method would be a choice made here rather than registered. B-i is also label-free,
so it cannot leak the clean labels the validation-free premise excludes.

If the intent is specifically to reproduce a TopoGeoScore-*style* result rather than a
representation baseline in general, B-iv is the faithful choice, and it should then be
pinned from the source paper the way Label Wave was — transcribed, not reconstructed,
with any detail the paper omits flagged as needs-verification.

**Also required for whichever option is pinned:** the selection direction (argmax or
argmin of the statistic, or a turning point), and whether any smoothing is applied across
the 24-point grid. E's registered procedure left the analogous choices open; they are
reported as unpinned in `results/report/E_tgrid.json` rather than silently resolved.

---

## Status

- Task 1 (E) — executed, `results/report/E_tgrid.json`.
- Task 2 (C1) — **STOPPED**, awaiting the four pins above.
- Task 3 (b) — **STOPPED**, awaiting a method pin plus direction and smoothing.

No split was drawn, no method was chosen, and no result was produced for either.

---

# PINS RECEIVED 2026-08-13 — verbatim, pre-computation

Registration-time evidence: at the moment of this commit no C1 or (b) output exists
(`results/report/` holds no `C1_noisyval.json` and no `B_representation.json`), and
neither computation script has been executed.

## Protocol rules now in force (permanent)

- **R1.** No history rewriting (amend/rebase) on any committed preregistration or
  amendment document. Corrections go in as follow-up commits. The `d6cc768` → `3366144`
  amend is accepted (pre-result, comment-only) and is the last of its kind.
- **R2.** Any script producing a registered output must be committed BEFORE execution.
  A dirty-tree stamp is a recovery mechanism, not a workflow. If a registered output's
  stamp reports `git_tree_dirty: true`, STOP and report before the output is consumed.
- **R3.** Pre-commit verification of a registered document: when a document must be
  verified before its own commit, it is held outside the working tree during
  verification (tree clean, R2 guard active), then restored and byte-compared
  (`diff -q` or hash equality) against the verified copy before committing. Restoration
  mismatch → STOP.
- **R3-note** (recommended, non-binding): verifiers for registered documents should
  parse expected values from the document rather than transcribing them into the
  script, removing transcription fidelity as a dependency.
- **R4.** Integrity-check tolerances. Comparisons along an identical computation path
  (*trajectory-identity class*) require exactly 0 deviation; recomputations along a
  different aggregation path (*internal-consistency class*) pass at absolute deviation
  ≤ 1e-12. The observed value is always reported, whichever class applies. Which class
  a check belongs to is stated in the check's own output, not inferred by the reader.
- **R5.** Campaign attribution invariant. Within a campaign, every stamped module
  appearing in ≥2 runs must be byte-identical across all its appearances. Per-run
  module SETS may legitimately differ by learner, because `code_stamp()` digests the
  modules the process actually imported and `build_loss` imports the learner lazily
  inside its branch. Verified by `scripts/check_campaign_stamps.py` at every milestone
  report. A campaign-wide `modules_combined_sha256` equality check is NOT the invariant
  and must not be used: it is satisfied only within a learner.
- **R6.** Public anchoring by snapshot. *"Registered commits are made internally, then
  anchored by snapshot push in the same session: the public repository receives the
  updated tree plus the updated ANCHOR_CHAIN.md. No registered analysis executes on real
  logs unless its governing plan appears in a pushed snapshot."*

R3 codifies the resolution used on 2026-08-13, when `docs/classification_record.md` had
to be verified before being committed: an untracked record makes the tree dirty, so the
R2 guard refuses to produce a registered output, while committing first would verify
nothing. Holding the document outside the tree is legitimate only because the verifier
does not read it — it recomputes from the selector JSONs and compares against expected
values. R3-note addresses exactly the weakness that leaves: those expected values were
transcribed by hand into command-line arguments, so their fidelity to the document was
an unchecked dependency. Parsing them from the document removes it.

R4 and R5 were added on 2026-08-13 at the Tier 1 launch, each in response to a check
that was specified more strongly than the thing it was checking.

R4 separates two claims that had been sharing one word. B2's trajectory identity says
the same computation was performed twice and must agree bit-for-bit; exactly 0 is the
only acceptable answer and 1e-16 there would be a real finding. The Tier 1 gate's check 2
says two different aggregation paths over the same logged per-class errors describe the
same quantity; floating-point summation order alone puts the answer near machine epsilon,
and demanding exactly 0 would be demanding an accident. The gate's observed 1.110e-16
is the internal-consistency class and passes. Stating the class in the output is the
operative half of the rule: a bare "max deviation 1.1e-16" invites the reader to supply
their own threshold after seeing the number.

R5 replaces the campaign check the Tier 1 launcher header originally asserted — that
`modules_combined_sha256` is identical across all 36 runs. It is not, and the assert
passed only because the first launched run happened to be CE and so matched the CE gate;
the first ELR run carries a different combined digest for a reason with no bearing on
attribution. The per-module invariant catches what the combined digest was meant to
catch, a source file edited mid-campaign so that later runs trained under different code,
without firing on the learner axis. `check_campaign_stamps.py` was validated against an
injected `src/train/trainer.py` digest change, which it detects.

## C1 PINS

> - Split: N = 5,000 samples from the noisy training set,
>   stratified by NOISY label (equal per-class counts; if not
>   divisible, distribute remainder by ascending class index).
> - Sampling rule: numpy.random.default_rng(20260813), one
>   permutation per class of that class's sample indices (indices
>   as ordered in the stored labels array), take the first
>   N/num_classes. Fully derivable from this paragraph + committed
>   arrays; no sealed-mask access.
> - Statistic: accuracy against NOISY labels on the split
>   (argmax(logits) == noisy_label), computed per checkpoint from
>   stored logits_train.npy. Selection = argmax over the 24-epoch
>   grid. Tie-break: earliest epoch.
> - Secondary (non-gating): mean CE loss against noisy labels on
>   the same split, selection = argmin, reported alongside.
> - No smoothing.
> - REQUIRED metadata field:
>     limitation: "split is drawn from data used in training
>     (no true held-out noisy split exists without retraining;
>     out of Phase-0 scope). Memorization inflates noisy accuracy
>     at late epochs; this bias is a property of the baseline
>     under audit and is reported, not corrected."

## (b) PINS — method B-i (effective rank)

> - Input: feats_train (all samples, no subsampling).
> - Statistic per checkpoint: mean-center features; compute the
>   512×512 covariance; effective rank = exp(Shannon entropy of
>   the eigenvalue distribution normalized to sum 1)
>   (Roy & Vetterli definition).
> - Selection (binding): argmax of effective rank over the
>   24-epoch grid. Tie-break: earliest epoch. No smoothing.
> - Direction rationale to record verbatim: "argmax is registered
>   on the hypothesis that representation quality peaks before
>   memorization-driven collapse; the late-epoch rank dynamic
>   under label noise is not settled in the literature — needs
>   verification. argmin is reported as a non-binding sensitivity."
> - Sensitivity (non-binding): argmin selection, reported
>   alongside; gates nothing.
> - Save the full 24-point effective-rank curve per run in the
>   output JSON.
> - B-iv (faithful TopoGeoScore reproduction) is TABLED, not
>   rejected; requires transcription from the original paper under
>   a separate instruction.

## Implementation notes (no discretion exercised)

Two points where the pins are exact and a reader might otherwise wonder:

- **Divisibility.** N = 5,000 divides evenly for both datasets — 500 per class for
  CIFAR-10, 50 per class for CIFAR-100 — so the remainder rule is implemented but never
  fires. It is implemented anyway so the code matches the pin as written.
- **Covariance normalization.** The pin does not say whether the covariance divides by
  `n` or `n-1`. It does not need to: effective rank normalizes the eigenvalues to sum 1,
  so any positive scaling of the covariance leaves the statistic unchanged. `n-1` is
  used. Eigenvalues of a PSD covariance equal its singular values, so this matches Roy &
  Vetterli's singular-value formulation exactly.

R6 was adopted on 2026-08-14 in place of a direct push of the internal repository. Six of
this repository's commits carry the email address of a third party who is not a party to
this project's publication decisions, and publishing that address is not ours to
authorize. The two obvious remedies both fail: rewriting history to change the author
fields would change every commit hash, and the internal record — preregistrations,
adjudications, verification reports — cites those hashes, so the rewrite would void its
own citations. Anchoring instead publishes a fresh snapshot of the tracked tree, with no
`.git` carried over, so no internal commit object and no author metadata crosses the
boundary. What the public repository attests is the state of the tree at a point in time;
`docs/ANCHOR_CHAIN.md` maps that state back to the internal commit chain by hash and
timestamp, without author fields. The immutability evidence is therefore the public
snapshot's own timestamp, and the internal hashes remain citable because they were never
rewritten.

- **R6-guard** (appended 2026-08-15, before the first anchor push). Before EVERY snapshot
  push, verify that the remote's HEAD equals the last snapshot HEAD we pushed. Any
  foreign commit → STOP and report, do not push. This exists because the first anchor
  target turned out to be an occupied name: a repository under the same account was
  already receiving another project's commits, and a mechanical push would have mixed two
  projects in a public record. The guard makes that class of collision detectable
  permanently rather than once. Until the first successful push, "our last pushed
  snapshot HEAD" is undefined and the gate is instead: the remote must be public and
  empty.
