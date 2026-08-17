# Defect and exposure ledger

Two kinds of entry. **EXP-*** records an exposure: a quantity that should have been
sealed but appeared in a channel a reader could see. **D-*** records a defect in code or
procedure. Neither kind is ever erased or rewritten; a superseded entry gains a note and
keeps its text.

## Reconciliation note (2026-08-15)

This file did not exist until now. The exposure entries below were filed on 2026-08-14 as
chat text and were never written to the tree, which is why the public tree returned 404
for `docs/defect_ledger.md` — the path was correct and the file was simply absent. Under
R7 the ledger now travels with the anchor like every other report.

`D-7` is named in `docs/remediation_plan_v2.md` §0 and is recorded below. Entries `D-1`
through `D-6` were never transmitted to this session; if they exist on the review side
they should be merged in, and their absence here is a gap in this file rather than a claim
that no such defects were found.

---

## Exposure entries

Filed 2026-08-14, session `ext_realnoise`. No re-sealing is claimed and none is possible:
each quantity below was visible in the channel named, and this record exists so that any
analysis informed by it is known to have been outcome-informed. Categories are described
without reproducing the values, so reading the ledger does not repeat the exposure.

### [DEFECT-IDENTIFIED] EXP-001
- **timestamp** — 2026-08-13 13:19 KST (REPORT #4); re-exposed 2026-08-14 ~22:00 KST
  (REPORT #11, verbatim re-attachment ordered by Task F)
- **session** — ext_realnoise server session
- **category** — G2/B2-frame oracle epochs and normalized regrets, 15 runs × 3 axes =
  45 cells. Fields: `t*_full`, `t*_24`, Δepoch, `R[t*_full]`, `R[t*_24]`, ΔR, ΔR/IQR;
  per-axis medians and maxima; aggregate counts.
- **channel** — terminal stdout of the lookup script → chat report, twice →
  agent-visible context
- **affected analysis decisions** — the FACT 2 registration (grid-restricted oracle
  adopted as the selector regret reference, 120-point oracle retained as ceiling, gap
  column unclamped) was decided with these values on record. Committed in `015b355`.

### [DEFECT-IDENTIFIED] EXP-002
- **timestamp** — 2026-08-13 13:19 KST (REPORT #4); re-exposed 2026-08-14 ~22:00 KST
  (REPORT #11)
- **session** — ext_realnoise server session
- **category** — G2/B2-frame per-pool per-score trajectory IQRs, 7 series × 15 runs =
  105 values. Fields: per-series IQR minimum/median/maximum, per-run primary-endpoint
  IQR, overall minimum.
- **channel** — terminal stdout → chat report, twice → agent-visible context
- **affected analysis decisions** — the FACT 4 registration (near-zero-IQR guard retained
  and characterised as empirically inert). Committed in `015b355`.

### [DEFECT-IDENTIFIED] EXP-003
- **timestamp** — 2026-08-13 14:43 KST
- **session** — ext_realnoise server session
- **category** — summary figures from EXP-001 and EXP-002 written into a tracked
  document (`docs/framing_preregistration.md`, "Measurement-validity registration").
- **channel** — git history, commit `015b355`. Removing them would require history
  surgery, which is forbidden because the internal record cites commit hashes.
- **affected analysis decisions** — as EXP-001/002. Additionally became the reason the
  public-anchor decision had to consider whether outcome values would be published.

### [DEFECT-IDENTIFIED] EXP-004
- **timestamp** — 2026-08-13 ~12:20 KST (REPORT #1)
- **session** — ext_realnoise server session
- **category** — Tier 1 gate run training-trajectory values: R_ID at start, at end, its
  minimum, and the epoch of that minimum.
- **channel** — terminal → chat report
- **affected analysis decisions** — none. The gate verdict rested on integrity and
  structural checks, not on these values. Recorded for completeness.

### [DEFECT-IDENTIFIED] EXP-005
- **timestamp** — 2026-08-14 ~22:00 KST (REPORT #11)
- **session** — ext_realnoise server session
- **category** — Tier 1 forward-pass recomputed metric values for one checkpoint
  (`c100n_sop_seed0` ep004: recomputed and logged R_ID, R_tail_dynamic, R_OOD).
- **channel** — container log → terminal → chat report
- **affected analysis decisions** — none. The smoke verdict rested on the deviation being
  exactly zero, not on the values themselves. Recorded for completeness.

### [DEFECT-IDENTIFIED] EXP-006
- **timestamp** — 2026-08-13 ~13:19 KST (REPORT #4)
- **session** — ext_realnoise server session
- **category** — exploratory selector median table (full-D, coarse-D, Label Wave) quoted
  from an already-committed document.
- **channel** — chat report; the source was `docs/framing_preregistration.md`, committed
  before the sealing regime existed.
- **affected analysis decisions** — restatement of the δ rationale. Recorded for
  completeness.

**Not recorded as exposures.** Integrity deviation magnitudes (exact zero, and values at
the 1e-16 scale) are classified as schema/integrity status, which the Tier 1 reporting
rules explicitly permit. If the review side disagrees, entries should be added rather than
the classification argued.

---

## Defect entries

### D-7 — `selection.py:31` fails open on a zero denominator

- **source** — `docs/remediation_plan_v2.md` §0, which names this defect and fixes the
  corrected rule.
- **location** — `src/analysis/selection.py:31`:
  `normalized_regret=(regret / scale if scale > 0 else 0.0)`
- **defect** — an exact-zero IQR maps to a normalized regret of **0.0**, i.e. a run whose
  risk never moved on an axis is scored as *perfect* on that axis. The failure direction
  is toward the selector looking good, which is the wrong way for a guard to fail.
- **corrected rule** — fail-closed: an exact-zero IQR excludes the run from that axis with
  a flag, rather than scoring it. Sensitivity floors ε ∈ {1e-4, 1e-3, 1e-2} absolute are
  reported alongside.
- **status** — the corrected rule is registered in the plan. **Whether the branch ever
  fired on real data is not yet adjudicated**: it is decided in analysis A9 from the
  unsealed EXP-002 data, and §4 has not reached that step.
- **exercised by** — the corrected battery on the G2-15 frame (2026-08-15,
  `results/corrected/battery_g2.json`). Updated in that analysis's own commit, per the
  plan's §5 rule.
- **audit result, both sides, G2-15 frame** — the two grids are kept apart per pin 14.
  *Historical side*: across 45 axis-instances the 120-point IQR was never exactly zero, so
  `selection.py:31` **fired 0 times**; the smallest 120-point IQR observed was 0.02233.
  *Corrected side*: across the same 45 axis-instances the 24-epoch IQR was never exactly
  zero either, so the fail-closed rule excluded **0** axes. The defect is real — a zero
  scale would have been scored as a perfect 0.0 — and on this frame it had no opportunity
  to fire. The Tier-1 frame is audited separately and this entry gains its result there.

- **audit result, Tier1-36 frame** — the frozen pipeline was never run on Tier 1, so the
  historical side does not apply there. On the corrected side the 24-epoch IQR was never
  exactly zero across 108 axis-instances, so the fail-closed rule excluded **0** axes.
  Across both frames D-7 has therefore had no opportunity to fire; it remains a real
  defect, and the corrected rule remains fail-closed.

### D-8 — five internally tracked files never reached the public anchor

- **found** — 2026-08-16, on resuming after a reboot, by comparing the public tracked tree
  against the internal one digest-for-digest. Anchors 1–15 are all affected.
- **defect** — the snapshot procedure extracted `git archive HEAD` into a fresh repository
  and staged it with `git add -A`. A fresh repository re-applies `.gitignore` to paths that
  are untracked *there*, and two project rules matched files the internal repository
  tracks: `results/` matched `results/cifar_n_masks/MANIFEST.json`, and `data/` matched
  `src/data/__init__.py`, `datasets.py`, `noise.py` and `ood_pools.py` — a bare `data/`
  pattern matches a directory of that name at any depth, not only at the root. Internally
  those files are exempt because tracking predates the rule; in the snapshot they were
  simply dropped.
- **impact** — the public anchor was missing four source modules and the CIFAR-N manifest,
  the file the README points at for dataset provenance. Every file that *did* cross was
  byte-identical to the internal copy, so nothing published was wrong; what was published
  was incomplete, and an anchor that does not reproduce the code that ran fails at the one
  thing it exists to do. No analysis result is affected: all analyses ran against the
  internal tree.
- **corrected rule** — `scripts/anchor_push.py` performs the push. It stages with
  `git add -A -f` so an ignore rule cannot drop a tracked file, and then enforces
  **file-set parity** against the internal `git ls-files`, refusing to push on any
  difference in either direction. The force-add fixes this cause; the parity check is what
  catches the next one.
- **verification** — the corrected push (public `4c56d2c9540957f64b9c2c0243073c79585fe155`,
  2026-08-16) was checked from a fresh clone: file-set difference 0, content differences
  none across all 120 tracked paths, and each of the five restored files returns HTTP 200
  at its raw URL. Prior anchors `7926c58` and `99c1b64` remain ancestors and the tag
  `v1-pre-remediation` still resolves to `7926c582624e`.
- **status** — fixed and re-anchored on 2026-08-16. Anchors 1–15 remain in the public
  history as they were pushed; they are not rewritten, and this entry is the record of what
  they omitted.

### D-8b — the R6-guard baseline in `ANCHOR_CHAIN.md` went stale for nine pushes

Filed alongside D-8; found in the same 2026-08-16 review.

- **defect** — the push-history table and the R6-guard baseline stopped being updated after
  anchor 6. The file continued to name `51b11618…` as the reference HEAD while the public
  anchor advanced through nine further pushes to `99c1b647…`.
- **why it did not fire** — every guard check in that window did compare against the
  correct value, because the value was carried in the working session rather than read from
  the file. That is precisely the fragility R7 exists to remove: a check that passes for a
  reason the record does not contain is not a check the record can be audited against, and
  it would have failed the moment a session resumed cold — which is how it was found.
- **fix** — rows 7–15 were backfilled from the public repository's own commit log rather
  than from memory, and the baseline was corrected. `scripts/anchor_push.py` now *parses*
  the baseline out of `ANCHOR_CHAIN.md` instead of accepting it as an argument, so a stale
  file makes the guard fail loudly rather than letting a session substitute a remembered
  value.
- **verification** — the corrected push ran through `anchor_push.py`, which read
  `99c1b647…` from the file, matched it against the live remote HEAD, and only then
  proceeded. The baseline now reads `4c56d2c9…` and row 16 is recorded.
