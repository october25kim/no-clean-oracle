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
- **exercised by** — no corrected analysis has run. Under the plan's reporting rules, the
  first analysis that exercises this defect must update this entry in the same commit.
