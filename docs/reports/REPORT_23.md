# REPORT 23 — Plan v2 anchored; §4 execution has NOT started

**Project** G1-NoCleanOracle / Real-Noise Extension Tier 1
**Session** ext_realnoise server session (DATA-PRODUCING QUARANTINED WORKER)
**Issued** 2026-08-15, after the Plan v2 anchor push
**Status of this file** re-emitted under R7 because the chat copy was lost in transit

## Correction of a standing presumption

The review side recorded the lost report as *"presumed to be the G2-15 milestone or
later."* That presumption is wrong and correcting it is the most important line in this
file: **no step of §4 has executed.** There are no A1–A9 tables, no G2-15 milestone, and
no corrected outputs of any kind. Nothing has been unsealed.

Verified state at the time of writing:

| check | value |
|---|---|
| `results/corrected/` | absent |
| `results/sealed/` | absent |
| `results/sealed_ext/` | 5 sealed files, 5 salts, all intact |
| unseal operations run | none |
| A1–A14 implementation in `scripts/` | 0 files |
| synthetic-fixture tests in `tests/` | 0 files |

## What did complete

**Plan v2 saved verbatim and committed.**

| field | value |
|---|---|
| path | `docs/remediation_plan_v2.md` |
| bytes | 5,745 |
| sha256 (canonical) | `2a5dae53e31548939b651166bedbe9eb5b4931ad2c0879b7ab820190e91f5405` |
| lines | 105 |
| internal commit | `483065cd66dbc837d20fa9a226ee58b1b4d22ac8` |

**R6-guard passed** — remote HEAD equalled the recorded baseline `7926c58…`; no foreign
writer had reached the anchor.

**Anchor push completed.**

| field | value |
|---|---|
| public HEAD | `dfc9263befa372a8fa071db02426c297f162db3b` |
| pushed (UTC) | 2026-08-15T05:26:39Z |
| parent | `7926c582624e6d0670926c9629bfe302e3e2ae56` — preserved as an ancestor |
| tag `v1-pre-remediation` | still `7926c58…`, unchanged |
| files changed | `docs/remediation_plan_v2.md`, `docs/ANCHOR_CHAIN.md` |
| required-zero checks | third-party email 0 · payloads 0 · sealed plaintext/salt 0 · gate1/kt0 0 · single author |

Public readability confirmed without credentials: the plan returns HTTP 200 at 5,745
bytes and its sha256 matches the local file exactly.

## Procedure correction: snapshots are chained, not replaced

The second snapshot push was rejected as a non-fast-forward. This is structural: each
snapshot was being built as an orphan repository from `git archive`, so it shares no
history with what is already published. Forcing it through would have discarded the
previously anchored commit — destroying exactly the public timestamp the anchor exists to
provide.

From snapshot 2 onward each snapshot is therefore built as a **child of the current public
HEAD**: clone the public repository, replace the worktree with the internal tracked tree,
commit, fast-forward push. Every earlier anchor remains reachable and citable.

Nothing about the boundary changed. The public history is composed solely of snapshot
commits under the noreply identity; no internal commit object, parent chain, or author
field is carried over. Recorded in `docs/ANCHOR_CHAIN.md` with the push-history table and
the updated R6-guard baseline.

## Why §4 has not started — three blockers

**1. The instruction was truncated.** The message ended mid-sentence at *"…fixture tests
already committed"*, so the EXECUTION paragraph is incomplete.

**2. §4 step (2) has no target.** The plan directs "unseal `results/sealed/*` (EXP-linked
FACT quantities)". That directory does not exist. The R6 sealing procedure was applied
only to the Tasks 2–4 outputs, which live in `results/sealed_ext/` (5 files). The
EXP-001/002 source quantities were never sealed — they were computed as a read-only
lookup, exposed, and ledgered. So it is not determined what step (2) is meant to open.

**3. The corrected-analysis implementation is absent, and writing it here would breach the
quarantine.** No A1–A14 code and no fixture tests exist in this repository. The truncated
sentence asserts fixture tests are "already committed", but they are not committed *here*.
The standing boundary from Amendment 2 remains in force: corrected-analysis code and
synthetic fixtures are to be written in a fresh session with no real-results mount, and
this session must not implement them. Writing A1–A14 here would mean the session that saw
EXP-001/002/003 designs the corrected analysis — precisely the contamination the quarantine
exists to prevent.

## What is needed to proceed

1. The truncated EXECUTION paragraph, re-sent.
2. A designation for §4 step (2): whether it means the five files in
   `results/sealed_ext/`, or a separate `results/sealed/` set that the clean session will
   produce and deliver.
3. A delivery path for the A1–A14 implementation — either committed into this repository
   by the clean session, or a path this session is told to execute. On arrival: verify it
   is committed before running (R2), produce outputs under `results/corrected/` with
   `code_stamp`, and report.

## Standing state

- **Public** — HEAD `dfc9263`, tag `v1-pre-remediation` → `7926c58`, 77 files. Plan v2,
  `FACT_REDACTED.md` and `ANCHOR_CHAIN.md` all readable unauthenticated.
- **Internal** — 45 commits, no remote (never pushed, by design), clean tree.
- **Sealed** — 5 commitments intact under `results/sealed_ext/`; Task 1 manifest digest
  `864d4a887dc42c227f0e3ea02f1f88117cc618ec552a626d589281d265b727ab` over 6,089 files.
- **Canonical integrity** — data hygiene exit 0; 24 baselined masks unchanged; R5 campaign
  invariant holds across 19 modules with zero drift; both SEALED markers present; G1 48
  runs and B2 15 runs unchanged.

**STOP** — §4 steps (2)–(5) await the three items above. Step (6) adjudication is
review-side and is not approached under any circumstance.
