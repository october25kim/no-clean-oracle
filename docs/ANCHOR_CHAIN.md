# Anchor chain — internal commits attested by public snapshots

Public snapshots attest to internal states by hash; the internal commit chain is
retained privately to avoid publishing third-party author metadata.

Each row is one internal commit: short hash, committer-date in ISO-8601, and subject
line. Author fields are deliberately absent — they are the reason the internal
repository is not published, and reproducing them here would defeat the purpose.
Nothing else about a commit is elided: the hash identifies it exactly, and any
holder of the internal repository can verify a snapshot against the row.

Internal commits listed: 39 (chronological, oldest first).

| # | commit | committed (ISO-8601) | subject |
|---|---|---|---|
| 1 | `6a26332` | 2026-08-13T06:17:40+09:00 | Baseline snapshot post-B2 certification, pre-forward-pass |
| 2 | `9795c64` | 2026-08-13T06:43:47+09:00 | Forward-pass schema, E preregistration amendment, full-D scope line |
| 3 | `2f3d9db` | 2026-08-13T06:53:50+09:00 | Stamp the executing-code fingerprint into forward-pass artifacts (Task E-4) |
| 4 | `1600646` | 2026-08-13T06:55:28+09:00 | Reviewer bundle: ship the B2 certification chain, forward-pass docs and manifest |
| 5 | `e85a4e9` | 2026-08-13T07:29:27+09:00 | Fix shard-overwrite race in the forward-pass summary (Task F2) |
| 6 | `64813b4` | 2026-08-13T07:50:13+09:00 | Record the framing preregistration as received (PENDING, not binding) |
| 7 | `9d80c6b` | 2026-08-13T07:54:21+09:00 | Record finalized thresholds; verify the r citation and the S1 null |
| 8 | `3366144` | 2026-08-13T07:57:56+09:00 | Framing preregistration — pre-baseline, pre-E |
| 9 | `098da43` | 2026-08-13T08:03:09+09:00 | Baseline E (T-grid) executed; C1 and (b) stopped for pins |
| 10 | `0040b91` | 2026-08-13T08:15:39+09:00 | C1 and (b) pins — pre-computation |
| 11 | `f0a2426` | 2026-08-13T08:23:36+09:00 | C1 and (b) computation scripts — committed before execution (R2) |
| 12 | `8d179b8` | 2026-08-13T08:36:17+09:00 | Classification verification script — committed before execution (R2) |
| 13 | `79e6470` | 2026-08-13T08:37:11+09:00 | Scope the verification checks to the binding delta and the stated metric |
| 14 | `0fcd36d` | 2026-08-13T08:41:20+09:00 | Extend verification to every number in the classification record (R2) |
| 15 | `fd4c908` | 2026-08-13T08:42:26+09:00 | Print the check list after all checks are registered |
| 16 | `d86c84a` | 2026-08-13T08:49:58+09:00 | Assert both P2 metric readings, not just the primary one |
| 17 | `07ff003` | 2026-08-13T08:51:24+09:00 | Classification adjudication — S3/audit ruling, E adjudication, statistic pin (verified) |
| 18 | `47ec5f6` | 2026-08-13T08:53:22+09:00 | Bundle: ship the classification chain |
| 19 | `ed7d805` | 2026-08-13T08:58:26+09:00 | Codify protocol rule R3 — pre-commit verification of a registered document |
| 20 | `0ec29ea` | 2026-08-13T09:21:38+09:00 | Extension preregistration — real-noise × learner grid, pre-launch |
| 21 | `8fd28a8` | 2026-08-13T09:25:27+09:00 | Seal CIFAR-N labels and masks (Task 1) |
| 22 | `0c9a38d` | 2026-08-13T09:26:09+09:00 | Ignore data/ (downloaded CIFAR-N labels; provenance is in the sealed manifest) |
| 23 | `9b19e11` | 2026-08-13T09:33:14+09:00 | GCE and SOP implementations with unit checks (Task 2, committed before any run) |
| 24 | `4250c0e` | 2026-08-13T09:39:23+09:00 | SOP alpha pin for CIFAR-N — Option 1, repo symmetric row |
| 25 | `1c27c0b` | 2026-08-13T09:41:46+09:00 | CIFAR-N pipeline: sealed-split loading, exact-match names, code_stamp in run metadata |
| 26 | `230cc67` | 2026-08-13T09:44:05+09:00 | code_stamp: distinguish unavailable git from a clean tree; inject host git state |
| 27 | `2a39e09` | 2026-08-13T09:57:55+09:00 | Tier 2 amendment — axis definitions and launch gate, pre-probe |
| 28 | `317624b` | 2026-08-13T10:05:47+09:00 | Tier 2 probe execution note |
| 29 | `2f5b122` | 2026-08-13T10:14:46+09:00 | Fix five blocking defects found by pre-launch review, before any of the 35 runs |
| 30 | `135e15d` | 2026-08-13T10:26:48+09:00 | Paper skeleton — audit framing, claims ledger |
| 31 | `552151f` | 2026-08-13T10:27:59+09:00 | Data hygiene guard: launcher refuses on unaccounted files in data/ (Task H) |
| 32 | `f1713d7` | 2026-08-13T10:32:40+09:00 | Campaign attribution note for the Tier 1 sweep |
| 33 | `f272827` | 2026-08-13T12:15:41+09:00 | Correct the launcher's modules-digest claim; add the check that is actually right |
| 34 | `908f3c1` | 2026-08-13T12:43:39+09:00 | Register R4 (integrity-check tolerances) and R5 (campaign attribution invariant) |
| 35 | `015b355` | 2026-08-13T14:43:50+09:00 | Correct the Label Wave attribution; register the measurement-validity decisions |
| 36 | `ecbf9c2` | 2026-08-14T21:58:19+09:00 | Tier 1 forward pass: script, sharded launcher, and the schema's Tier 1 revision |
| 37 | `3afae66` | 2026-08-15T13:10:14+09:00 | Add LICENSE (MIT) and a README stating what this repository is not |
| 38 | `48c994f` | 2026-08-15T13:12:43+09:00 | Tier 1 Tasks 2-4 under the sealed-commitment regime |
| 39 | `9ea98d8` | 2026-08-15T13:24:32+09:00 | Register R6: public anchoring by snapshot, not by pushing the internal repo |

## Tags

| tag | tag object | target commit |
|---|---|---|
| `v1-pre-remediation` | `141a582c887d` | `3afae661f46c` |

## What a snapshot attests, and what it does not

A public snapshot is a single commit holding the tracked tree exactly as it stood at one
internal commit. It carries no `.git` from this repository, so it contains no internal
commit object, no parent chain, and no author metadata. Its public timestamp is the
immutability evidence a paper cites; this file is what maps that evidence back to the
internal commit it was taken from.

The snapshot does not attest anything about untracked state. Results, checkpoints,
logits, embeddings and sealed analysis outputs live under `results/`, which is
git-ignored, and are attributed by their own digests instead — per-run `code_stamp`,
sealed-input manifests, per-tree `MANIFEST.sha256`, and salted commitments for
outcome-bearing outputs.

