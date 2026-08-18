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

## Anchor record

| field | value |
|---|---|
| anchor URL | https://github.com/october25kim/no-clean-oracle |
| first push (UTC) | 2026-08-15T05:07:57Z |
| public HEAD | `7926c582624e6d0670926c9629bfe302e3e2ae56` |
| public tag | `v1-pre-remediation` → object `27f5b8037e066ab58a39d7abe280e444499e861f` |
| internal lineage | `2bf5332` (this repository) |
| snapshot contents | 76 tracked files, single author `october25kim@users.noreply.github.com` |

The first anchor target was a different repository whose name was already in use by
another project under the same account. It was ceded rather than reclaimed: taking the
name back would have severed GitHub's rename redirect and captured that project's
hard-coded push URL into our public record. The R6-guard in
`docs/g2_pin_requests.md` is the standing check that grew out of that near miss.

### Push history

| # | public commit | pushed (UTC) | internal lineage | subject |
|---|---|---|---|---|
| 1 | `7926c582624e6d0670926c9629bfe302e3e2ae56` | 2026-08-15T05:07:57Z | `2bf5332` | post-Tier1-forward-pass, pre-remediation |
| 2 | `dfc9263befa372a8fa071db02426c297f162db3b` | 2026-08-15T05:26:39Z | `483065c` | Remediation Analysis Plan v2 committed |
| 3 | `8b6f5783ebe0b6d425c6839295ebba3a0de349e3` | 2026-08-15T05:34:13Z | `3364a81` | R7, defect ledger, REPORT_23 |
| 4 | `e4e9648932e726e796111815e42ba447a21d83ca` | 2026-08-15T05:56:52Z | `9ee15ec` | R7 clarification, REPORT_24 pin batch |
| 5 | `577c62b8f5cd93063bd5453a5dcc63bfcbbae2a5` | 2026-08-15T06:35:51Z | `52d37fe` | pin answers and plan restorations |
| 6 | `51b11618a3d4ae700b843599441b9011a533410b` | 2026-08-15T06:38:51Z | `60fccdd` | theory_spec_v2 received and saved |
| 7 | `cd829f3535f409ebd449eb0ee771fa03cd6042e7` | 2026-08-15T06:49:03Z | `85cff04` | corrected battery implementation, pre-execution |
| 8 | `e2e82fba16bf4d42f3f95a74ae55a72d6c5b7d50` | 2026-08-15T07:11:22Z | `c877f62` | MILESTONE-G2 corrected battery report |
| 9 | `773de6e6b674d946541e5302b4293f070632394a` | 2026-08-15T08:27:09Z | `153573c` | FINAL corrected battery on Tier1-36 |
| 10 | `06517af7951b3acddf35a1d994d80779c3a8b21b` | 2026-08-15T08:44:49Z | `69ea4ac` | step-6 adjudication |
| 11 | `988d7757c8216f4616f0aef8e605569ff9daf5b1` | 2026-08-15T09:54:18Z | `ab0b0fe` | manuscript figures and REPORT_31 |
| 12 | `c7d98e17d3eff40774c126c52f2c440714a63bfd` | 2026-08-15T10:45:51Z | `6cdaea8` | figures v2, data tables, REPORT_32 |
| 13 | `d02a4727911f94d550b55e2a45d85b41b7d9ffb8` | 2026-08-15T10:54:27Z | `2b8e0c2` | A9 addendum registered, pre-execution |
| 14 | `a7b9afe40e2917aabc0b1b51f518d3a1185a6690` | 2026-08-15T10:55:59Z | `5c8164f` | A9 addendum executed, REPORT_33 |
| 15 | `99c1b647b813cd63fea62d29109902cfdf1d1d4e` | 2026-08-15T11:07:43Z | `09a1fce` | frozen-record tables T1 and T2 |
| 16 | `4c56d2c9540957f64b9c2c0243073c79585fe155` | 2026-08-16T03:5x:00Z | `9365900` | restore the five files the old procedure dropped (D-8) |
| 17 | `174c1e8ff5b0b24fed09be17ee1634ca9d45f71c` | 2026-08-16T04:0x:00Z | `b9660c7` | D-8b, D-8 verification, A10b DROP and Tier 2 dates |
| 18 | `17668d9055d0bd21f32aa89ed53a6c7e631fbcf2` | 2026-08-17T07:2x:00Z | `8722234` | exploratory manifest and hygiene-guard extension |
| 19 | `5d7449f960929908cd61b5dca5cf88bbbab63ccf` | 2026-08-17T07:3x:00Z | `a183116` | untrack macOS artifacts; record push 18 |
| 20 | `f697e5bb75e41c0046240c86755b167e71899a64` | 2026-08-17T08:0x:00Z | `138d5d8` | T6-T8 descriptive tables |
| 21 | `ff43169b83a80a02af429a90903c4daa1d66663a` | 2026-08-17T09:39:03Z | `b388e35` | T-EMIT-3 registration, pre-computation |
| 22 | `f2f866b17f39a4fcf6191265737faeac85e58bb1` | 2026-08-17T10:0x:00Z | `0e54776` | T-EMIT-3 transcription half and self-recording anchor |
| 23 | `8c2228c1191eb34fb22c1b99db0c3c5f73107141` | 2026-08-17T09:42:56Z | `49d8fc6` | self-recording anchor fix |
| 24 | `0570293f7af85ebd91ec25fa148f0fb2694979da` | 2026-08-17T10:57:35Z | `ac05d1a` | F3 defect fix with assertions, F4/F5 deltas, T15, T16 |
| 25 | `af4e983ae0f75aaa200045ba5f765eb8f1f76902` | 2026-08-17T12:28:25Z | `0bfa916` | T11 estimand pin, pre-execution |
| 26 | `5c0da2bf4f95237288e4a96a4ec71a2da6c92228` | 2026-08-17T14:14:43Z | `1845f2a` | recon wave-1 exit fixes (D-9, D-10) and wave-2 launch |
| 27 | `ca06fbeb567d7e089eac155333dc2b34af0bc0ef` | 2026-08-17T14:15:50Z | `811ad38` | defect ledger D-9 through D-12 |
| 28 | `1f394295bd75c2aed1a068807ef52f0428314986` | 2026-08-17T14:34:01Z | `d3f8a69` | D-9 withdrawn |
| 29 | `d18cc76fb41834c02ba88de704130e4b1b9b2dd3` | 2026-08-17T15:53:49Z | `c364341` | recon forward pass registered before execution |
| 30 | `84623c1bffa9e130070c206b90de04219dbc87d7` | 2026-08-17T17:56:55Z | `c207edd` | recon two-axis frame and its tests |
| 31 | `0126d721a2e119f516082788bea4ef34aced7bab` | 2026-08-17T19:09:15Z | `a5ae63e` | recon A1-A5 battery registered before execution |
| 32 | `349e19c149820fa7e523114b133da9b97af407db` | 2026-08-17T20:42:04Z | `ef1b660` | placement capture and the D-9 instrumentation addendum |
| 33 | `110baa9d29e8a0a705a15a258642cbea543c71bc` | 2026-08-17T21:04:50Z | `5d6a947` | recon forward pass and A1-A5 executed |
| 34 | `1fc5a084fbb96a7319c2b065b4bdcf20ddc9514a` | 2026-08-17T21:18:49Z | `6814811` | REPORT 36 and D-13 |

### R6-guard baseline

Reference HEAD for the equality check that precedes every future snapshot push:

```
1fc5a084fbb96a7319c2b065b4bdcf20ddc9514a
```

Before the next push, the remote's HEAD must equal this value. Any other commit means a
foreign writer reached the anchor, and the push STOPS pending a report. After each
successful push the baseline is updated to the newly pushed HEAD, and the push history
above gains a row.

### Snapshots are chained, not replaced

The first snapshot was an orphan repository built from `git archive`. Pushing a second
orphan snapshot to the same branch is a non-fast-forward, and the only way to force it
through would discard the previously anchored commit — destroying exactly the public
timestamp the anchor exists to provide. So from snapshot 2 onward each snapshot is built
as a **child of the current public HEAD**: clone the public repository, replace the
worktree with the internal tracked tree, commit, push fast-forward. Every earlier anchor
stays reachable as an ancestor and stays citable, while the new state is a normal
fast-forward.

This changes nothing about what crosses the boundary. The public history is composed
solely of snapshot commits authored by `october25kim@users.noreply.github.com`; no
internal commit object, parent chain, or author field is ever carried over.


### Backfill and parity note (2026-08-16)

Rows 7-15 were backfilled from the public repository's own commit log, not from memory:
the anchor pushes happened but the table stopped being updated after row 6, and the
R6-guard baseline in this file was left at row 6's hash while the actual public HEAD had
moved nine pushes ahead. Every guard check in that window compared against the correct
value, because the value was carried in the session rather than read from here — which is
exactly the fragility R7 exists to remove. Timestamps are the public commit dates
converted to UTC.

The same review found that five internally tracked files had never reached the anchor at
all. See `docs/defect_ledger.md`, D-8. `scripts/anchor_push.py` now performs the push and
refuses on any file-set difference against the internal `git ls-files`.
