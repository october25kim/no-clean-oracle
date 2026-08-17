# T15 — Sealed-artifact scope and grade  [CORRECTED-DESCRIPTIVE]

What was sealed, by what mechanism, and where the record of it sits — so the
prospective grade of the Tier-1 (Extension) compatibility claim can be checked file by
file rather than taken on assertion.

## The mechanism was salted commitment, not encryption

This must be stated plainly because the task asks for ciphertext digests and a
key-release commit, and **neither exists**. Amendment 2's SEALED-COMMITMENT rule
described encrypting each outcome file under a fresh key and storing the key
separately. What was implemented, reported in REPORT #15 and accepted, is the salted
form: for each file F a fresh 32-byte salt is drawn, `commitment = sha256(salt ‖ F)` is
published, and the salt is withheld. The salt is the key-analog; it was never committed
and there is no key-release commit, because unsealing here meant revealing the salt in
the session that verified it, not publishing it.

Both forms answer the same question — could these numbers have been changed after the
fact — and the salt is what stops a low-entropy counts file from being brute-forced back
out of a bare digest. What the salted form does **not** provide is confidentiality
against the holder of the plaintext, which was never the threat model: the plaintext sat
on this machine throughout.

## Sealed files

| file | schema | commitment `sha256(salt ‖ F)` | salt | plaintext class |
|---|---|---|---|---|
| `ext_B_representation.json` | `ext_B_representation.v1` | `474c0ac37b64007927c8d4241efee89a8bdc4880f6ff0c2684d85e06ded0d11f` | 32 B, withheld | selector ER epochs, effective-rank curves, regrets |
| `ext_C1_noisyval.json` | `ext_C1_noisyval.v1` | `b5f76bc8f6c3c42a2b8ff0ea36124ef3ac705a34f1f271a068e866814f268136` | 32 B, withheld | selector NA epochs, accuracy/CE curves, regrets |
| `ext_E_tgrid.json` | `ext_E_tgrid.v1` | `352744bea1b024ace54de01c434460c7be4e410fe08e916f7e644efee5edec4d` | 32 B, withheld | selector E epochs, τ-grid curves, regrets |
| `ext_counts.json` | `ext_counts.v1` | `83d4b10c6aef79aaaabf8230dae38e068084fc1e495547d91355905b6a7517c9` | 32 B, withheld | per-selector per-axis success counts at three δ |
| `ext_oracle_epochs.json` | `ext_oracle_epochs.v1` | `09462aa922a15cb417eb0bd945a920cfd88b3c6d0ad4cb95827fdb039ba40819` | 32 B, withheld | oracle epochs, per-axis, per run |

## What was NOT sealed

| class | sealed? | why |
|---|---|---|
| risk curves (per-epoch R_ID, R_tail, R_OOD) | no | logged during training, before any selector existed; they are the inputs the seals were computed from |
| forward-pass outputs (logits, embeddings) | no | attributed by `MANIFEST.sha256` over 6,089 files, digest `864d4a88…`; sealing them would protect nothing the manifest does not |
| Phase-I / Phase-II results | no | frozen and published long before the sealing regime existed |
| corrected battery outputs | no | produced after unsealing, under the anchored plan |

## Anchoring and release

| event | anchor / commit |
|---|---|
| seals created (Tasks 2–4 executed) | internal `48c994f`, anchored in public push #1 `7926c582…` |
| plan committed and anchored (the precondition for unsealing) | internal `483065c`, public push #2 `dfc9263b…` |
| commitments verified 5/5 and files opened | REPORT_28, anchored in public push #9 `773de6e6…` |
| key (salt) release commit | **none — salts are git-ignored and were never published** |

The prospective grade rests on the ordering, which is checkable from the public history:
the seals were anchored before the plan that governs their use was anchored, and both
preceded the verification transcript. Spec E.1 makes the seal status itself the thing
that decides whether Tier-1 counts as prospective confirmation or as an extended audit.

