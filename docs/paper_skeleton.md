# Paper Skeleton — No Clean Oracle
Working title: *No Clean Oracle: An Audit of Deployment-Robust
Checkpoint Selection for Noisy-Label Learning without Clean
Validation*
Framing: S1-style audit (per classification_record.md, 07ff003).
Target venues: TMLR / NeurIPS Datasets & Benchmarks / analysis
tracks (ICLR/ICML). Status tags: [NOW] writable from existing
certified results · [T1] awaits Tier 1 · [T2] awaits Tier 2.

## Claims ledger
C1 — Per-axis optimal checkpoints diverge under label noise:
conflict in 7/16 cells, concentrated ID↔OOD (5/16), 2 cells
CI-separated; persists under ELR (4/8). [NOW]
C2 — No clean-validation-free selector achieves deployment-robust
selection: OOD failure 45/45 at δ ≤ 0.20; preregistered
predictions confirmed (P1 15/15; P2 aggregate). [NOW]
C3 — C1–C2 generalize to real human noise across four learner
mechanisms (CIFAR-N × {CE, ELR, SOP, GCE}; branch table R1/R2/R3).
[T1]
C4 — Failure persists at scale under pretrained init (Clothing1M
existence check; own table, never pooled). [T2]
M — Audit methodology (preregistration + mechanical branch
tables + bit-exact independent verification + self-attesting
provenance) makes the negative result attack-resistant. [NOW]
Overclaiming guards: conflict not claimed universal (9/16 null);
selector failure claimed for the registered set only; Tier 2
never a statistical pillar.

## §1 Introduction [NOW]
Deployment gap → single-axis illusion → two audit questions →
answers unsoftened (7/16; 0/45; predictions confirmed) →
contributions C1, C2, [C3], M; negative-result-as-contribution
paragraph.

## §2 Related Work [NOW]
2.1 mechanism taxonomy (pre-announces learner grid). 2.2 selection
without clean validation; boundary = evaluation axes, not
criterion. 2.3 K-criteria (K1/K2/K2'/K3); PES as lineage. 2.4
Label Wave applicability ruling. 2.5 preregistration positioning.

## §3 Problem Setup and Audit Design [NOW]
3.1 formalism (trajectory, axes, oracles, CR, NCR, selector
regret). 3.2 no-clean-oracle constraint. 3.3 audit architecture
(preregistration, verification, provenance; ~1 page; M lives
here). 3.4 experimental frame.

## §4 Phase I — Does the Conflict Exist? (G1) [NOW]
4.1 design + preregistered PASS/WEAK/KILL rule verbatim. 4.2
verdict first sentence: WEAK, 7/16; ID↔OOD primary (re-scope
amendment); 4.3 ELR survival 4/8 (strongest finding). 4.4 honest
negative space (9/16 null → motivates §6). 4.5 bit-exact
independent verification.

## §5 Phase II — Can Anything Select? (G2) [NOW]
5.1 certified re-runs (bit-identical 15/15, negative control);
selectors E/C1/(b); full-D exploratory fenced. 5.2 E
preregistration adjudication (P1 15/15; P2 aggregate / 9/15
regret / 12/15 epoch-gap; 13→15 correction on record). 5.3
headline: 0/45 OOD at δ ≤ 0.20; min OOD regret +0.289; ELR ×
low-noise compact rule licenses nothing. 5.4 mechanical
classification (S2 rejected, disagreement 6/15, S3 → audit;
verified 14/14). 5.5 failure structure (T-sensitivity 11/15;
C1 saturation 0.999+; argmax/argmin divergence). 5.6 full-D
fenced (median stops 34/59/44).

## §6 Phase III — Real Noise, Real Learners [T1][T2]
6.1 design [NOW]: K-criteria grid restoration; CIFAR-N 36 runs;
unified protocol + schedule-deviation defense; SOP pins; branch
table verbatim. 6.2 results slot [T1]. 6.3 Tier 2 slot [T2].

## §7 Discussion [mostly NOW; finalize after T1]
7.1 what the audit licenses; practical pointer only. 7.2 why OOD
fails unconditionally (descriptive, hypothesis-generating). 7.3
limitations exhaustive (WEAK scope; three selector families;
synthetic noise pre-T1; one backbone; unified schedule; TM family
out of scope; Tier 2 non-comparability). 7.4 what the methodology
caught (13→15; stamp fail-open; five blocking pre-launch bugs).

## §8 Conclusion [after T1] — three-sentence discipline.

## Appendices
A protocol + BCa spec · B preregistrations verbatim + hashes ·
C verification records · D provenance regime + R1–R3 ·
E full tables · F Label Wave applicability criterion ·
G CIFAR-C-local / C1M-C-local conventions.

## Open decisions
1. Subtitle "An Audit of…" vs declarative — at venue targeting.
2. M in contribution list — TMLR/D&B yes; analysis tracks fold
into §3.3. 3. §7.1 depth — pointer until T1 lands.
