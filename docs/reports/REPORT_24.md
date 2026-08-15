# REPORT 24 — R-2 done, fixtures pass, and a pin-request batch under safeguard (a)

**Project** G1-NoCleanOracle / Real-Noise Extension Tier 1
**Session** ext_realnoise server session (DATA-PRODUCING QUARANTINED WORKER)
**Issued** 2026-08-15 · re-emitted under R7 after the chat copy was lost
**Status** PINS-PENDING — implementation not started, no real-log execution, nothing unsealed

## 1. Completed

**R-2 — plan correction appended.** Internal commit `e485d45`. The specified text was
**appended** to `docs/remediation_plan_v2.md` rather than edited into the body, per R1.
The plan grew 5,745 → 6,162 bytes; new sha256 `ab8a73c9be34ba2c…`.

**R-1 fixture run — committed suite against current code.**

```
34 passed, 3 skipped in 1.07s
```

The three skips are all `could not import 'torch'` on the host
(`test_checkpointing.py:20`, `test_elr.py:17`, `test_gce_sop.py:15`); they run inside the
container. **Caveat worth a pin (item 23):** these nine files are the original unit tests
— AUROC, checkpointing, ELR, GCE/SOP, NCR, noise, sweep gating, tail. They are not
A1–A14 synthetic fixtures, and no such fixtures exist in this repository.

## 2. Why implementation has not started

R-3 authorised implementation on the premise that *"the anchored plan is mechanically
complete (grids, denominators, fail-closed rule, floors, δ, tie rules, aggregation
variants all pinned); implementation is transcription, not design; no free parameter
remains that exposure knowledge could tune."*

Reading A1 through A14 against that premise, it does not hold **for the plan alone**.

The root cause is one missing document. The plan's header names its governing theory as
`theory_spec_v2 (213fddf7…)`. That document is not in this repository and was never
transmitted to this session — a repository-wide search finds `theory_spec` exactly once,
in the plan's own header. The plan's core estimands are named but never defined:

| symbol | used in | defined in this repository |
|---|---|---|
| `ρ̂*_LE` | A2, A3 | no |
| `F_δ` | A2 | no |
| `w_δ` | A2, A3, A5 | no |
| `Ĵ_s,LE`, `η̂_s,LE` | A4 | no |
| `g_a(t)` | A5 | no |

The dependency chain makes this blocking rather than cosmetic: without `ρ̂*_LE` there is no
A2; without A2 there is no A3 taxonomy; A4's "adjudicated on compatible runs", A5's
`p_unif = w_δ`, and §3's branch language all hang off A3. So **A2 through A6 are blocked
as a group**, and §4 step (3) — "A1–A9 + A12 on G2-15" — cannot complete.

**Highest-value single request: `theory_spec_v2` inline, verbatim.** It plausibly closes
most of the batch below.

## 3. Pin-request batch

Each item gives the plan-text location it arises from. Items marked **[spec?]** look like
they belong to `theory_spec_v2`; **[absent]** items appear to exist nowhere. Where a
reading seems natural, it is stated as *our reading* so a one-word confirmation suffices.
Alternatives are listed where they are genuine forks. **No recommendation is offered on
any item whose resolution could be informed by the exposed EXP-001/002 magnitudes** —
those are marked *no recommendation offered, by policy*.

### A2 / A3 — the classification backbone

1. **[spec?]** §2 A2 "ρ̂*_LE, F_δ, w_δ at all δ" — definitions of all three.
2. **[absent]** §2 A3 lists four classes but gives a rule for only one. The indeterminate
   band is pinned (`ρ̂*_LE within ±0.025 of δ under any sensitivity`); the rule separating
   **incompatible** from **compatible** is not stated. Alternatives: (i) `ρ̂*_LE > δ` →
   incompatible; (ii) a separate feasibility predicate on `F_δ`. *No recommendation
   offered, by policy* — the choice interacts with the magnitudes in EXP-001.
3. **[absent]** §2 A3 — what makes a compatible run **solved** vs **unsolved**. Our
   reading: "some registered selector attains ≤ δ on every axis simultaneously". Confirm
   or replace.
4. **[absent]** §2 A2 "raw Δ vectors saved alongside" — Δ of what, over what index. Our
   reading: per-axis raw regret relative to the corrected oracle, as a 24-vector per run
   per axis.

### A4 / A5 — selector accounting and chance baselines

5. **[spec?]** §2 A4 "Ĵ_s,LE, η̂_s,LE within-layer" — definitions, and what "within-layer"
   quantifies over (layer = the `[CORRECTED-PRIMARY]` / `[FROZEN-HISTORICAL]` tag?).
6. **[absent]** §2 A4 "best-achievable gaps on incompatible" — definition of the gap.
7. **[spec?]** §2 A5 "expected uniform joint regret = mean_t max_a g_a(t)" — definition of
   `g_a(t)`. Our reading: normalized regret of axis *a* at epoch *t* against the corrected
   oracle, with `mean_t` over the 24 retained epochs. Confirm the index set.

### A6 — CF benchmarks

8. **[absent]** §2 A6 "CF selection benchmarks" — what CF expands to. Candidates:
   cross-fitted, counterfactual, closed-form. Not inferable from context.
9. **[absent]** §2 A6 "procedure risks" and "sign-free" — definitions.
10. **[absent]** §1 "K_CV = 5 stratified" — **stratified by what**. A single run is a
    24-point trajectory with no obvious stratum; if the cross-fitting is across runs, the
    stratification variable (learner? split? seed?) needs naming.

### A7 — set-valued oracles

11. **[absent]** §2 A7 "ε ∈ {0.005, 0.01, 0.02}" — units. Absolute raw risk, or normalized
    regret? Both are dimensionally plausible at that scale. *No recommendation offered, by
    policy.*
12. **[absent]** §2 A7 "verdict-flip table" — which verdict flips are tabulated: the frozen
    Phase-I PASS/WEAK/KILL, or the A3 taxonomy class.

### A9 — OOD decomposition

13. **[absent]** §2 A9 "per-(u,o) corrected oracles and feasibility" — definition of
    feasibility at the (score, pool) level.
14. **[absent]** §2 A9 "zero-IQR guard firing audit from unsealed EXP-002". EXP-002 records
    IQRs computed over the **120-point** trajectory, while §1's corrected denominator is
    the IQR over the **24 retained** epochs. These are different quantities and can differ
    in whether they are exactly zero. Which one adjudicates D-7? *No recommendation
    offered, by policy* — this is precisely a question about the exposed magnitudes.

### A10 — post-registered selectors

15. **[absent]** §2 A10 "TGS-N-torsion on all runs; TGS-N-full on G2-15 only" — **both
    variants are entirely undefined**: no statistic, no algorithm, no library, no selection
    direction. By contrast LW-N in the same paragraph is fully pinned (first local minimum
    of PC′, k = 3, plateau/tie → earliest, mapped to the nearest retained checkpoint, ties
    → earlier) and is implementable as written. The timing pilot and the >20 min/ckpt abort
    are also pinned; only the statistic itself is missing.
16. **[absent]** §2 A10 "3 deviations recorded verbatim" — which three. Three registered
    deviations are named for LW-N (train-mode augmented per-sample-timing statistic, grid
    mapping, k and warm-up convention); is this the same three, or three TGS-specific ones?

### A11 — clean-label budget

17. **[absent]** §2 A11 — definition of `q`. Our reading: the probability that a selection
    made on `D_sel` attains ≤ δ when measured on `D_assess`, over the seeded resampling.
    Confirm, and state the randomization `B = 1000` resamples.
18. **[absent]** §2 A11 — identity of `D_eval` (the clean test set?), and the composition of
    the **mixed** condition in `(mixed→OOD)`.

### A12 / A13 / A14

19. **[absent]** §2 A12 "LOSO verdict stability" — what the O is (seed / split / learner),
    and which verdict's stability is measured.
20. **[absent]** §2 A13 "equivalence-style reanalysis" — the equivalence margin.
21. **[absent]** §2 A14 "at both granularities and both metric scales" — naming of each.
    Our reading: granularity = 24-grid vs 120-grid; metric scale = normalized regret vs
    epoch gap. Confirm.

### Cross-cutting

22. **[absent]** §5 "spec G.2 forbidden phrases bind" — the phrase list itself. Report
    wording is bound by it and it is not available here.
23. **[confirm]** R-1 "run the committed suite" — whether this means the nine existing unit
    tests (run above), or synthetic fixtures the clean session will deliver separately.

## 4. What is implementable without any pin

**A1 is fully pinned.** §1 fixes the grid (24 retained checkpoints), the corrected oracle
(argmin of the raw logged risk, earliest tie), the denominator (IQR of the raw risk over
those 24 epochs), the fail-closed zero-IQR rule, the sensitivity floors, and δ with its
two sensitivity values. A1's deltas-vs-historical comparison is likewise computable, with
one small ambiguity: the historical side's regret was formed against the 120-grid smoothed
oracle with a 120-point IQR denominator, and we read "per-run deltas vs historical" as
comparing each side as originally defined rather than re-denominating one of them.
Parts of A8 and A9 follow once A1 exists.

Implementing A1 alone would consume an anchor cycle for a fragment of one milestone, so
this session's default is to implement the whole battery in one commit once the pins land.
Say the word if A1 should go first instead.

## 5. State at issue

- **Internal** — HEAD `e485d45`, 48 commits, no remote, clean tree. Two commits not yet
  anchored: `e48ca8f` (R6-guard baseline advance, push-history row) and `e485d45` (R-2).
- **Public** — HEAD `8b6f5783ebe0b6d425c6839295ebba3a0de349e3`, unchanged; R6-guard
  baseline matches.
- **Sealed** — 5 commitments under `results/sealed_ext/`, 5 salts intact, **no unseal
  operation has run**. `results/corrected/` does not exist.
- **Canonical integrity** — hygiene exit 0; 24 baselined masks unchanged; R5 holds across
  19 modules with zero drift; SEALED markers 2/2; G1 48 runs and B2 15 runs unchanged.

**HOLD** — awaiting the pin batch. On arrival: implement → fixtures → anchor the
implementation commit before any real-log execution (safeguard b) → §4 step (3) → R7
milestone. Step (6) adjudication is review-side and is not approached.
