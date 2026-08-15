# Pin answers to REPORT_24 — verbatim transcript

Issued review-side 2026-08-15, answering the 23-item batch in `docs/reports/REPORT_24.md`.
Recorded verbatim; nothing paraphrased. The answers were issued **outcome-blind** — the
review side never received the EXP-001/002 magnitudes, all of which were lost in transport
— including the three items this session had withheld a recommendation on by policy.

## Status of the governing theory document

`theory_spec_v2` was **not** delivered with these answers. Section §II of the issuing
message contained a placeholder instructing that the file be pasted, not the file itself,
so `docs/theory_spec_v2.md` does not exist and nothing was saved under that name. This is
recorded here rather than worked around.

The answers below nevertheless define every quantity the battery computes, inline. Where
they cite spec sections (A.4, B, C.2–C.5, D.1, D.5, E.2, G.2, H.4) the citation is a
pointer, not the operative content — the operative content is in the answer text. The
residual spec-dependent items are naming and reporting-layer questions, not computations:
notably the expansion of the subscript **LE** and the full population / LE / CF layer
taxonomy of spec A.4. Implementation proceeds on the answers; if the spec later contradicts
them, the spec governs and the affected outputs are recomputed.

---

## §I — PIN ANSWERS (1–23), verbatim

1. Defined in spec C.3/C.4 (below). ρ̂*_LE = min_t max_a ĝ_a(t);
   F_δ = {t: max_a ĝ_a(t) ≤ δ}; w_δ = |F_δ|/24.
2. PINNED: incompatible ⟺ ρ̂*_LE > δ (equivalently F_δ = ∅);
   the ±0.025 band overrides to indeterminate.
3. CONFIRMED, with the selector set pinned: compatible-solved ⟺
   some selector in {E(τ=1), NA, ER-argmax} attains
   Ĵ_s,LE ≤ δ. Post-registered selectors (LW-N, later TGS-N)
   are reported but NEVER gate the taxonomy.
4. CONFIRMED: Δ_a(t) = R̂_a(t) − R̂_a(t*_corrected), a 24-vector
   per run per axis, raw units.
5. Spec D.1: Ĵ_s,LE = max_a ĝ_a(t̂_s); η̂_s,LE = Ĵ_s,LE − ρ̂*_LE.
   "Layer" = estimand layer (population / LE / CF; spec A.4),
   NOT the report tag. Both terms of η̂ must come from the LE
   layer.
6. PINNED: on incompatible runs report BOTH (i) the trajectory
   deficit ρ̂*_LE − δ and (ii) η̂_s,LE per selector (still
   defined).
7. CONFIRMED: g_a(t) per spec C.2 with the §1 corrected oracle
   and denominator; mean_t over the 24 retained epochs.
8. CF = cross-fitted (spec A.4 table). Per-axis: partition that
   axis's evaluation sample into K_CV = 5 folds; select
   argmin_t R̂_a^(−q); evaluate on fold q; average over q.
   Minimax: argmin_t max_a ĝ_a^(−q) evaluated on fold q.
9. Spec A.4/B: the CF quantity is the risk of a data-driven
   selection PROCEDURE; "sign-free" = selector-minus-CF values
   may be negative and are reported as benchmark excesses with
   no nonnegativity claim.
10. PINNED: folds partition evaluation SAMPLES, fixed once per
    dataset (not per run), seed 20260814: labeled sets (ID/WC)
    class-stratified; OOD pools simple seeded random within
    pool. The same fold partition serves every run of that
    dataset.
11. PINNED: ε in ABSOLUTE raw-risk units (spec C.5 uses raw
    R_a).
12. PINNED: the A3 taxonomy class per run is the tabulated flip;
    frozen Phase-I PASS/WEAK/KILL is never recomputed.
13. PINNED: per-(u,o) analysis replaces the OOD component of the
    axis triple with that single (u,o) risk (own corrected
    oracle + own 24-epoch IQR denominator; ID and WC unchanged);
    report per-(u,o) ρ̂*_LE and F_δ. ADDENDUM (outcome-blind
    design addition, recorded as such): also report, per run,
    whether any pool pair has disjoint OOD-axis feasible sets —
    the empirical two-world certificate count (spec-adjacent
    H.4; costs nothing beyond A9 outputs).
14. PINNED separation: the D-7 firing AUDIT is about the frozen
    pipeline → adjudicated on the 120-point IQRs as historically
    computed (did selection.py:31 map any zero scale to 0.0?).
    The corrected analyses use ONLY the 24-epoch IQR with
    fail-closed exclusion; report 24-grid zero-IQR incidence
    separately. The two never mix.
15. TGS-N SPLIT OUT as A10b — deferred registered addendum. The
    review side declines to pin the statistic from memory
    (transcription-fidelity principle). Battery proceeds WITHOUT
    TGS-N. A10b path: in a later cycle you fetch arXiv
    2605.08870 (as done for Label Wave), produce
    docs/TGS_TRANSCRIPTION.md (statistic, sampling, weights,
    direction, verbatim quotes + hashes) plus proposed pins;
    review side confirms; then it executes. Non-blocking.
16. Answered for the record: the "3 deviations" in A10 are
    TGS-specific (noisy-label conditioning; in-sample training
    examples in place of held-out source validation;
    detection-based R_OOD target in place of OOD-generalization
    ranking — spec D.5). LW-N's own three are as you listed.
17. CONFIRMED + PINNED: q = fraction of B = 1000 seeded
    resamples whose selected checkpoint attains ĝ_a ≤ δ on
    D_assess, where the assessment-side risk, corrected oracle,
    AND denominator are computed exclusively on D_assess.
    RNG stream per (dataset, b, n), seed 20260814.
18. PINNED: D_eval = the clean test set per dataset, split
    50/50 → D_sel/D_assess (class-stratified, seed 20260814);
    OOD pools likewise split 50/50 per pool. (ID→·): V_n from
    D_sel(test). (OOD→OOD): V_n = n/2 ID-sel + n/4 per semantic
    pool-sel, statistic = empirical 1−AUROC(energy) on the
    subsample. (mixed→OOD): n/2 ID-sel class-stratified + n/4
    per semantic pool-sel, selection by minimax over {ID, OOD}
    empirical ĝ. Budget curves cover ONLY the four listed
    pairs; WC excluded.
19. PINNED: O = seed. Verdict = the cell's majority A3 class
    across its runs; stability = whether the majority class
    changes when each seed is dropped. Frozen Phase-I cell
    verdicts are not recomputed.
20. PINNED: equivalence margin = 0.10 on NCR; criterion = BCa
    95% upper bound below the margin (descriptive, per spec
    E.2); the value was dropped in plan compression — restore
    by follow-up commit.
21. CORRECTED reading: granularity = aggregate (median over
    runs) vs per-run — from the frozen adjudication rule; metric
    scale = normalized regret vs epoch gap. NOT 24-vs-120 grid.
22. The forbidden-phrase list, verbatim (spec G.2):
    "intrinsically incompatible objectives" (absent
    population-scope evidence); "no selector can" (finite
    registered family); "deployment-relevant conflict" (absent
    external per-axis tolerances); "systematic failure under
    conflict generally" (conditioning). Binds all report prose.
23. PINNED: the A-battery synthetic fixtures are WRITTEN BY YOU
    as part of implementation, covering exactly the earlier
    Task-D list (exact ties; IQR = 0; singleton and empty
    feasible sets; minimax selection correctness;
    η̂ ≥ 0 within-estimand; exact uniform baseline |F_δ|/24;
    fold-partition leakage absence). The nine legacy unit tests
    are separate and stay green. Both suites gate execution.

A1-first question: NO — implement the whole battery (minus
A10b) in one commit, as your default proposed. Your A1
historical-comparison reading (each side as originally defined,
no re-denominating) is CONFIRMED.
