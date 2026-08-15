# Framing Preregistration — Phase-0 → Paper
Status: registered BEFORE any confirmatory baseline or E output
was computed or inspected. Stored logits/features exist on disk as
shared inputs; no selector results exist at registration time.

## Definitions (thresholds finalized)
- Axes: ID, tail-dynamic, OOD-primary (as pinned by the G1 protocol).
- Confirmatory selector set: {E (T-grid primary variant),
  C1 noisy-validation, (b) representation-based}.
  full-D is EXCLUDED from scenario classification
  (analysis_class: exploratory; scope caveat on record).
- r(s, a, k): normalized regret of selector s on axis a in run k,
  relative to the per-axis oracle epoch (G1 frame).
- Success on an axis: r ≤ δ, with δ = 0.10.
  Normalized regret r is defined as
  r(s, a, k) = (R_a[t_selected] − R_a[t*_a]) / IQR_a, using the
  oracle epoch and IQR definitions of docs/analysis_protocol.md §5;
  implemented in src/analysis/selection.py::regret_at_epoch.
  Rationale: (i) 10% relative-to-oracle degradation is the
  operational boundary for "effectively oracle-free"; (ii) δ lies
  below the best median regret observed in the EXPLORATORY full-D
  analysis (+0.14), which is excluded from classification — the
  threshold is informed only by excluded pilot data; no
  confirmatory selector output existed at registration time.
- Reliable success: success in ≥ 12 of 15 runs (π = 0.8).
  Rationale: under H0 of coin-flip success (p = 0.5),
  P(≥12/15) ≈ 0.018 — chance-level performance is rejected at
  ~2%. The same π applies to the S1 axis-disagreement condition;
  this reuse is intentional and shares the same binomial logic.
- Sensitivity (non-binding): classification is additionally
  reported under δ = 0.05 and δ = 0.20 for transparency;
  the binding classification uses δ = 0.10 only.

## Mechanical classification rule (fixed now)
- S2 (sufficiency): at least one selector in the confirmatory set
  achieves reliable success on ALL three axes simultaneously.
- S1 (conflict / no free lunch): no selector achieves S2, AND the
  per-axis best selector differs across axes in ≥ π of runs.
- S3 (mixed): neither S1 nor S2 holds.
Ties and boundary cases: classified by the rule as written;
no post-hoc threshold adjustment. Any deviation requires a
timestamped amendment committed BEFORE further analysis.

## E prediction adjudication (separate from classification)
The frozen predictions (late-epoch selection bias; largest
oracle gap on OOD) are adjudicated exactly as registered in
G2_design_memo.md. Their outcome strengthens or weakens the
narrative but does NOT gate the S1/S2/S3 classification.

## Framing commitments (binding)
- S1 → audit-paper framing ("no clean oracle, no free lunch across
  deployment axes"); target analysis-track venues (TMLR / NeurIPS
  D&B / analysis tracks); a real-noise dataset extension is
  scheduled as a required robustness addition before submission.
- S2 → sufficiency framing; positioning obligation: explicit
  boundary against existing noisy-validation literature (criterion
  vs. evaluation-axes contribution); method track becomes eligible.
- S3 → conditional-guideline framing ONLY if a compact rule
  (≤ 2 factors, e.g., noise type × learner) explains the pattern;
  otherwise default to S1-style audit with reduced claims.
- No reframing after results are seen except via timestamped
  amendment with stated rationale.

---

*Recording note (not part of the registered rule).* The issued text carried the
placeholder `[VERIFY & CITE the exact definition here before commit]` inside the δ
definition. Per that instruction the verification was performed before this commit and
the placeholder replaced with the verified citation shown above. What the verification
found is on record in `docs/framing_preregistration.md`: `analysis_protocol.md` does
**not** define selector regret — the word "selector" does not occur in it — so the
citation names §5 for the oracle-epoch and IQR definitions it does supply, and names
`src/analysis/selection.py::regret_at_epoch` for the selector-regret formula itself.
No unfilled threshold placeholder remained in the issued text — both δ and π arrived
with values, so the STOP condition did not fire. (This note deliberately avoids writing
the placeholder token itself, so that a mechanical scan of this file for it stays
meaningful.) Open items previously reported and not
addressed by this issuance — the S1 disagreement clause's null rate, B3 (best-selector
tie rule), B4 (E's confirmatory variant under T-sensitivity), B5 (missing-output
handling), B6, and the conflict-selected sample scope — remain open; they are recorded
in `docs/framing_preregistration.md` and are not resolved here.
