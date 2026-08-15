# Theoretical Specification v2 — No Clean Oracle

> **Transport note (recorded by the receiving session, not part of the specification).**
> This file is the text exactly as it reached the ext_realnoise session on 2026-08-15.
> The transport rendered every mathematical expression twice — once as spaced Unicode
> glyphs and once as a plain-text reduction — so the source is legible but not clean. It
> has NOT been de-duplicated, re-typeset, or corrected: "verbatim" is taken literally, and
> silently tidying a governing document is precisely the kind of unrecorded edit the
> protocol exists to prevent. The sha256 computed here therefore describes the
> post-transport text and necessarily differs from the review-side `213fddf7…`, which
> hashed the pre-transport original. A clean copy, if one can be transmitted, should be
> committed as a follow-up and both hashes retained.

Supersedes v1 (sha256 f0e0ba6d…, preserved under tag v1-pre-remediation; defects catalogued in docs/defect_ledger.md). Status tags: [FROZEN-HISTORICAL] · [DEFECT-IDENTIFIED] · [CORRECTED-PRIMARY] · [SENSITIVITY] · [EXPLORATORY] · [PENDING-Fn] (non-outcome fact n outstanding).

Central thesis (v2). The audit determines whether multi-objective checkpoint-selection failure arises because no jointly adequate checkpoint exists along the learned trajectory (trajectory incompatibility), or because such a checkpoint exists but cannot be identified from source-only noisy-training observables (checkpoint identifiability failure).

Notation (collision-free). Epochs
𝑡
∈
𝑇
t∈T,
∣
𝑇
∣
=
24
∣T∣=24, horizon
𝑇
max
⁡
=
120
T
max
	​

=120. Runs
𝑟
=
1
,
…
,
𝑛
r=1,…,n; cells = (dataset × noise × learner). Axes
𝑎
∈
{
I
D
,
W
C
,
O
O
D
}
a∈{ID,WC,OOD} (WC = worst-class, pending F3 rename confirmation). Softmax temperature
𝜏
τ. CV folds
𝐾
C
V
K
CV
	​

. OOD score type
𝑢
∈
{
m
s
p
,
e
n
e
r
g
y
}
u∈{msp,energy}, pool
𝑜
∈
{
1
,
2
,
3
}
o∈{1,2,3}. Selectors: E (confidence), NA (in-sample noisy agreement; formerly "C1"), ER (effective rank), LW-N (Label Wave adaptation), TGS-N-torsion / TGS-N-full (TopoGeoScore adaptations). Claims C1–C4 refer only to paper claims.

## A. Estimands and information sets

A.1 Population risk. For a fixed run,
𝑅
𝑎
(
𝑡
)
R
a
	​

(t) denotes the population deployment risk of checkpoint
𝜃
𝑡
θ
t
	​

 on axis
𝑎
a under the axis's evaluation distribution.
𝑅
𝑎
R
a
	​

 is never observed.

A.2 Finite evaluation metric.
𝑅
^
𝑎
(
𝑡
)
R
a
	​

(t) is the computed metric on the finite evaluation sample. Define the conditional mean estimand

𝜇
𝑎
(
𝑡
)
=
𝐸
[
𝑅
^
𝑎
(
𝑡
)
∣
𝐹
𝑟
]
,
μ
a
	​

(t)=E[
R
a
	​

(t)∣F
r
	​

],

where
𝐹
𝑟
F
r
	​

 (A.3) fixes the run. Whether
𝜇
𝑎
(
𝑡
)
=
𝑅
𝑎
(
𝑡
)
μ
a
	​

(t)=R
a
	​

(t) is metric-specific: sample-mean error metrics — yes; AUROC-type two-sample statistics — approximately, via U-statistic structure; FPR@95TPR and threshold-estimated risks — not in general (finite-sample bias) [PENDING-F4 for the actual OOD functional]. All bias results in §B are stated for
𝜇
𝑎
μ
a
	​

; transfer to
𝑅
𝑎
R
a
	​

 is a separately stated, metric-specific assumption.

A.3 Information sets.
𝐹
𝑟
=
𝜎
(
noisy data, corruption realization, optimizer randomness, trajectory, selector-side observables and hyperparameters
)
F
r
	​

=σ(noisy data, corruption realization, optimizer randomness, trajectory, selector-side observables and hyperparameters). Admissible selector:
𝑡
^
𝑠
t
^
s
	​

 is
𝐹
𝑟
F
r
	​

-measurable — evaluation samples, clean labels, and OOD pool membership are excluded from its domain.

A.4 Three benchmark layers (never conflated). [CORRECTED-PRIMARY]

Layer	Definition	Meaning
Population oracle
𝑅
𝑎
∗
=
min
⁡
𝑡
𝑅
𝑎
(
𝑡
)
R
a
∗
	​

=min
t
	​

R
a
	​

(t)	best attainable on the trajectory; an estimand, not a statistic
Empirical lower envelope (LE)
𝑅
^
𝑎
∗
,
L
E
=
min
⁡
𝑡
𝑅
^
𝑎
(
𝑡
)
R
a
∗,LE
	​

=min
t
	​

R
a
	​

(t)	optimistic benchmark from searching one evaluation sample $
Cross-fitted selection benchmark (CF)
1
𝐾
C
V
∑
𝑞
𝑅
^
𝑎
(
𝑞
)
(
arg
⁡
min
⁡
𝑡
𝑅
^
𝑎
(
−
𝑞
)
(
𝑡
)
)
K
CV
	​

1
	​

∑
q
	​

R
a
(q)
	​

(argmin
t
	​

R
a
(−q)
	​

(t))	expected risk of a data-driven selection procedure; NOT an estimator of the population oracle

[DEFECT-IDENTIFIED, v1] v1 called the CF quantity "approximately unbiased oracle risk." Corrected: the CF benchmark estimates a procedure's risk;
𝐸
[
C
F
]
≥
min
⁡
𝑡
𝜇
𝑎
(
𝑡
)
E[CF]≥min
t
	​

μ
a
	​

(t) in general, and selector-minus-CF differences may be negative — they are cross-fitted benchmark excesses, not regrets.

## B. Oracle estimation bias

B.1 Conditional winner's-curse proposition. Assume: (i) the evaluation sample ⟂
𝐹
𝑟
F
r
	​

; (ii)
𝜇
𝑎
(
𝑡
)
=
𝐸
[
𝑅
^
𝑎
(
𝑡
)
∣
𝐹
𝑟
]
μ
a
	​

(t)=E[
R
a
	​

(t)∣F
r
	​

]; (iii)
𝑡
^
𝑠
t
^
s
	​

 is
𝐹
𝑟
F
r
	​

-measurable; (iv) oracle and selector share
𝑇
T. Then

𝐸
[
𝑅
^
𝑎
(
𝑡
^
𝑠
)
∣
𝐹
𝑟
]
=
𝜇
𝑎
(
𝑡
^
𝑠
)
,
𝐸
[
min
⁡
𝑡
𝑅
^
𝑎
(
𝑡
)
∣
𝐹
𝑟
]
≤
min
⁡
𝑡
𝜇
𝑎
(
𝑡
)
,
E[
R
a
	​

(
t
^
s
	​

)∣F
r
	​

]=μ
a
	​

(
t
^
s
	​

),E[
t
min
	​

R
a
	​

(t)∣F
r
	​

]≤
t
min
	​

μ
a
	​

(t),

so the empirical selector excess over the LE benchmark is conditionally upward-biased for the
𝜇
μ-scale excess:

𝐸
[
𝑅
^
𝑎
(
𝑡
^
𝑠
)
−
𝑅
^
𝑎
∗
,
L
E
∣
𝐹
𝑟
]
−
[
𝜇
𝑎
(
𝑡
^
𝑠
)
−
min
⁡
𝑡
𝜇
𝑎
(
𝑡
)
]
=
min
⁡
𝑡
𝜇
𝑎
(
𝑡
)
−
𝐸
[
min
⁡
𝑡
𝑅
^
𝑎
(
𝑡
)
∣
𝐹
𝑟
]
≥
0.
E[
R
a
	​

(
t
^
s
	​

)−
R
a
∗,LE
	​

∣F
r
	​

]−[μ
a
	​

(
t
^
s
	​

)−
t
min
	​

μ
a
	​

(t)]=
t
min
	​

μ
a
	​

(t)−E[
t
min
	​

R
a
	​

(t)∣F
r
	​

]≥0.

Direction: measured selector failure overstates
𝜇
μ-scale failure. Magnitude: bounded in the worst case by a
𝜎
log
⁡
∣
𝑇
∣
σ
log∣T∣
	​

-type term under mean-zero sub-Gaussian errors, but not monotone in
∣
𝑇
∣
∣T∣ or in cross-
𝑡
t correlation for arbitrary covariance structures; equicorrelated/common-noise heuristics ("shared test set shrinks the bias") are stated as heuristics only. "Negligible for ID at
𝑛
=
10,000
n=10,000" is a hypothesis to be checked, not assumed.

B.2 Cross-regret is doubly contaminated.
𝑡
𝑗
∗
,
L
E
=
arg
⁡
min
⁡
𝑡
𝑅
^
𝑗
(
𝑡
)
t
j
∗,LE
	​

=argmin
t
	​

R
j
	​

(t) uses evaluation data, so empirical cross-regret
𝑅
^
𝑗
′
(
𝑡
𝑗
∗
,
L
E
)
−
𝑅
^
𝑗
′
∗
,
L
E
R
j
′
	​

(t
j
∗,LE
	​

)−
R
j
′
∗,LE
	​

 carries (i) selection bias through axis
𝑗
j, (ii) LE optimism on axis
𝑗
′
j
′
, (iii) cross-axis noise correlation (shared ID examples). [SENSITIVITY] Cross-fitted cross-regret: select both oracles on folds
−
𝑞
−q, compare on fold
𝑞
q; this estimates a difference between two data-driven procedures, may be negative, and is reported alongside — never as the population cross-regret.

## C. Multi-objective compatibility

C.1 Pairwise cross-regret (retained as descriptor).
C
R
[
𝑗
′
∣
𝑗
]
=
𝑅
𝑗
′
(
𝑡
𝑗
∗
)
−
𝑅
𝑗
′
∗
CR[j
′
∣j]=R
j
′
	​

(t
j
∗
	​

)−R
j
′
∗
	​

: measures oracle disagreement, not joint infeasibility (counterexample: a compromise checkpoint δ-good on all axes can coexist with large pairwise CR). [FROZEN-HISTORICAL] Phase-I NCR verdicts stand as registered outcomes under this measure.

**C.2 Normalized per-axis regret with pinned denominator.** [PIN 2]

𝑔
𝑎
(
𝑡
)
=
𝑅
𝑎
(
𝑡
)
−
𝑅
𝑎
∗
𝑑
𝑎
,
𝑑
𝑎
,
𝑟
=
max
⁡
{
I
Q
R
𝑡
∈
𝑇
𝑅
^
𝑎
,
𝑟
(
𝑡
)
,
 
𝜖
𝑎
}
,
g
a
	​

(t)=
d
a
	​

R
a
	​

(t)−R
a
∗
	​

	​

,d
a,r
	​

=max{IQR
t∈T
	​

R
a,r
	​

(t), ϵ
a
	​

},

with
𝜖
𝑎
ϵ
a
	​

 fixed **before** any real-log inspection by rule, not by observed IQRs: exactly-zero denominators are fail-closed (run excluded-with-flag on that axis, primary), and a pre-fixed floor grid is run as [SENSITIVITY]. Interpretation is binding:
𝑔
𝑎
g
a
	​

 is **trajectory-relative**, not deployment significance. The raw regret vector
Δ
𝑎
(
𝑡
)
=
𝑅
^
𝑎
(
𝑡
)
−
𝑅
^
𝑎
∗
Δ
a
	​

(t)=
R
a
	​

(t)−
R
a
∗
	​

 accompanies every normalized quantity; "deployment-relevant conflict" language requires externally justified per-axis tolerances
𝜏
𝑎
τ
a
	​

, which this paper does not claim to possess ("trajectory-relative compatibility under the registered normalization" is the licensed phrase). Properties, corrected from v1 [DEFECT-IDENTIFIED]: invariance holds under strictly increasing affine maps (
𝑎
>
0
a>0) only; flat trajectories make NCR-type ratios large or undefined only when the denominator vanishes faster than the numerator or the mid-50% is degenerate while oracle/victim points are outliers.

**C.3 Joint compatibility radius — three layers.** [PIN 1] For run
𝑟
r:

𝜌
𝑟
∗
=
min
⁡
𝑡
∈
𝑇
max
⁡
𝑎
𝑔
𝑎
,
𝑟
(
𝑡
)
(population estimand)
;
ρ
r
∗
	​

=
t∈T
min
	​

a
max
	​

g
a,r
	​

(t)(population estimand);
𝜌
^
L
E
,
𝑟
∗
=
min
⁡
𝑡
max
⁡
𝑎
𝑔
^
𝑎
,
𝑟
(
𝑡
)
,
𝑔
^
𝑎
,
𝑟
(
𝑡
)
=
𝑅
^
𝑎
,
𝑟
(
𝑡
)
−
𝑅
^
𝑎
,
𝑟
∗
,
L
E
𝑑
𝑎
,
𝑟
(descriptive empirical value)
;
ρ
	​

LE,r
∗
	​

=
t
min
	​

a
max
	​

g
	​

a,r
	​

(t),
g
	​

a,r
	​

(t)=
d
a,r
	​

R
a,r
	​

(t)−
R
a,r
∗,LE
	​

	​

(descriptive empirical value);

and the **cross-fitted minimax selection benchmark**:
𝑡
^
𝜌
,
−
𝑞
=
arg
⁡
min
⁡
𝑡
max
⁡
𝑎
𝑔
^
𝑎
(
−
𝑞
)
(
𝑡
)
t
^
ρ,−q
	​

=argmin
t
	​

max
a
	​

g
	​

a
(−q)
	​

(t) evaluated on fold
𝑞
q (a procedure's performance, not an estimator of
𝜌
∗
ρ
∗
). The bias of
𝜌
^
L
E
∗
ρ
	​

LE
∗
	​

 has **no general sign** (per-axis LE optimism inflates
𝑔
^
g
	​

; the outer min over 24 checkpoints deflates the compromise). Licensed language:
𝜌
^
L
E
∗
>
𝛿
ρ
	​

LE
∗
	​

>δ = "empirically incompatible";
≤
𝛿
≤δ = "empirically compatible"; population-level intrinsic incompatibility is claimed only if the verdict survives all registered sensitivities and the CF benchmark.

**C.4 Feasible set and window width.**

𝐹
𝛿
,
𝑟
=
{
𝑡
:
max
⁡
𝑎
𝑔
𝑎
,
𝑟
(
𝑡
)
≤
𝛿
}
,
𝑤
𝛿
,
𝑟
=
∣
𝐹
𝛿
,
𝑟
∣
∣
𝑇
∣
.
F
δ,r
	​

={t:
a
max
	​

g
a,r
	​

(t)≤δ},w
δ,r
	​

=
∣T∣
∣F
δ,r
	​

∣
	​

.

𝐹
𝛿
,
𝑟
=
∅
⟺
𝜌
𝑟
∗
>
𝛿
F
δ,r
	​

=∅⟺ρ
r
∗
	​

>δ.
𝑤
𝛿
w
δ
	​

 grades difficulty among compatible runs: failure at
𝑤
=
12
/
24
w=12/24 is far stronger evidence of information insufficiency than at
𝑤
=
1
/
24
w=1/24.

C.5 Set-valued and ε-optimal oracles. [SENSITIVITY]
𝑂
𝑎
(
𝜖
)
=
{
𝑡
:
𝑅
𝑎
(
𝑡
)
≤
𝑅
𝑎
∗
+
𝜖
}
O
a
	​

(ϵ)={t:R
a
	​

(t)≤R
a
∗
	​

+ϵ}; report
C
R
min
⁡
/
max
⁡
[
𝑏
∣
𝑎
]
CR
min/max
	​

[b∣a] over
𝑂
𝑎
(
𝜖
)
O
a
	​

(ϵ) for a pre-fixed
𝜖
ϵ grid, because single empirical argmins flip under tiny evaluation noise and the flip can reverse conflict verdicts. Earliest-tie-break remains the frozen operational rule; the set-valued analysis is the corrected descriptor.

## D. Selector insufficiency

**D.1 Joint selector regret and inefficiency.** [PIN 3]

𝐽
𝑠
,
𝑟
=
max
⁡
𝑎
𝑔
𝑎
,
𝑟
(
𝑡
^
𝑠
)
,
𝜂
𝑠
,
𝑟
=
𝐽
𝑠
,
𝑟
−
𝜌
𝑟
∗
≥
0
,
J
s,r
	​

=
a
max
	​

g
a,r
	​

(
t
^
s
	​

),η
s,r
	​

=J
s,r
	​

−ρ
r
∗
	​

≥0,

with nonnegativity valid **only within one estimand layer** (same risk estimates, normalization, grid, evaluation data, tie rules):
𝜂
^
𝑠
,
L
E
=
𝐽
^
𝑠
,
L
E
−
𝜌
^
L
E
∗
≥
0
η
	​

s,LE
	​

=
J
s,LE
	​

−
ρ
	​

LE
∗
	​

≥0 holds; mixing layers (e.g., full-sample
𝐽
^
J
 minus CF benchmark) can be negative and is not an inefficiency.

D.2 Corrected run taxonomy. [CORRECTED-PRIMARY; PIN 4]

State	Condition	Reading
Incompatible
𝜌
𝑟
∗
>
𝛿
ρ
r
∗
	​

>δ (empirically:
𝜌
^
L
E
∗
>
𝛿
ρ
	​

LE
∗
	​

>δ, surviving sensitivities)	joint success impossible; selector failure here is necessary, not evidence against selectors
Compatible–solved
𝜌
𝑟
∗
≤
𝛿
ρ
r
∗
	​

≤δ, some selector with
𝐽
𝑠
≤
𝛿
J
s
	​

≤δ	selector sufficiency
Compatible–unsolved
𝜌
𝑟
∗
≤
𝛿
ρ
r
∗
	​

≤δ, all selectors fail	information insufficiency — the audit's sharpest possible finding
Indeterminate	uncertainty straddles
𝛿
δ	withheld

Selector insufficiency is adjudicated on compatible runs only. The historical "0/45" [FROZEN-HISTORICAL] decomposes into: #incompatible runs; #compatible; per-selector joint successes on compatible runs; #compatible–unsolved; best-achievable gaps on incompatible runs. Either extreme is publishable: mostly-incompatible → trajectory-incompatibility paper; mostly-compatible-yet-unsolved → identifiability-failure paper.

D.3 Exact chance baselines (replacing the binomial rationale [DEFECT-IDENTIFIED:
𝑝
=
0.5
p=0.5 null was arbitrary]). Uniform-random checkpoint selection has exact per-run joint success probability
𝑝
u
n
i
f
,
𝑟
=
𝑤
𝛿
,
𝑟
p
unif,r
	​

=w
δ,r
	​

 and exact expected joint regret
1
∣
𝑇
∣
∑
𝑡
max
⁡
𝑎
𝑔
𝑎
,
𝑟
(
𝑡
)
∣T∣
1
	​

∑
t
	​

max
a
	​

g
a,r
	​

(t) — no simulation.
𝜋
=
0.8
π=0.8 is retained solely as a preregistered high-consistency criterion, non-inferential; nominal binomial tail probabilities are removed from all claim-bearing text.

D.4 S2 joint definition. [PIN, Tier 1 onward] "Simultaneous" success = interpretation B: per-run joint indicator
1
{
max
⁡
𝑎
𝑟
(
𝑠
,
𝑎
,
𝑘
)
≤
𝛿
}
1{max
a
	​

r(s,a,k)≤δ}, reliability
∑
𝑘
≥
⌈
𝜋
𝑛
⌉
∑
k
	​

≥⌈πn⌉ (
=
29
=29 at
𝑛
=
36
n=36). Phase II's classification is unaffected (max marginal success 3/15 rejects S2 under either reading — recorded as a robustness fact, not re-adjudicated).

D.5 Selector-specific closure items. E: temperature is
𝜏
τ; both epoch-robustness and deployment-risk-robustness across
𝜏
τ reported; primary
𝜏
=
1
τ=1 was pinned pre-outcome — no study-level
𝜏
τ selection. NA: renamed; the memorization statement is weakened to "optimistically biased in-sample agreement, increasing toward the interpolation regime" (no upper-bound theorem). ER:
0
log
⁡
0
:
=
0
0log0:=0; zero-spectrum fail-closed. LW-N: pending F1; if reconstructed on the 24-grid it is named LabelWave-N (coarse-grid adaptation), never "Label Wave". TGS-N: three registered deviations (noisy-label conditioning; in-sample training examples in place of held-out source validation; detection-based
𝑅
O
O
D
R
OOD
	​

 target in place of OOD-generalization ranking); torsion variant runs on all Phase-II and Tier-1 runs as the common transport selector; full variant on Phase-II only; separate selectors, no pooling; failure of the adaptations is not evidence against the published method.

## E. Inference, conditioning, and the three-layer record

E.1 Conditioning structure. Phase-II trajectories are conflict-enriched by Phase-I selection; licensed wording: "the 15 audited trajectories belonging to the five empirically conflict-enriched configurations." Tier 1 was designed and launched before any remediation analysis: if its selector outputs remain sealed until the anchored plan, it functions as prospective confirmation for the corrected taxonomy; otherwise it is an extended audit — the seal status determines the label, and is itself recorded.

E.2 Small-seed inference. [FROZEN-HISTORICAL] 3-seed BCa intervals are descriptive uncertainty only (10 distinct resampling multisets; non-smooth ratio statistics; shared test set excludes evaluation-sample uncertainty). The Phase-I KILL rule treated CI-overlap-with-zero as absence evidence [DEFECT-IDENTIFIED]; the corrected equivalence-style reanalysis (upper confidence bound within a pre-fixed practical-null region) is reported as [SENSITIVITY] without altering the frozen WEAK verdict. Targeted +2 seeds are replication (reported separately from discovery seeds; pooled 5-seed estimates secondary), never unconditional confirmation.

E.3 Registration-scope principle (protocol text, verbatim): "Preregistration prevents undisclosed adaptivity; it does not repair an invalid estimand, an overlapping assessment set, or an inappropriate inferential procedure." Frozen results are preserved; defects are ledgered; claims are licensed from [CORRECTED-PRIMARY] analyses only; where frozen and corrected disagree, both are published.

## F. Information-budget theory (clean-label supply curves)

**F.1 Corrected design.** [DEFECT-IDENTIFIED in v1: selection and assessment overlapped.] Evaluation data are partitioned once,
𝐷
e
v
a
l
=
𝐷
s
e
l
∪
˙
𝐷
a
s
s
e
s
s
D
eval
	​

=D
sel
	​

∪
˙
D
assess
	​

 (or outer cross-fitting); budget samples
𝑉
𝑛
⊂
𝐷
s
e
l
V
n
	​

⊂D
sel
	​

; assessment exclusively on
𝐷
a
s
s
e
s
s
D
assess
	​

:

𝑞
𝑏
→
𝑎
(
𝑛
,
𝛿
)
=
Pr
⁡
[
𝑔
𝑎
a
s
s
e
s
s
(
𝑡
^
𝑏
(
𝑉
𝑛
)
)
≤
𝛿
]
,
𝑡
^
𝑏
(
𝑉
𝑛
)
=
arg
⁡
min
⁡
𝑡
𝑅
^
𝑏
(
𝑡
;
𝑉
𝑛
)
,
q
b→a
	​

(n,δ)=Pr[g
a
assess
	​

(
t
^
b
	​

(V
n
	​

))≤δ],
t
^
b
	​

(V
n
	​

)=arg
t
min
	​

R
b
	​

(t;V
n
	​

),

separating selection objective
𝑏
b from evaluated axis
𝑎
a:
𝑞
I
D
→
I
D
q
ID→ID
	​

,
𝑞
I
D
→
O
O
D
q
ID→OOD
	​

,
𝑞
O
O
D
→
O
O
D
q
OOD→OOD
	​

,
𝑞
m
i
x
e
d
→
O
O
D
q
mixed→OOD
	​

. "
𝑛
∗
n
∗
 not reached" is operationalized:
𝑛
max
⁡
n
max
	​

 stated;
𝑞
(
𝑛
)
<
0.9
q(n)<0.9 for all
𝑛
≤
𝑛
max
⁡
n≤n
max
	​

; non-monotone curves never define
𝑛
∗
n
∗
 by first crossing; Monte Carlo variation is conditional on the fixed evaluation dataset (outer resampling required for dataset-level uncertainty).

F.2 Same-axis anchor (theorem scope:
𝑎
→
𝑎
a→a only). [PIN 5] For bounded additive metrics and
𝑀
=
∣
𝑇
∣
M=∣T∣ candidates, with
𝑛
n independent clean selection samples, w.p.
≥
1
−
𝜁
≥1−ζ:

𝑅
𝑎
(
𝑡
^
𝑎
(
𝑛
)
)
−
𝑅
𝑎
∗
≤
2
log
⁡
(
2
𝑀
/
𝜁
)
2
𝑛
.
R
a
	​

(
t
^
a
	​

(n))−R
a
∗
	​

≤2
2n
log(2M/ζ)
	​

	​

.

Not applicable as-is to AUROC (U-statistic bound needed), quantile-thresholded metrics, data-selected class subsets, or normalized (random-denominator) scales.

F.3 Cross-axis structural mismatch (proposition). For
𝑏
≠
𝑎
b

=a with a unique
𝑏
b-oracle and consistent selection,
𝑅
𝑎
(
𝑡
^
𝑏
(
𝑛
)
)
−
𝑅
𝑎
∗
→
C
R
[
𝑎
∣
𝑏
]
R
a
	​

(
t
^
b
	​

(n))−R
a
∗
	​

→CR[a∣b] as
𝑛
→
∞
n→∞: if
C
R
[
O
O
D
∣
I
D
]
CR[OOD∣ID] exceeds the tolerance,
𝑛
I
D
→
O
O
D
∗
n
ID→OOD
∗
	​

 is unreachable at any budget — sample scarcity and information-currency mismatch are distinct failure modes, and the supply curve identifies which one binds. With set-valued ID oracles, the limit lies in
[
C
R
‾
,
C
R
‾
]
[
CR
	​

,
CR
] over
𝑂
I
D
O
ID
	​

, and tie-breaking information determines the point.

## G. Claim licensing

**G.1 Two-world non-identifiability theorem.** Let
𝑋
X be all selector-visible information, and let worlds
𝑃
0
,
𝑃
1
P
0
	​

,P
1
	​

 satisfy
𝐿
𝑃
0
(
𝑋
)
=
𝐿
𝑃
1
(
𝑋
)
L
P
0
	​

	​

(X)=L
P
1
	​

	​

(X) with disjoint δ-good sets
𝐴
0
,
𝛿
∩
𝐴
1
,
𝛿
=
∅
A
0,δ
	​

∩A
1,δ
	​

=∅. Then any
𝑋
X-measurable (possibly randomized) selector
𝑆
S obeys

min
⁡
𝑤
∈
{
0
,
1
}
𝑃
𝑤
(
𝑆
(
𝑋
)
∈
𝐴
𝑤
,
𝛿
)
≤
1
2
,
and under approximate indistinguishability
max
⁡
𝑤
𝑃
𝑤
(
failure
)
≥
1
−
T
V
(
𝑃
0
𝑋
,
𝑃
1
𝑋
)
2
.
w∈{0,1}
min
	​

P
w
	​

(S(X)∈A
w,δ
	​

)≤
2
1
	​

,and under approximate indistinguishability
w
max
	​

P
w
	​

(failure)≥
2
1−TV(P
0
X
	​

,P
1
X
	​

)
	​

.

Scope discipline: this licenses "without additional structure on the deployment-shift class, uniformly reliable source-only checkpoint selection is impossible" — not "all realistic shifts defeat all selectors." The concrete construction ties theory to audit: identical noisy-training distribution and trajectory observables; two admissible OOD distributions reversing the checkpoint ranking with disjoint δ-good sets. Paper structure: general theorem + empirical compatible–unsolved incidence.

G.2 Licensing table.

Claim	Licensed formulation	Requires
C1	Axis-specific empirical optima frequently differ; deploying one axis's empirical oracle incurs non-negligible raw cross-regret on another, with trajectory-relative magnitudes reported under the registered normalization	frozen Phase-I + corrected sensitivities
C1′ (new)	Empirical joint incompatibility (
𝜌
^
L
E
∗
>
𝛿
ρ
	​

LE
∗
	​

>δ) in [count] of audited runs, robust to registered sensitivities	corrected ρ-analysis
C2	On the 15 conflict-enriched audited trajectories, no registered evaluation-blind selector met the joint criterion; decomposition into incompatible vs compatible–unsolved attributes the failure	corrected taxonomy + CF sensitivity
C3	The qualitative pattern reproduced on the prespecified human-annotation-noise datasets and learner mechanisms (named scope; sealed-until-anchored Tier-1 counts as prospective confirmation)	Tier-1 unsealing after anchoring
C4	Existence check at scale (own table)	Tier 2
Budget corollary	Empirical selection-budget curves
𝑞
𝑏
→
𝑎
q
b→a
	​

 with the structural-mismatch reading	corrected F-design
Impossibility	G.1 wording only	theorem + construction
Forbidden without further evidence: "intrinsically incompatible objectives" (needs
𝜌
∗
>
𝛿
ρ
∗
>δ at population scope), "no selector can" (finite registered family), "deployment-relevant conflict" (needs external
𝜏
𝑎
τ
a
	​

), "systematic failure under conflict generally" (conditioning).

## Open-items ledger

F1–F4 non-outcome facts [PENDING]; outcome-bearing quantities sealed until the anchored Remediation Analysis Plan v2; external anchoring method [user decision — hard blocker]; 2-world concrete construction to be written for the paper appendix; F3 outcome determines WC naming.
