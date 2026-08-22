# SOP transcription record

Plain SOP is transcribed from the authors' own sources, not reconstructed. This file
records where every value came from, and the two places where the sources disagree.

## Sources

| what | where |
|---|---|
| loss implementation | `model/loss.py::overparametrization_loss.forward`, https://github.com/shengliu66/SOP |
| repository commit | **`4d991cedf1fafec98f858a213ccc31e52318a77f`** (main, 2022-09-15T01:14:50Z) |
| optimizer wiring | `train.py` lines 83-89 of that commit |
| config defaults | `config_cifar100.json` of that commit |
| hyperparameter table | Table A.1, arXiv:2202.14026 (ICML 2022) |

## Transcribed values

**Table A.1, `lr for {u_i, v_i}`** — the columns are (CIFAR-10 / ResNet34),
(CIFAR-10 / PreActResNet18), (CIFAR-100 / ResNet34), (CIFAR-100 / PreActResNet18),
(Clothing-1M), (WebVision):

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| symmetric | `alpha_u = 10, alpha_v = 10` | `alpha_u = 1, alpha_v = 10` |
| asymmetric | `alpha_u = 10, alpha_v = 100` | `alpha_u = 1, alpha_v = 100` |

The values are identical across the two architecture columns within a dataset, so they
are a function of the dataset, not of the backbone — which is what makes them
transcribable onto our unified PreActResNet18 protocol.

**Also from Table A.1 and the config, all unambiguous:**

- `wd for {u,v}` = **0** in every column — u and v are excluded from weight decay. The
  official `train.py` sets this per param group, and `optimizer_overparametrization` is
  SGD with `momentum 0`, `weight_decay 0`.
- `init std for {u,v}` = **1e-8**, mean 0, gaussian (Table A.1 and
  `config_cifar100.json: reparam_arch.args`).
- `eps` = **1e-4** (`model/loss.py`).
- `u` has shape **(N, 1)** and `v` has shape **(N, C)** — not both (N, C). Easy to get
  wrong from the paper's notation alone; taken from the code.

## Plain SOP vs SOP+ — resolved by Table A.1, not by us

The preregistration pins "plain SOP (NOT SOP+; no consistency regularizer)" but does not
say what to do with the class-balance regularizer, which is a separate term. Table A.1
settles it:

| | CIFAR-10 R34 | CIFAR-10 PreAct18 | CIFAR-100 R34 | CIFAR-100 PreAct18 |
|---|---|---|---|---|
| `lambda_C` (consistency) | **0.0** | 0.9 | **0.0** | 0.9 |
| `lambda_B` (balance) | **0.0** | 0.1 | **0.0** | 0.1 |

The ResNet34 columns are plain SOP and carry **both** regularizers at zero; the
PreActResNet18 columns are the SOP+ configuration. So plain SOP means
`lambda_C = lambda_B = 0`, and that is what is implemented. No judgement call was made
here — the paper's own table distinguishes the two.

Note the consequence for our protocol: the paper's PreActResNet18 numbers are SOP+
numbers. We use PreActResNet18 with plain SOP, a combination the paper does not report.
That is a direct consequence of Decision A (unified trajectory protocol) and is already
covered by the preregistration's stated limitation.

## Discrepancy 1 — asymmetric `alpha_u`, paper vs repository

The repository README states, for 40% asymmetric noise on CIFAR-100:

> `python train.py -c config_cifar100.json --lr_u 0.1 --lr_v 100 --percent 0.4 --name CIFAR100 --asym True`
>
> "(Because we modified the code for better delivery after camera ready, for asymmetric
> noise we use lr for u equals 0.1 rather than 1 in the paper)"

Table A.1 gives `alpha_u = 1` for CIFAR-100 asymmetric. The authors document the repo
value as superseding. Per the preregistration ("the official repository … supersedes the
paper where they differ"), the operative asymmetric CIFAR-100 value is
**`alpha_u = 0.1, alpha_v = 100`**.

## Discrepancy 2 — weight decay for theta, paper vs repository

Table A.1 gives `wd = 5e-4` for all four CIFAR columns. `config_cifar100.json` sets
`optimizer.args.weight_decay = 1e-3`. This one does not affect us: our unified protocol
pins the sealed G1/B2 optimizer (`wd = 5e-4`), and learner-specific *schedules and
optimizers* are explicitly not transcribed. Recorded because it is a second point where
the two sources differ and a future reader should not be surprised by it.

## OPEN — no transcribable `alpha_u` / `alpha_v` exists for CIFAR-N

**This blocks the SOP runs and needs a pin. It does not block CE, ELR or GCE.**

Table A.1 gives values for *symmetric* and *asymmetric* synthetic noise only. The repo
ships one config, `config_cifar100.json`, whose defaults (`lr_u 1, lr_v 10`) are the
CIFAR-100 symmetric row. CIFAR-N is real human annotation noise: neither symmetric nor
asymmetric, and no source in either the paper or the repository states a value for it.

There is therefore nothing to transcribe, and picking one would be exactly the
improvisation the protocol forbids. Options, for a pin:

1. **Symmetric row** — CIFAR-10 `(10, 10)`, CIFAR-100 `(1, 10)`. The repository's shipped
   default; the most literal reading of "transcribed per config".
2. **Asymmetric row, repo-superseded** — CIFAR-10 `(10, 100)`, CIFAR-100 `(0.1, 100)`.
   Defensible on noise structure: human confusions are class-dependent, closer to
   asymmetric than to uniform. But this is an inference about CIFAR-N, not a
   transcription.
3. **Both, as a registered sensitivity** — doubles the SOP cell count (12 runs -> 24).

Recommendation: **option 1**, because it is the only one that is actually a
transcription, with the choice and its reasoning recorded here. Option 2 should be taken
only if the review side wants the noise-structure argument on the record as a
registered decision.

Until this is pinned, `configs/base.yaml` carries no `alpha` for CIFAR-N and
`build_loss` raises rather than defaulting.

---

## PIN — CIFAR-N alpha_u / alpha_v (received 2026-08-13, Option 1)

**Pinned values**

| split | `alpha_u` | `alpha_v` |
|---|---|---|
| `c10n_worst` | 10 | 10 |
| `c10n_random1` | 10 | 10 |
| `c100n` | 1 | 10 |

**Recorded rationale (verbatim as issued)**

> CIFAR-N carries real human noise, addressed by neither the symmetric nor asymmetric
> synthetic rows of Table A.1. No transcribable CIFAR-N value exists in paper or
> repository. Per the registered transcription-first principle, the repository default
> path (symmetric row) is adopted as the only value that is a transcription rather than
> an inference. The asymmetric row was considered and rejected: human noise is
> instance-dependent (per the CIFAR-N benchmark's own characterization), so the
> class-dependent analogy underlying that choice is itself an inference. A ±sensitivity
> grid over both rows is TABLED, to be invoked only if SOP cells prove pivotal to
> classification.

`build_loss` keeps no default: the exception on a missing `alpha_u`/`alpha_v` stays as a
guard, so the pinned values must be supplied by config and cannot be silently reinstated
if the config is edited.

## PIN — Clothing-1M column (received 2026-08-19, ruling 48-L2)

Transcribed from the primary source: Table A.1, *Robust Training under Label Noise by
Over-parameterization*, arXiv:2202.14026v2 / PMLR v162 pp.14153–14172, **Clothing-1M column**.

| quantity | value |
|---|---|
| `alpha_u` (lr for u_i) | **0.1** |
| `alpha_v` (lr for v_i) | **1** |
| init std for {u_i, v_i} | 1e-8 |
| weight decay for {u_i, v_i} | 0 |
| `lambda_C` (consistency) | 0.0 |
| `lambda_B` (class balance) | 0.0 |
| u-update | CE(phi(f + s)) |
| v-update | MSE, per Algorithm 1 |
| projection | u, v projected to [-1, 1] after each step |

`lambda_C = lambda_B = 0` confirms **plain SOP** for Clothing-1M — no SOP+ consistency or
class-balance regularizer — consistent with the plain-SOP resolution already recorded above.

**Attribution status: VERIFIED** against the paper text directly. This is a stronger status
than the CIFAR-N `alpha` pin, which remains a registered choice among options rather than a
transcription, and it therefore carries **no NEEDS-VERIFICATION flag**.

### Disclosure — intra-paper learning-rate discrepancy (required by 48-L2)

The paper's main text §2.2 states an initial learning rate of **0.001** for Clothing-1M; Table
A.1 states **0.002**. The two disagree within the same paper. `tier2_amendment.md` pinned
**0.001**, the main-text value, before this discrepancy was known; the CE runs have already
executed under it (CE seed0 complete, 20/20; CE seed1 running). The pin **stands** for the SOP
arm, so that both learners in Tier 2 share one optimizer configuration and the arm is
internally consistent. The discrepancy is disclosed here rather than silently resolved, and any
report using Tier-2 numbers carries it.

### OPEN — is the [-1, 1] projection general to Algorithm 1, or specific to this column?

**This needs a ruling and I cannot settle it from the repository.**

The projection of `u, v` to `[-1, 1]` after each step appears in this pin. It appears **nowhere
else**: not in this document before today, and not in `src/train/sop.py`, which implements
`clamp(u^2 * label, 0, 1)` inside the loss. That clamp bounds the *loss term* and zeroes the
gradient outside the range; it does not project the *parameter* after the optimizer step. The
two are not equivalent — a parameter can drift outside [-1, 1] under the second and not the
first, and its gradient contribution then stays clipped rather than being pulled back.

The pin phrases the u- and v-updates as "per Algorithm 1", which reads as restating general
algorithm properties for this column rather than naming Clothing-1M-specific behaviour. If the
projection is likewise general to Algorithm 1, then **the twelve Tier-1 SOP runs already
executed without it**, and that is a defect in executed artifacts, not a gap in a pending one.
If it is specific to the Clothing-1M configuration, Tier 1 is unaffected.

Pending that ruling: Tier 2 implements the projection **as pinned**, in the Tier-2 trainer
rather than in `src/train/sop.py`. The shared module is the code that produced the Tier-1 SOP
results and is left byte-identical, so that whichever way the ruling goes, the Tier-1 artifacts
and the module that made them still correspond.
