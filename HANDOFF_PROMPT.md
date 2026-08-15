# G1 Audit — Continuation Prompt

> Paste this as the opening prompt of a Claude Code session started **inside
> `/data/workspace/sanghoon/g1_audit`**. It is self-contained: it does not depend on
> the Fed-CORE repo's CLAUDE.md. (You may also rename this file `CLAUDE.md` so it
> auto-loads — but keep the execution-order gates: do NOT launch the 48-run sweep
> until the owner approves the JSONL schema and trajectory plot.)

You are continuing **G1**, the Phase-0 gate experiment of the project *"No Clean
Oracle: deployment-robust checkpoint selection for noisy-label learning without clean
validation."* This is a **measurement/audit only**. Do **NOT** implement any
checkpoint-selection method, proxy score, or selector — the only outputs are logged
per-epoch trajectories, oracle epochs, cross-regret statistics, and an automated
go/no-go verdict against pre-registered kill criteria.

**Hypothesis under test:** under noisy-label training, the checkpoints optimal for ID
accuracy, tail-class accuracy, and OOD reliability systematically diverge, and the
regret of using one objective's optimal checkpoint for another is large.

## Ground rules
- **Language:** discuss/diagnose/plan in **Korean**; code, filenames, variable names,
  config keys, commit messages, and final deliverables in **English**.
- **Isolation:** this project lives entirely under `/data/workspace/sanghoon/g1_audit`.
  **Never modify** the Fed-CORE repo at `/data/workspace/sanghoon/fedcore2`; you may
  only **READ** its CIFAR data at `/data/workspace/sanghoon/fedcore2/data`.
- **Fed-CORE coexistence:** Fed-CORE is running concurrently on all 4 GPUs (its
  campaigns + a detached priority orchestrator, `pgrep -f medmnist_to_cifar_merge.sh`,
  PID was 1301882). **Co-locate** G1 on spare VRAM; do **NOT** stop Fed-CORE and do
  **NOT** touch its `cifsw_*/medm_*/cifpx_*/tiss_*` containers. GPU physical UUIDs:
  `0=GPU-d6e53d0c…`, `1=GPU-94b3a414…`, `2=GPU-c3326fea…`, `3=GPU-afbc9e02…`.
- Surface any forced deviation and propose the minimal substitution; never silently
  change the design.

## Report format (owner preference)
진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동.

## Current status (as of this handoff)
- **Step 1 DONE (CPU):** scaffold + deterministic seeded noise injection + analysis
  core (oracle epochs, cross-regret CR, normalized NCR, BCa bootstrap, verbatim
  PASS/WEAK/KILL verdict) + OOD-score/tail utils. **24 unit tests pass**
  (`python -m pytest tests/ -q`). Noise gate PASS (`python scripts/make_noise_report.py`
  — empirical flip rates within 0.01 of config on real CIFAR; CIFAR-10 asymmetric
  overall = 0.20 by design since only 5/10 classes are flip sources, per-source = 0.40).
  Noisy-label masks for seed 0 saved under `results/noise_masks/`; seeds 1,2 are
  auto-generated on demand by `load_or_make_noisy_labels`.
- **OOD pools FROZEN** under `results/ood_pools/` (never regenerate; `--force` exists
  but must not be used — G2+ reuse these exact files):
  - `ood_pools_cifar10.npz`  sha256 `e8647d06…`
  - `ood_pools_cifar100.npz` sha256 `97308d5f…`
  - each: `svhn (26032)`, `cross_cifar (10000)`, `CIFAR-C-local (8000)`.
- **Step 2 code built + compiles:** `src/models/preact_resnet.py`,
  `src/eval/evaluate.py`, `src/train/trainer.py` (CE done; ELR pluggable),
  `scripts/run_single.py`, `scripts/run_sweep.py` (4-GPU scheduler),
  `scripts/build_ood_pools.py`.
- **Smoke IN PROGRESS — VERIFY FIRST.** A 20-epoch smoke was launched **co-located on
  physical GPU 3**: container `g1smoke_c10_sym04_ce_s0`, run
  `cifar10 symmetric 0.4 CE seed0`, `--co-tenant fedcore2`, workers 6. Check it before
  anything else:
  ```bash
  docker ps -a --filter name=g1smoke_ --format '{{.Names}} {{.Status}}'
  wc -l results/runs/cifar10_symmetric0.4_ce_seed0/metrics.jsonl 2>/dev/null   # 1 meta + N epochs
  docker logs g1smoke_c10_sym04_ce_s0 2>&1 | tail -20
  ```
  If it crashed or is incomplete, resume/re-run it (same command below). Non-fatal
  cudnn "CUDNN_STATUS_NOT_SUPPORTED / Plan failed" warnings are expected under
  deterministic mode (`warn_only=True`) and do not stop training.

## Sealed decisions (binding — from the owner)
1. **CIFAR-C-local (covariate OOD pool):** the official Zenodo CIFAR-10-C/100-C
   (~2.9 GB each) is substituted by a fixed **4-corruption × 2,000 subsample at
   severity 3** generated with the **official `imagecorruptions` library** on a seeded
   local subsample. It must **NEVER** be called or reported as official
   CIFAR-10-C/100-C, and its numbers are **not comparable** to official CIFAR-C. The
   name is **`CIFAR-C-local`** everywhere (code, JSONL, report, plots). Library +
   params are pinned (`requirements.txt`: `imagecorruptions==1.1.2`,
   `opencv-python-headless==4.10.0.84`) and recorded in each npz's `__meta__` +
   `results/ood_pools/provenance_*.json`. Corruptions
   `[gaussian_noise, defocus_blur, brightness, contrast]`, subsample seed `20260811`.
   Any report using this pool must include one sentence stating the substitution and
   non-comparability.
2. **JSONL schema (owner checklist):** log **raw per-epoch values only — NO
   pre-smoothed metrics** (the 3-epoch moving average is applied only in analysis when
   finding oracle epochs). `effective_overall_noise` must be present in the run
   metadata; the OOD subsample seed and full pool provenance must be in the metadata.
3. **Sweep is gated:** do **NOT** launch any of the 48 sweep runs until the owner
   approves the JSONL schema and one trajectory plot.
4. **Sweep parallelization (owner spec):** process-level, one worker per GPU, one run
   per worker via `CUDA_VISIBLE_DEVICES`; first-free-GPU dispatch (48 runs → ~12/GPU);
   each run single-GPU and internally identical to serial (seeds/determinism/logging
   unchanged); crash-tolerant (a failed run retried **once** on the next free GPU,
   resuming from its last checkpoint; sweep continues); log run→GPU + start/end
   timestamps; dataloader `num_workers` capped at `cores // 4`; OOD pools kept resident
   in memory per worker. This is implemented in `scripts/run_sweep.py`.
5. Smoke/any co-located run records `co_tenant: fedcore2` in metadata so wall-clock is
   interpreted correctly.

## Repo layout
```
configs/base.yaml        all hyperparameters (arch preact_resnet18, SGD lr0.02 cosine,
                         120 epochs, ELR λ/β, eval settings, seeds, noise configs)
src/data/noise.py        deterministic transition-matrix noise + mask save/load
src/data/datasets.py     CIFAR load (local), noisy-label train ds, OOD source datasets
src/data/ood_pools.py    frozen OOD-pool loader + CIFAR-C-local constants
src/models/preact_resnet.py   PreAct ResNet-18
src/eval/ood.py          MSP, energy, AUROC (validated vs sklearn)
src/eval/tail.py         static/dynamic tail-class construction
src/eval/evaluate.py     per-epoch R_ID / R_tail / R_OOD -> ONE JSONL record
src/train/trainer.py     shared trainer: CE done; ELR via src/train/elr.py (TODO);
                         per-epoch eval, JSONL, checkpoint every 10 + final, resume
src/analysis/ncr.py      oracle epochs, CR, NCR, BCa bootstrap, verdict (verbatim rule)
scripts/run_single.py    one run (dataset×noise×learner×seed)
scripts/run_sweep.py     48-run 4-GPU scheduler (owner spec)
scripts/build_ood_pools.py   ONE-TIME frozen OOD-pool builder (already run)
scripts/make_noise_report.py step-1 noise gate report
tests/                   test_noise/test_tail/test_auroc/test_ncr (24 pass)
results/noise_masks/     seed-0 masks; results/ood_pools/ frozen pools; results/runs/ JSONL
```

## Environment
- **Training image:** `fedcore-c400r:latest` (torch 2.3.0, torchvision 0.18.0). Run
  containers with `--gpus device=<physical GPU UUID>` and
  `-e CUBLAS_WORKSPACE_CONFIG=:4096:8` (required for deterministic CuBLAS), mounting
  both `/data/workspace/sanghoon/g1_audit` and `/data/workspace/sanghoon/fedcore2/data`
  at their real paths. Analysis stage runs on the **host** (numpy/scipy/sklearn/
  matplotlib/pyyaml — no GPU).
- **Two container flags are MANDATORY** (the first smoke died without them):
  `--shm-size=16g` — docker's default 64 MB `/dev/shm` is exhausted by the train
  dataloader's worker processes (`Bus error … out of shared memory`); and
  `--user 1000:1000` — otherwise every file under `results/` is written as root and
  the host-side analysis stage cannot manage them.
- **OOD-pool rebuild gotcha (only if ever forced; pools are FROZEN):** `imagecorruptions`
  pulls in non-headless `opencv-python` whose `cv2` fails with `libxcb.so.1`. Fix:
  `pip install imagecorruptions==1.1.2 && pip uninstall -y opencv-python &&
  pip install opencv-python-headless==4.10.0.84`.
- **Smoke launch command (co-located GPU 3):**
  ```bash
  docker run -d --name g1smoke_c10_sym04_ce_s0 \
    --gpus device=GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f \
    --user 1000:1000 --shm-size=16g \
    -v /data/workspace/sanghoon/g1_audit:/data/workspace/sanghoon/g1_audit \
    -v /data/workspace/sanghoon/fedcore2/data:/data/workspace/sanghoon/fedcore2/data \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e G1_DATALOADER_WORKERS=6 \
    fedcore-c400r:latest \
    python /data/workspace/sanghoon/g1_audit/scripts/run_single.py \
      --config /data/workspace/sanghoon/g1_audit/configs/base.yaml \
      --dataset cifar10 --noise-type symmetric --eta 0.4 --learner ce --seed 0 \
      --assigned-gpu 3 --co-tenant fedcore2 --epochs 20
  ```

## Remaining execution order (do in this order; STOP at the gate)
1. **Finish/verify the smoke.** Then produce, for the owner: (i) the **JSONL schema**
   (one metadata line + one per-epoch line; enumerate every field, confirm raw values
   only + `effective_overall_noise` + OOD seed/provenance present); (ii) **one
   trajectory plot** (R_ID, R_tail, R_OOD vs epoch with oracle-epoch markers) — write a
   small host-side plotting script under `src/analysis/` or `scripts/`; (iii) the
   **measured per-epoch wall-clock under co-location**; (iv) a **projected full-sweep
   completion time for two scenarios — A: 4-GPU co-located now; B: 4-GPU exclusive
   after Fed-CORE finishes** (state the assumed Fed-CORE remaining time; if unknown, ask
   the owner). **PAUSE for approval — no sweep before approval.**
2. After approval: implement **ELR** at `src/train/elr.py` (Liu et al., NeurIPS 2020;
   per-sample target-probability EMA regularizer; `λ=3, β=0.7` for CIFAR-10 and
   `λ=7, β=0.9` for CIFAR-100 per `configs/base.yaml learners.elr`; expose `.state()`
   /`.load_state()` for resume). Confirm CE and ELR consume the **identical** noisy
   masks per cell.
3. Launch the **48-run sweep** via `scripts/run_sweep.py --gpus …` per the owner's A/B
   scheduling decision.
4. Implement analysis/report: `scripts/make_report.py` reading only the JSONL logs
   (no GPU) → per-cell NCR tables (mean ± BCa 95% CI), NCR heatmaps, trajectory plots,
   conflict-severity summary, and the **verdict** (PASS/WEAK/KILL — the rule is already
   implemented verbatim in `src/analysis/ncr.py::verdict`; do not soften it). The
   static tail set (primary R_tail) is derived post-hoc in analysis from the logged
   `per_class_test_error` of the CE seed-0 final epoch, per (dataset×noise) cell.

## Quick commands
```bash
python -m pytest tests/ -q                 # 24 tests
python scripts/make_noise_report.py        # step-1 noise gate
cat results/ood_pools/provenance_cifar10.json
```
