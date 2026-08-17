#!/usr/bin/env bash
# Recon wave 2: {CE seed1, ELR seed1}.  [EXPLORATORY-UNVERIFIED-PROVENANCE]
#
# Nothing this produces may enter registered artifacts, paper claims, or be pooled with
# any registered result.  Outputs live under results/exploratory_c1m/ only.
#
# Two corrections to the wave-1 launch are encoded here rather than remembered:
#
#   1. Explicit device pins -- instrumentation, NOT a correction.  Wave 1 was already pinned
#      one GPU per container by UUID (CE on GPU 1, ELR on GPU 3), exactly as approved; the
#      claim that it ran both on GPU 0 was wrong and is withdrawn as D-9.  What wave 1 could
#      not do was say so from its own artifacts, which is why the misreading was possible.
#      Each container still sees exactly one GPU, and the run now records which one.
#
#   2. No fedcore2 mount.  Wave 1 inherited /data/workspace/sanghoon/fedcore2/data as a
#      READ-WRITE bind from the registered-track template.  recon_c1m.py contains no
#      reference to fedcore2, so nothing was ever written -- verified: no file under that
#      tree has an mtime after 2026-08-11 -- but "the script happens not to write there" is
#      not the same guard as "the script cannot write there".  The recon does not need the
#      mount at all, so it does not get it.
#
#   3. --shm-size=16g.  Wave 1 had it; this launcher initially did not, so the containers
#      fell back to Docker's 64 MB default and every DataLoader worker died of a bus error
#      partway through the first epoch.  DataLoader shared memory is a runtime flag with no
#      trace in the script, which is exactly the kind of setting that goes missing when a
#      launch is retyped rather than reused, so it is asserted below against wave 1's own
#      recorded container config instead of being carried in someone's head.
#
# Fed-CORE is never stopped and its containers are never touched.  We co-locate on spare
# VRAM only; if a GPU cannot take the run, halve the batch rather than evicting anything.
set -euo pipefail

ROOT=/data/workspace/sanghoon/g1_audit
IMAGE=fedcore-c400r:latest
BATCH="${BATCH:-256}"

cd "$ROOT"

# R2: the tree that produces artifacts must be clean and attested.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "R2: internal tree is dirty; commit before launching" >&2
  git status --porcelain >&2
  exit 1
fi
HEAD_SHA=$(git rev-parse HEAD)

python "$ROOT/scripts/check_data_hygiene.py" --allow-unmanifested

echo "[wave2] internal HEAD $HEAD_SHA"
echo "[wave2] pre-flight GPU state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/         /'
echo "[wave2] Fed-CORE containers (untouched): $(docker ps --format '{{.Names}}' \
  | grep -cE '^(cifsw|medm|cifpx|tiss)_')"

launch () {
  local learner=$1 seed=$2 gpu=$3 name="recon_${1}_s${2}"
  if [[ -n "$(docker ps -aq -f name="^${name}$")" ]]; then
    echo "[wave2] removing exited container ${name}" && docker rm -f "$name" >/dev/null
  fi
  docker run -d --name "$name" --gpus "\"device=${gpu}\"" \
    --shm-size=16g \
    -v "$ROOT:$ROOT" -w "$ROOT" \
    -e TORCH_HOME="$ROOT/results/exploratory_c1m/torch_home" \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python "$ROOT/scripts/recon_c1m.py" \
      --learner "$learner" --seed "$seed" --batch "$BATCH" --gpu 0 >/dev/null
  echo "[wave2] launched ${name} on host GPU ${gpu} (container-local cuda:0), batch ${BATCH}"

  # Runtime-parity gate: wave 1's containers are still on disk (exited, not removed), so
  # the settings that shaped them are readable rather than remembered.  Compare the ones
  # that silently change behaviour when dropped.
  local ref=recon_ce_s0
  if [[ -n "$(docker ps -aq -f name="^${ref}$")" ]]; then
    local want got
    want=$(docker inspect "$ref"  --format '{{.HostConfig.ShmSize}}')
    got=$(docker inspect "$name" --format '{{.HostConfig.ShmSize}}')
    if [[ "$want" != "$got" ]]; then
      echo "[wave2] shm parity FAILED: wave 1 ${want} vs ${name} ${got}" >&2
      docker rm -f "$name" >/dev/null; exit 1
    fi
    echo "[wave2]   shm parity with wave 1 OK (${got} bytes)"
  fi
}

# --gpus "device=N" exposes exactly one GPU, which the container then sees as index 0.
# The host index is what the operator reasons about; the container-local index is what
# torch sees.  Both are printed so the two never get confused in a report.
launch ce  1 1
launch elr 1 3

sleep 20
docker ps --format '  {{.Names}}\t{{.Status}}' | grep recon || true
