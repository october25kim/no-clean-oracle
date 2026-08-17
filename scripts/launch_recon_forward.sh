#!/usr/bin/env bash
# Recon forward pass over all 80 checkpoints.  [EXPLORATORY-UNVERIFIED-PROVENANCE]
#
# Nothing this produces may enter registered artifacts, paper claims, or be pooled with any
# registered result.  Outputs live under results/exploratory_c1m/forward/ only.
#
# Two processes, one per GPU, split by learner so each GPU carries one CE and one ELR run's
# worth of checkpoints:
#
#   host GPU 1 <- c1m_ce_seed0,  c1m_ce_seed1
#   host GPU 3 <- c1m_elr_seed0, c1m_elr_seed1
#
# Same placement the training used, so nothing new is introduced on the coexistence side.
# Fed-CORE is never stopped and its containers are never touched.
#
# No --shm-size here, unlike the training launcher: recon_forward.py reads numpy slices
# directly and uses no DataLoader workers, so there is no shared-memory consumer to size.
# No fedcore2 mount either, for the reason recorded as D-10 -- the pass does not need that
# tree, so it does not get access to it.
set -euo pipefail

ROOT=/data/workspace/sanghoon/g1_audit
IMAGE=fedcore-c400r:latest

cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "R2: internal tree is dirty; commit before launching" >&2
  git status --porcelain >&2
  exit 1
fi
echo "[forward] internal HEAD $(git rev-parse HEAD)"

python "$ROOT/scripts/check_data_hygiene.py" --allow-unmanifested >/dev/null
echo "[forward] data hygiene OK"

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/         /'
echo "[forward] Fed-CORE containers (untouched): $(docker ps --format '{{.Names}}' \
  | grep -cE '^(cifsw|medm|cifpx|tiss)_')"

launch () {
  local name=$1 gpu=$2; shift 2
  if [[ -n "$(docker ps -aq -f name="^${name}$")" ]]; then
    docker rm -f "$name" >/dev/null
  fi
  docker run -d --name "$name" --gpus "\"device=${gpu}\"" \
    -v "$ROOT:$ROOT" -w "$ROOT" \
    -e TORCH_HOME="$ROOT/results/exploratory_c1m/torch_home" \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python "$ROOT/scripts/recon_forward.py" --gpu 0 --runs "$@" >/dev/null
  echo "[forward] launched ${name} on host GPU ${gpu}: $*"
}

launch fwd_ce  1 c1m_ce_seed0  c1m_ce_seed1
launch fwd_elr 3 c1m_elr_seed0 c1m_elr_seed1

sleep 15
docker ps --format '  {{.Names}}\t{{.Status}}' | grep fwd_ || true
