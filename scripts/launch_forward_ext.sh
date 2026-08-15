#!/usr/bin/env bash
# Tier 1 forward pass, sharded one container per GPU: 36 runs x 24 = 864 checkpoints.
#
# Same frozen container flags as every other G1/B2 launch, because the outputs must sit
# in the same numerical frame as the trajectories they are sampled from. Inference only:
# no training, no checkpoint is written or modified.
#
# The launcher refuses on a dirty tree (R2) and injects the host's git state, exactly as
# launch_ext_tier1.sh does -- the training image has no git binary, so without the
# injection code_stamp would record git_head null and git_tree_dirty false, which reads
# as a clean tree but means "git could not be asked".
#
# Deliberately no `docker rm -f`: a second invocation must fail on the name conflict and
# leave a running pass alone rather than killing it mid-checkpoint.
set -euo pipefail

REPO=/data/workspace/sanghoon/g1_audit
DATA=/data/workspace/sanghoon/fedcore2/data
IMAGE=fedcore-c400r:latest

cd "$REPO"
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "R2 violation: working tree is dirty; commit before launching." >&2
  echo "$DIRTY" >&2
  exit 1
fi
GIT_HEAD=$(git rev-parse HEAD)

python3 "$REPO/scripts/check_data_hygiene.py"

GPU0=GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb
GPU1=GPU-94b3a414-8e7a-3454-83dc-6132a9124a28
GPU2=GPU-c3326fea-a08f-9192-eee8-27fb8017984c
GPU3=GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f
GPUS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")

# 36 runs, 9 per shard. The c100n runs carry 100-dim logits and are the heavier half of
# the write volume, so each shard gets exactly 3 of them and 6 CIFAR-10 runs; the four
# shards then finish within a few minutes of each other instead of one trailing by hours.
SHARD0="c100n_ce_seed0,c100n_ce_seed1,c100n_ce_seed2,c10n_worst_ce_seed0,c10n_worst_ce_seed1,c10n_worst_ce_seed2,c10n_random1_ce_seed0,c10n_random1_ce_seed1,c10n_random1_ce_seed2"
SHARD1="c100n_elr_seed0,c100n_elr_seed1,c100n_elr_seed2,c10n_worst_elr_seed0,c10n_worst_elr_seed1,c10n_worst_elr_seed2,c10n_random1_elr_seed0,c10n_random1_elr_seed1,c10n_random1_elr_seed2"
SHARD2="c100n_gce_seed0,c100n_gce_seed1,c100n_gce_seed2,c10n_worst_gce_seed0,c10n_worst_gce_seed1,c10n_worst_gce_seed2,c10n_random1_gce_seed0,c10n_random1_gce_seed1,c10n_random1_gce_seed2"
SHARD3="c100n_sop_seed0,c100n_sop_seed1,c100n_sop_seed2,c10n_worst_sop_seed0,c10n_worst_sop_seed1,c10n_worst_sop_seed2,c10n_random1_sop_seed0,c10n_random1_sop_seed1,c10n_random1_sop_seed2"
SHARDS=("$SHARD0" "$SHARD1" "$SHARD2" "$SHARD3")

for i in 0 1 2 3; do
  name="g1fwdext$i"
  docker run -d --name "$name" \
    --gpus "device=${GPUS[$i]}" \
    --user 1000:1000 --shm-size=16g \
    -v "$REPO:$REPO" -v "$DATA:$DATA" \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e G1_GIT_HEAD="$GIT_HEAD" \
    -e G1_GIT_DIRTY_PATHS="" \
    -w "$REPO" \
    "$IMAGE" \
    python "$REPO/scripts/forward_ext.py" --only "${SHARDS[$i]}" >/dev/null
  echo "launched $name on GPU $i (9 runs)"
done

echo
echo "launched at HEAD ${GIT_HEAD:0:7}"
echo "  docker logs -f g1fwdext0"
echo "  ls -d $REPO/results/forward_ext/*/ep*/ | wc -l          # target 864"
echo "  ls $REPO/results/forward_ext/ABORT_shard_*.json 2>/dev/null  # must stay empty"
