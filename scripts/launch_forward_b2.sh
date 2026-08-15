#!/usr/bin/env bash
# Approved B2 forward pass, sharded one container per GPU.
#
# Same frozen container flags as every other G1 launch (--shm-size, --user, the
# determinism env), because the outputs must sit in the same numerical frame as the
# B2 trajectories they are sampled from. Inference only: no training, no checkpoint is
# written or modified.
set -euo pipefail

REPO=/data/workspace/sanghoon/g1_audit
DATA=/data/workspace/sanghoon/fedcore2/data
IMAGE=fedcore-c400r:latest

GPU0=GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb
GPU1=GPU-94b3a414-8e7a-3454-83dc-6132a9124a28
GPU2=GPU-c3326fea-a08f-9192-eee8-27fb8017984c
GPU3=GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f
GPUS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")

# 15 runs split 4/4/4/3, cifar100 runs (heavier logits) spread across shards
SHARD0="cifar100_symmetric0.4_ce_seed0,cifar100_symmetric0.4_ce_seed1,cifar10_asymmetric0.4_ce_seed0,cifar10_symmetric0.2_elr_seed0"
SHARD1="cifar100_symmetric0.4_ce_seed2,cifar100_symmetric0.6_ce_seed0,cifar10_asymmetric0.4_ce_seed1,cifar10_symmetric0.2_elr_seed1"
SHARD2="cifar100_symmetric0.6_ce_seed1,cifar100_symmetric0.6_ce_seed2,cifar10_asymmetric0.4_ce_seed2,cifar10_symmetric0.2_elr_seed2"
SHARD3="cifar10_asymmetric0.4_elr_seed0,cifar10_asymmetric0.4_elr_seed1,cifar10_asymmetric0.4_elr_seed2"
SHARDS=("$SHARD0" "$SHARD1" "$SHARD2" "$SHARD3")

for i in 0 1 2 3; do
  name="g1fwd$i"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" \
    --gpus "device=${GPUS[$i]}" \
    --user 1000:1000 --shm-size=16g \
    -v "$REPO:$REPO" -v "$DATA:$DATA" \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -w "$REPO" \
    "$IMAGE" \
    python "$REPO/scripts/forward_b2.py" --only "${SHARDS[$i]}" >/dev/null
  echo "launched $name on GPU $i: ${SHARDS[$i]}"
done

echo
echo "follow with:  docker logs -f g1fwd0"
echo "count:        ls -d $REPO/results/forward_b2/*/ep*/ | wc -l   # target 360"
