#!/usr/bin/env bash
# Canonical launcher for the G1 48-run sweep (owner-approved scenario A:
# 4 GPUs, co-located with Fed-CORE). Every container flag the owner froze lives
# here so no run can inherit a different environment:
#
#   --shm-size=16g   docker's 64 MB default /dev/shm kills the train dataloader
#                    workers with a Bus error minutes into a run
#   --user 1000:1000 otherwise results/ is written as root and the host-side
#                    analysis stage cannot manage it
#   -e CUBLAS_WORKSPACE_CONFIG=:4096:8   required for deterministic CuBLAS
#   -e CUDA_DEVICE_ORDER=PCI_BUS_ID      makes CUDA_VISIBLE_DEVICES=k select the
#                    same GPU that nvidia-smi lists at index k, so the gpu_uuid
#                    recorded in each run's metadata is the true physical device
#
# run_sweep.py re-checks the first two at startup and refuses to run without them.
# NEVER stop or disturb the Fed-CORE containers (cifsw_/cifpx_/medm_/tiss_).
set -euo pipefail

REPO=/data/workspace/sanghoon/g1_audit
DATA=/data/workspace/sanghoon/fedcore2/data
IMAGE=fedcore-c400r:latest
NAME=${NAME:-g1sweep}
CONFIG=${CONFIG:-$REPO/configs/base.yaml}   # CONFIG=configs/b2.yaml for the B2 rerun

GPU0=GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb
GPU1=GPU-94b3a414-8e7a-3454-83dc-6132a9124a28
GPU2=GPU-c3326fea-a08f-9192-eee8-27fb8017984c
GPU3=GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f

docker run -d --name "$NAME" \
  --gpus "\"device=$GPU0,$GPU1,$GPU2,$GPU3\"" \
  --user 1000:1000 --shm-size=16g \
  -v "$REPO:$REPO" -v "$DATA:$DATA" \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$IMAGE" \
  python "$REPO/scripts/run_sweep.py" \
    --config "$CONFIG" \
    --gpus 0,1,2,3 \
    --co-tenant fedcore2 \
    "$@"

echo "launched $NAME; follow with:"
echo "  docker logs -f $NAME"
echo "  tail -f $REPO/results/runs/sweep_log.jsonl"
