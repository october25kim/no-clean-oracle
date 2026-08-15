#!/usr/bin/env bash
# Tier 1 of the real-noise extension: CIFAR-N x {CE, ELR, SOP, GCE} x 3 seeds.
#
# Same frozen container flags as every other campaign, plus one addition: the training
# image has no git binary, so the host's git state is captured HERE and injected as
# G1_GIT_HEAD / G1_GIT_DIRTY_PATHS. Without it code_stamp records git_head null and
# git_tree_dirty false, which reads as a clean tree but means "git could not be asked".
#
# The launcher refuses to start on a dirty tree: R2 requires the code producing a
# registered output to be committed first, and an injected stamp can only be as honest
# as the tree it was taken from.
#
# Campaign attribution: gate run git_head = 2f5b122; remaining 35 runs launch from a
# later HEAD. Attribution rests on the modules axis per the dual-axis stamp design.
#
# CORRECTION (post-launch, observed): modules_combined_sha256 is NOT identical across
# all 36. code_stamp() digests the modules the process actually imported, and build_loss
# imports the learner lazily inside its branch, so a CE run stamps 16 modules and an ELR
# run stamps 17 (it adds src/train/elr.py). The digest is therefore constant WITHIN a
# learner and differs BETWEEN learners, by design of the stamp rather than by drift.
# The invariant that does hold campaign-wide, and the one attribution should be checked
# against: every module present in more than one run has the same sha256 in all of them.
set -euo pipefail

REPO=/data/workspace/sanghoon/g1_audit
DATA=/data/workspace/sanghoon/fedcore2/data
IMAGE=fedcore-c400r:latest
NAME=${NAME:-g1ext}
CONFIG=${CONFIG:-$REPO/configs/ext_tier1.yaml}

cd "$REPO"
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "R2 violation: working tree is dirty; commit before launching." >&2
  echo "$DIRTY" >&2
  exit 1
fi
GIT_HEAD=$(git rev-parse HEAD)

# data/ must hold only files the seal accounts for. Nothing is deleted; an unaccounted
# file stops the launch until someone has looked at it and said so explicitly.
HYGIENE_ARGS=()
if [ "${ALLOW_UNMANIFESTED:-0}" = "1" ]; then HYGIENE_ARGS+=(--allow-unmanifested); fi
python3 "$REPO/scripts/check_data_hygiene.py" "${HYGIENE_ARGS[@]+"${HYGIENE_ARGS[@]}"}"

GPU0=GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb
GPU1=GPU-94b3a414-8e7a-3454-83dc-6132a9124a28
GPU2=GPU-c3326fea-a08f-9192-eee8-27fb8017984c
GPU3=GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f

# Deliberately no `docker rm -f "$NAME"`: launch_sweep.sh omits it for a reason. A second
# invocation must fail on the name conflict and leave the running sweep alone, rather
# than force-killing it mid-epoch.
docker run -d --name "$NAME" \
  --gpus "\"device=$GPU0,$GPU1,$GPU2,$GPU3\"" \
  --user 1000:1000 --shm-size=16g \
  -v "$REPO:$REPO" -v "$DATA:$DATA" \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e G1_GIT_HEAD="$GIT_HEAD" \
  -e G1_GIT_DIRTY_PATHS="" \
  -w "$REPO" \
  "$IMAGE" \
  python "$REPO/scripts/run_sweep.py" \
    --config "$CONFIG" \
    --gpus 0,1,2,3 \
    --co-tenant fedcore2 \
    "$@"

echo "launched $NAME at HEAD ${GIT_HEAD:0:7}"
echo "  docker logs -f $NAME"
echo "  ls -d $REPO/results/runs_ext/*/TERMINAL.json | wc -l   # target 36"
