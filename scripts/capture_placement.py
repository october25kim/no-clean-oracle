"""Capture where the recon runs actually executed.  [EXPLORATORY-UNVERIFIED-PROVENANCE]

Nothing here may enter registered artifacts or be pooled with any registered result.

The four recon runs do not record their physical GPU in their own ``meta.json``. The
``device_uuid`` field was added to ``recon_c1m.py`` after they were launched, and the fields
they *do* carry -- ``device: cuda:0``, ``device_index: 0``, ``device_name: NVIDIA TITAN
RTX`` -- are the container-local view: under ``--gpus "device=N"`` every container sees its
single GPU as index 0, and all four cards are the same model. So the artifacts cannot answer
"which GPU", which is exactly the question D-9 was misread on.

The answer currently exists only in the Docker container objects, which are pruned sooner or
later. This reads it out into a file that survives them.

One trap this refuses to fall into: ``.State.ExitCode`` is ``0`` for a *running* container,
which is indistinguishable from a clean exit if you only read the number. A first hand-run
of this capture recorded ``exit=0`` for a run that was still training. The status is checked
first, and a non-exited container is recorded as running with its exit code withheld rather
than reported as a success it has not yet earned.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exploratory_c1m", "PLACEMENT.json")
CONTAINERS = ("recon_ce_s0", "recon_elr_s0", "recon_ce_s1", "recon_elr_s1")
TAG = "EXPLORATORY-UNVERIFIED-PROVENANCE"


def sh(cmd) -> str:
    return subprocess.run(cmd, text=True, capture_output=True).stdout.strip()


def inspect(name: str, fmt: str) -> str:
    return sh(["docker", "inspect", name, "--format", fmt])


def gpu_index_by_uuid() -> Dict[str, int]:
    out = {}
    for line in sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        out[uuid] = int(idx)
    return out


def main(argv=None) -> int:
    uuid2idx = gpu_index_by_uuid()
    idx2uuid = {v: k for k, v in uuid2idx.items()}
    rows, incomplete = {}, []

    for name in CONTAINERS:
        dr = inspect(name, "{{json .HostConfig.DeviceRequests}}")
        if not dr or dr == "null":
            rows[name] = dict(error="container absent; placement no longer recoverable")
            continue
        did = json.loads(dr)[0]["DeviceIDs"][0]
        if did in uuid2idx:                       # pinned by UUID (wave 1)
            host_idx, host_uuid = uuid2idx[did], did
        else:                                     # pinned by index (wave 2)
            host_idx, host_uuid = int(did), idx2uuid.get(int(did), "unknown")

        status = inspect(name, "{{.State.Status}}")
        row = dict(pinned_as=did, host_gpu_index=host_idx, host_gpu_uuid=host_uuid,
                   shm_size=int(inspect(name, "{{.HostConfig.ShmSize}}")),
                   status=status,
                   started=inspect(name, "{{.State.StartedAt}}"),
                   mounts=inspect(name, "{{range .Mounts}}{{.Source}}(RW={{.RW}}) {{end}}"))
        if status == "exited":
            row["exit_code"] = int(inspect(name, "{{.State.ExitCode}}"))
            row["finished"] = inspect(name, "{{.State.FinishedAt}}")
        else:
            # A running container reports ExitCode 0. Recording that would assert a clean
            # completion the run has not reached.
            row["exit_code"] = None
            row["exit_code_note"] = f"withheld: container is {status}, not exited"
            incomplete.append(name)
        rows[name] = row

    out = dict(
        classification=TAG,
        note=("Placement facts for the four recon runs, read out of the live container "
              "objects because none of the runs records its physical GPU in meta.json: "
              "device_uuid was added to recon_c1m.py after they were launched, and "
              "device_index is the container-local 0 for every run. Wave 1's containers "
              "also show the read-write fedcore2 bind of D-10, which wave 2 does not "
              "carry."),
        complete=not incomplete,
        still_running=incomplete,
        gpu_uuid_to_index=uuid2idx,
        containers=rows)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = f"{OUT}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, OUT)

    for name, r in rows.items():
        if "error" in r:
            print(f"  {name}: {r['error']}")
        else:
            ec = r["exit_code"] if r["exit_code"] is not None else f"({r['status']})"
            print(f"  {name}: host GPU {r['host_gpu_index']} "
                  f"{r['host_gpu_uuid'][:22]}... exit={ec} shm={r['shm_size']}")
    if incomplete:
        print(f"  INCOMPLETE: {', '.join(incomplete)} still running; re-run to finalise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
