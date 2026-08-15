"""G1 48-run sweep: process-level, one-worker-per-GPU scheduler (NOT DDP).

Design (owner spec 2026-08-11):
  - one worker thread per GPU; one run per worker at a time via CUDA_VISIBLE_DEVICES.
  - first-free-GPU dispatch from a shared queue (48 runs -> ~12 per GPU).
  - each run is single-GPU and internally IDENTICAL to the serial design (seeds,
    determinism, logging unchanged); the GPU is just pinned by env.
  - crash-tolerant: a failed run is retried ONCE on the next free GPU (resuming from
    its last checkpoint); the sweep continues regardless.
  - the run->GPU assignment and start/end timestamps are logged both into each run's
    JSONL metadata (by run_single) and into results/sweep_log.jsonl here.
  - dataloader num_workers per run is capped at (total_cores // 4).

Does NOT launch anything on import; call ``main``. Runs already having a final
checkpoint + terminal marker are skipped (resumable sweep).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SINGLE = os.path.join(ROOT, "scripts", "run_single.py")
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import load_config   # noqa: E402


@dataclass
class RunSpec:
    """One run. Either synthetic (dataset+noise_type+eta) or a sealed real-noise split.

    ``split`` is empty for synthetic runs and an EXACT sealed split name otherwise; the
    two are never mixed, and run_id/cell are derived from whichever applies.
    """
    dataset: str
    noise_type: str
    eta: float
    learner: str
    seed: int
    split: str = ""

    @property
    def run_id(self) -> str:
        if self.split:
            return f"{self.split}_{self.learner}_seed{self.seed}"
        return f"{self.dataset}_{self.noise_type}{self.eta:g}_{self.learner}_seed{self.seed}"

    @property
    def cell(self) -> str:
        """The analysis cell, named as the report names it, so --cells takes exactly
        the strings the report prints."""
        if self.split:
            return f"{self.split}_{self.learner}"
        return f"{self.dataset}_{self.noise_type}_{self.eta:g}_{self.learner}"


def enumerate_runs(cfg: dict) -> List[RunSpec]:
    """Real-noise grid when the config declares ``splits``, synthetic grid otherwise."""
    runs = []
    if cfg.get("splits"):
        spec = cfg["real_noise_splits"]
        for split in cfg["splits"]:
            if split not in spec:            # exact match; no substring fallback
                raise SystemExit(f"config lists split {split!r} with no entry in "
                                 f"real_noise_splits (have {sorted(spec)})")
            for learner in cfg["learners"]:
                for seed in cfg["seeds"]:
                    runs.append(RunSpec(spec[split]["dataset"], "", float("nan"),
                                        learner, seed, split=split))
        return runs
    for dataset in cfg["datasets"]:
        for nc in cfg["noise_configs"]:
            for learner in cfg["learners"]:
                for seed in cfg["seeds"]:
                    runs.append(RunSpec(dataset, nc["type"], float(nc["eta"]), learner, seed))
    return runs


def is_done(results_dir: str, run_id: str, epochs: Optional[int] = None) -> bool:
    """Done means: finished, and finished the CONFIGURED number of epochs.

    A ``--epochs N`` smoke writes the same run_id and its own TERMINAL.json, so a bare
    existence check would let the sweep adopt a 20-epoch smoke as a finished 120-epoch
    run and skip it silently.
    """
    path = os.path.join(results_dir, run_id, "TERMINAL.json")
    if not os.path.isfile(path):
        return False
    if epochs is None:
        return True
    try:
        with open(path) as fh:
            terminal = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return (not terminal.get("smoke")) and int(terminal.get("epochs", -1)) == int(epochs)


MIN_SHM_BYTES = 8 * 1024 ** 3      # the 64 MB docker default killed the first smoke


def _shm_bytes() -> int:
    st = os.statvfs("/dev/shm")
    return st.f_blocks * st.f_frsize


def preflight() -> dict:
    """Refuse to start unless the container flags the owner froze are actually in
    effect. A missing --shm-size only shows up as a dataloader Bus error minutes into
    the first run, and a missing --user leaves 48 runs' output owned by root."""
    shm = _shm_bytes()
    if shm < MIN_SHM_BYTES:
        raise SystemExit(
            f"/dev/shm is {shm / 1024 ** 3:.2f} GiB, need >= {MIN_SHM_BYTES / 1024 ** 3:.0f} GiB. "
            f"Launch the container with --shm-size=16g (see scripts/launch_sweep.sh).")
    if os.geteuid() == 0:
        raise SystemExit("running as root would leave results/ unmanageable from the host; "
                         "launch the container with --user 1000:1000 "
                         "(see scripts/launch_sweep.sh).")
    return dict(shm_bytes=shm, uid=os.geteuid(),
                cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER", ""),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""))


@dataclass
class Sweep:
    cfg_path: str
    gpus: List[int]
    results_dir: str
    max_retries: int = 1
    co_tenant: str = ""
    cells: List[str] = field(default_factory=list)
    log_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.cfg = load_config(self.cfg_path)
        self.results_dir = os.path.join(ROOT, self.cfg["results_dir"])
        os.makedirs(self.results_dir, exist_ok=True)
        sealed = os.path.join(self.results_dir, "SEALED")
        if os.path.exists(sealed):
            raise SystemExit(
                f"{self.results_dir} is SEALED ({open(sealed).read().strip()}). A sweep "
                f"pointed at a certified artifact tree would add runs to it and "
                f"invalidate everything derived from it. Refusing to start.")
        self.sweep_log = os.path.join(self.results_dir, "sweep_log.jsonl")
        self.num_workers = max(1, (os.cpu_count() or 4) // 4)
        self.epochs = int(self.cfg["train"]["epochs"])

    def _log(self, rec: dict) -> None:
        with self.log_lock:
            with open(self.sweep_log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _launch(self, spec: RunSpec, gpu: int, resume: bool) -> int:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)          # pin ONE physical GPU
        env["G1_DATALOADER_WORKERS"] = str(self.num_workers)
        cmd = [sys.executable, RUN_SINGLE, "--config", self.cfg_path,
               "--learner", spec.learner, "--seed", str(spec.seed),
               "--assigned-gpu", str(gpu)]
        if spec.split:
            cmd += ["--split", spec.split]
        else:
            cmd += ["--dataset", spec.dataset, "--noise-type", spec.noise_type,
                    "--eta", str(spec.eta)]
        if self.co_tenant:
            cmd += ["--co-tenant", self.co_tenant]
        if resume:
            cmd.append("--resume")
        run_dir = os.path.join(self.results_dir, spec.run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "worker.out.log"), "a") as out:
            return subprocess.call(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)

    def _worker(self, gpu: int, q: "queue.Queue[RunSpec]") -> None:
        while True:
            try:
                spec = q.get_nowait()
            except queue.Empty:
                return
            if is_done(self.results_dir, spec.run_id, self.epochs):
                self._log(dict(event="skip_done", run=spec.run_id, gpu=gpu, ts=time.time()))
                q.task_done(); continue
            attempt = 0
            while attempt <= self.max_retries:
                t0 = time.time()
                self._log(dict(event="start", run=spec.run_id, gpu=gpu, attempt=attempt, ts=t0))
                rc = self._launch(spec, gpu, resume=(attempt > 0))
                t1 = time.time()
                ok = (rc == 0) and is_done(self.results_dir, spec.run_id, self.epochs)
                self._log(dict(event="end", run=spec.run_id, gpu=gpu, attempt=attempt,
                               rc=rc, ok=ok, seconds=round(t1 - t0, 1), ts=t1))
                if ok:
                    break
                attempt += 1
                if attempt <= self.max_retries:
                    self._log(dict(event="retry", run=spec.run_id, gpu=gpu, ts=time.time()))
            q.task_done()

    def run(self) -> int:
        runs = enumerate_runs(self.cfg)
        if self.cells:
            wanted = set(self.cells)
            unknown = wanted - {r.cell for r in runs}
            if unknown:
                raise SystemExit(f"unknown cell(s): {sorted(unknown)}")
            runs = [r for r in runs if r.cell in wanted]
        q: "queue.Queue[RunSpec]" = queue.Queue()
        for r in runs:
            q.put(r)
        self._log(dict(
            event="sweep_start", n_runs=len(runs), gpus=self.gpus,
            num_workers=self.num_workers, co_tenant=(self.co_tenant or None),
            co_tenancy_note=(
                "Runs are co-located with the Fed-CORE workload on every GPU. Fed-CORE's "
                "own load changes as it crosses campaign phase boundaries, so per-epoch "
                "wall-clock is NOT expected to be stationary across this sweep. This "
                "affects the ETA only -- seeds, determinism and all logged metrics are "
                "independent of how much GPU time a run gets."),
            epochs=self.cfg["train"]["epochs"], env=preflight(), ts=time.time()))
        threads = [threading.Thread(target=self._worker, args=(g, q), daemon=True) for g in self.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        done = sum(is_done(self.results_dir, r.run_id, self.epochs) for r in runs)
        self._log(dict(event="sweep_end", done=done, n_runs=len(runs), ts=time.time()))
        print(f"[sweep] {done}/{len(runs)} runs terminal")
        return 0 if done == len(runs) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--co-tenant", default="fedcore2",
                   help="workload sharing these GPUs; recorded in every run's metadata")
    p.add_argument("--cells", default="",
                   help="comma-separated (dataset_noise_eta_learner) cells to run; "
                        "default is the whole grid")
    a = p.parse_args(argv)
    preflight()                      # fail fast, before any run is dispatched
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    return Sweep(cfg_path=a.config, gpus=gpus, results_dir="", max_retries=a.max_retries,
                 co_tenant=a.co_tenant, cells=cells).run()


if __name__ == "__main__":
    raise SystemExit(main())
