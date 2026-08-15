"""Shared trainer: CE now, ELR pluggable at step 3. Online per-epoch evaluation,
JSONL logging (one line per epoch), checkpoint every N + final, crash-safe resume.

The learner is a loss STRATEGY: it maps (logits, targets, indices) -> scalar loss.
CE is stateless; ELR keeps a per-sample target-probability EMA (added at step 3).
Everything else (backbone, optimizer, schedule, eval, logging) is shared so that
CE vs ELR differ ONLY in the training objective.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CELoss:
    name = "ce"

    def __call__(self, logits, targets, indices=None):
        return F.cross_entropy(logits, targets)


def build_loss(learner: str, n_train: int, n_classes: int, params: dict):
    if learner == "ce":
        return CELoss()
    if learner == "elr":
        from train.elr import ELRLoss
        missing = [k for k in ("lambda", "beta") if k not in params]
        if missing:                       # never silently fall back to CIFAR-10 defaults
            raise ValueError(f"ELR is missing hyperparameter(s) {missing}; configs/base.yaml "
                             f"must supply learners.elr.per_dataset.<dataset>")
        return ELRLoss(n_train, n_classes, lam=params["lambda"], beta=params["beta"])
    if learner == "gce":
        from train.gce import GCELoss
        if "q" not in params:
            raise ValueError("GCE is missing hyperparameter 'q'; configs must supply "
                             "learners.gce.q")
        return GCELoss(n_train, n_classes, q=params["q"])
    if learner == "sop":
        from train.sop import SOPLoss
        missing = [k for k in ("alpha_u", "alpha_v") if k not in params]
        if missing:
            # No default: Table A.1 covers symmetric and asymmetric synthetic noise
            # only, so for CIFAR-N there is nothing to transcribe and a fallback would
            # be an invented hyperparameter. See docs/TRANSCRIPTION.md, "OPEN".
            raise ValueError(
                f"SOP is missing {missing}; configs must supply "
                f"learners.sop.per_dataset.<dataset>.alpha_u/alpha_v. For CIFAR-N no "
                f"official value exists — see docs/TRANSCRIPTION.md, section OPEN.")
        return SOPLoss(n_train, n_classes, lr_u=params["alpha_u"],
                       lr_v=params["alpha_v"])
    raise ValueError(f"unknown learner {learner!r}")


def cosine_lr(base_lr: float, epoch: int, total: int) -> float:
    return 0.5 * base_lr * (1 + math.cos(math.pi * epoch / total))


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    lr_schedule: str = "cosine"
    save_every: int = 10


def _atomic_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _truncate_jsonl_to_epoch(path: str, last_epoch: int) -> None:
    """Keep the metadata line + per-epoch lines with epoch <= last_epoch."""
    if not os.path.isfile(path):
        return
    kept = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                kept.append(line); continue
            try:
                if json.loads(line).get("epoch", 1 << 30) <= last_epoch:
                    kept.append(line)
            except json.JSONDecodeError:
                break
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.writelines(kept)
    os.replace(tmp, path)


class Trainer:
    def __init__(self, model: nn.Module, loss_fn, tcfg: TrainConfig, device: str,
                 run_dir: str, metadata: dict, eval_fn: Callable[[nn.Module], Dict],
                 diag_fn: Optional[Callable] = None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.tcfg = tcfg
        self.device = device
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl = os.path.join(run_dir, "metrics.jsonl")
        # Two concerns, two paths, never shared: the per-epoch files are the SELECTION
        # trajectory (a checkpoint-selection audit needs every scheduled epoch kept),
        # latest.pt is only the crash-resume pointer. G1's original runs wrote both
        # roles to one path, so each save overwrote the last and only the final epoch
        # survived; legacy_ckpt lets those runs still resume.
        self.latest = os.path.join(run_dir, "latest.pt")
        self.legacy_ckpt = os.path.join(run_dir, "checkpoint.pt")
        self.lock_path = os.path.join(run_dir, "RUNNING.lock")
        self.metadata = metadata
        self.eval_fn = eval_fn
        self.diag_fn = diag_fn
        self.opt = torch.optim.SGD(model.parameters(), lr=tcfg.lr, momentum=tcfg.momentum,
                                   weight_decay=tcfg.weight_decay, nesterov=False)
        self.start_epoch = 0
        self._prev_pred: Optional[np.ndarray] = None

    def acquire_lock(self, stale_after_s: float = 900.0) -> None:
        """Refuse to train into a run_dir another process is already writing.

        The sweep's only completion test is TERMINAL.json, so a run that is in flight
        looks identical to one that never started: launching the sweep over a live run
        would put two processes in one directory, interleaving metrics.jsonl and racing
        checkpoint writes. The lock is created O_EXCL and heartbeated each epoch, so a
        genuinely dead process (no heartbeat for stale_after_s) can be taken over while
        a live one cannot.
        """
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            age = time.time() - os.path.getmtime(self.lock_path)
            if age < stale_after_s:
                with open(self.lock_path) as fh:
                    holder = fh.read().strip()
                raise SystemExit(
                    f"run_dir is locked by a live process (heartbeat {age:.0f}s ago): "
                    f"{holder}. Refusing to write into {self.run_dir}. If that process "
                    f"is truly gone, remove {self.lock_path}.")
            os.remove(self.lock_path)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(dict(pid=os.getpid(), started=time.time(),
                                     host=os.uname().nodename)))

    def _heartbeat(self) -> None:
        try:
            os.utime(self.lock_path, None)
        except OSError:
            pass

    def release_lock(self) -> None:
        try:
            os.remove(self.lock_path)
        except OSError:
            pass

    def _write_metadata(self) -> None:
        """Metadata is rewritten whenever the run starts from epoch 0.

        Leaving the previous process's line in place would attribute a fresh trajectory
        to the old code_stamp, GPU and start time.
        """
        if self.start_epoch == 0 or not os.path.isfile(self.jsonl):
            with open(self.jsonl, "w") as fh:
                fh.write(json.dumps({"record": "metadata", **self.metadata}) + "\n")

    def _append(self, rec: dict) -> None:
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def epoch_ckpt(self, epoch: int) -> str:
        return os.path.join(self.run_dir, f"checkpoint_ep{epoch:03d}.pt")

    def maybe_resume(self) -> None:
        path = self.latest if os.path.isfile(self.latest) else self.legacy_ckpt
        if os.path.isfile(path):
            ck = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.start_epoch = ck["epoch"] + 1
            self._prev_pred = ck.get("prev_pred")
            if hasattr(self.loss_fn, "load_state"):
                self.loss_fn.load_state(ck.get("loss_state"))
            _truncate_jsonl_to_epoch(self.jsonl, ck["epoch"])

    def _save_ckpt(self, epoch: int) -> None:
        state = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), epoch=epoch,
                     prev_pred=self._prev_pred)
        if hasattr(self.loss_fn, "state"):
            state["loss_state"] = self.loss_fn.state()
        path = self.epoch_ckpt(epoch)
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, path)
        # latest.pt is a hard link to the newest epoch file: atomic to swap, and it
        # costs no second copy of a ~90 MB state.
        link = f"{self.latest}.tmp.{os.getpid()}"
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.link(path, link)
        except OSError:                      # different filesystem / no hardlink support
            torch.save(state, link)
        os.replace(link, self.latest)

    def train(self, train_loader: DataLoader, n_classes: int, train_labels: np.ndarray) -> None:
        # Any epoch record at or beyond where we are about to start is a leftover from a
        # previous attempt. Truncating unconditionally (not only inside maybe_resume)
        # is what stops a restart from appending a second 0..119 sequence.
        _truncate_jsonl_to_epoch(self.jsonl, self.start_epoch - 1)
        self._write_metadata()
        for epoch in range(self.start_epoch, self.tcfg.epochs):
            t0 = time.time()
            lr = cosine_lr(self.tcfg.lr, epoch, self.tcfg.epochs) if self.tcfg.lr_schedule == "cosine" else self.tcfg.lr
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.model.train()
            losses = np.zeros(len(train_labels), dtype=np.float32)
            preds = np.zeros(len(train_labels), dtype=np.int64)
            running = 0.0
            for xb, yb, idx in train_loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                self.opt.zero_grad()
                if hasattr(self.loss_fn, "zero_grad"):
                    self.loss_fn.zero_grad()          # SOP's separate u,v optimizer
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb, idx)
                loss.backward()
                self.opt.step()
                if hasattr(self.loss_fn, "step"):
                    self.loss_fn.step()
                running += float(loss) * xb.size(0)
                with torch.no_grad():
                    idx_np = np.asarray(idx)
                    losses[idx_np] = F.cross_entropy(logits, yb, reduction="none").detach().cpu().numpy()
                    preds[idx_np] = logits.argmax(1).detach().cpu().numpy()
            t_train = time.time()
            rec = {"epoch": epoch, "lr": lr, "train_loss_noisy": running / len(train_labels)}
            rec.update(self.eval_fn(self.model))
            t_eval = time.time()
            rec["train_time_s"] = round(t_train - t0, 2)
            rec["eval_time_s"] = round(t_eval - t_train, 2)
            rec["epoch_time_s"] = round(t_eval - t0, 2)
            if self.diag_fn is not None:
                rec.update(self.diag_fn(losses, self._prev_pred, preds, train_labels, n_classes))
            self._prev_pred = preds
            self._append(rec)
            self._heartbeat()
            if (epoch + 1) % self.tcfg.save_every == 0 or epoch == self.tcfg.epochs - 1:
                self._save_ckpt(epoch)
        _atomic_json(os.path.join(self.run_dir, "TERMINAL.json"),
                     {"status": "completed", "epochs": self.tcfg.epochs, **self.metadata})
