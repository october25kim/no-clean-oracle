"""Tier 2 — Clothing1M training, per tier2_amendment.md and tier2_followup.md.

Registered pins, none of them chosen here: pretrained ResNet-50 (ImageNet), batch 64,
lr 0.001 divided by 10 after epoch 5, 10 epochs, SGD momentum 0.9, weight decay 0.001 with
u and v excluded; checkpoint grid of 2 per epoch by iteration count, 20 points per run;
1,000,000 noisy training images at 224 px.

**Loader/compute instrumentation.** Ruling 46-L2 asks for a mechanism rather than a number:
if data loading dominates, the completion estimate has to say so. The training loop therefore
times two disjoint spans per iteration -- the wait for the next batch, and the forward /
backward / step -- and reports both per half-epoch. They are disjoint by construction
(the timer starts again only after the optimizer step returns), so loader + compute + a small
residual is the wall clock, and the split is measured rather than inferred from utilisation.

**Transform provenance.** The amendment pins the optimizer but not the augmentation. The
official SOP Clothing1M convention is Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip
-> ImageNet normalize, and that is what is used, recorded here as a transcription rather than
a choice. Evaluation is Resize(256) -> CenterCrop(224) -> normalize, deterministic.

**SOP is not launchable yet.** Table A.1 of the SOP paper has a Clothing-1M column, but
docs/TRANSCRIPTION.md transcribes only the CIFAR-10 and CIFAR-100 rows, and no alpha_u /
alpha_v for Clothing1M exists anywhere in this repository. configs/ext_tier1.yaml keys its
SOP alphas by split precisely "so a new split cannot silently inherit a value"; Tier 2 has no
entry, and borrowing c100n's (1.0, 10.0) is exactly the silent inheritance that comment
forbids. This script refuses --learner sop until the pin exists rather than improvising one.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data", "clothing1m_official", "extracted")
OUT = os.path.join(ROOT, "results", "tier2")
N_CLASSES = 14
BATCH = 64
LR = 0.001
LR_DROP_AFTER_EPOCH = 5
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
EPOCHS = 10
CKPT_PER_EPOCH = 2
# Clothing-1M column of Table A.1, arXiv:2202.14026v2 / PMLR v162 pp.14153-14172, pinned by
# ruling 48-L2 and transcribed in docs/TRANSCRIPTION.md. VERIFIED against the paper text.
SOP_ALPHA_U = 0.1
SOP_ALPHA_V = 1.0
SOP_PROJECT = 1.0          # u, v projected to [-1, 1] after each step
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find(name: str) -> str:
    for dp, _d, fs in os.walk(DATA):
        if name in fs:
            return os.path.join(dp, name)
    raise SystemExit(f"{name} not found under {DATA}")


def load_index() -> Tuple[List[str], np.ndarray]:
    """Noisy training keys and labels, in the official file's own order."""
    lab = {}
    with open(find("noisy_label_kv.txt")) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                lab[p[0]] = int(p[1])
    keys = [l.strip() for l in open(find("noisy_train_key_list.txt")) if l.strip()]
    missing = [k for k in keys[:1000] if k not in lab]
    if missing:
        raise SystemExit(f"{len(missing)} of the first 1000 keys have no noisy label")
    y = np.asarray([lab[k] for k in keys], dtype=np.int64)
    return keys, y


def key_to_path(key: str) -> str:
    # keys read "images/3/25/<file>.jpg"; the tars extracted "3/25/<file>.jpg".
    return os.path.join(DATA, key[len("images/"):] if key.startswith("images/") else key)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--learner", choices=["ce", "sop"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)


    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as T
    from provenance import code_stamp

    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to train from a dirty tree")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True          # fixed shapes; safe and much faster here

    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    run_id = f"c1m_tier2_{a.learner}_seed{a.seed}"
    run_dir = os.path.join(OUT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    keys, y = load_index()
    print(f"[tier2] {len(keys):,} noisy training images, {N_CLASSES} classes", flush=True)

    train_tf = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class C1M(Dataset):
        def __len__(self): return len(keys)
        def __getitem__(self, i):
            img = Image.open(key_to_path(keys[i])).convert("RGB")
            return train_tf(img), int(y[i]), i

    loader = DataLoader(C1M(), batch_size=BATCH, shuffle=True, num_workers=a.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True,
                        prefetch_factor=4)

    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, N_CLASSES)
    net = net.to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    sop = None
    if a.learner == "sop":
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from train.sop import SOPLoss
        # u and v are per-sample, so n_train is the full 1M noisy set.
        sop = SOPLoss(n_train=len(keys), n_classes=N_CLASSES, lr_u=SOP_ALPHA_U,
                      lr_v=SOP_ALPHA_V, device=dev)
        print(f"[tier2] SOP: alpha_u={SOP_ALPHA_U} alpha_v={SOP_ALPHA_V} "
              f"projection to [-{SOP_PROJECT:g}, {SOP_PROJECT:g}] after each step; "
              f"u {tuple(sop.u.shape)} v {tuple(sop.v.shape)}", flush=True)

    iters_per_epoch = len(loader)
    half = iters_per_epoch // CKPT_PER_EPOCH
    print(f"[tier2] {iters_per_epoch:,} iters/epoch, checkpoint every {half:,} iters "
          f"({CKPT_PER_EPOCH}/epoch, {CKPT_PER_EPOCH * EPOCHS} points)", flush=True)

    meta = dict(run_id=run_id, learner=a.learner, seed=a.seed, n_train=len(keys),
                n_classes=N_CLASSES, batch=BATCH, lr=LR, lr_drop_after_epoch=LR_DROP_AFTER_EPOCH,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, epochs=EPOCHS,
                iters_per_epoch=iters_per_epoch, ckpt_every_iters=half,
                backbone="resnet50", weights="ResNet50_Weights.IMAGENET1K_V1",
                train_transform="Resize(256)+RandomCrop(224)+HFlip+ImageNetNorm",
                sop=(dict(alpha_u=SOP_ALPHA_U, alpha_v=SOP_ALPHA_V, projection=SOP_PROJECT,
                          lambda_C=0.0, lambda_B=0.0, init_std=1e-8, wd_uv=0.0,
                          source="Table A.1 Clothing-1M column, arXiv:2202.14026v2, "
                                 "VERIFIED; ruling 48-L2",
                          lr_disclosure="main text 0.001 vs Table A.1 0.002; amendment pins "
                                        "0.001 and it stands, disclosed not resolved")
                     if a.learner == "sop" else None),
                device_uuid_note="see PLACEMENT-style record in the launcher log",
                code_stamp=stamp, classification="TIER2-VERIFIED-OFFICIAL-DATA")
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1)

    # "a" would concatenate a relaunch onto the previous attempt's rows, which is how a
    # restarted run ends up with 21 points and two point-01s. A run directory that already
    # holds metrics is a previous attempt: refuse rather than silently merge or silently
    # delete someone's data.
    mpath = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(mpath) and os.path.getsize(mpath) > 0:
        raise SystemExit(f"{mpath} already has rows from a previous attempt; remove the run "
                         f"directory deliberately before relaunching")
    mf = open(mpath, "w")
    t_run = time.time()
    step = 0
    for epoch in range(EPOCHS):
        for g in opt.param_groups:
            g["lr"] = LR * (0.1 if epoch >= LR_DROP_AFTER_EPOCH else 1.0)
        net.train()
        t_load = t_comp = 0.0
        losses, seen, correct = 0.0, 0, 0
        t_epoch = time.time()
        t0 = time.time()
        for xb, yb, _idx in loader:
            t_load += time.time() - t0                      # span 1: waiting for data
            t1 = time.time()
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            out = net(xb)
            if sop is None:
                loss = crit(out, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            else:
                idxb = _idx.to(dev, non_blocking=True)
                loss = sop(out, yb, idxb)
                opt.zero_grad(set_to_none=True)
                sop.zero_grad()
                loss.backward()
                opt.step()
                sop.step()
                # Projection to [-1, 1] after each step, per the Clothing-1M pin. It is done
                # HERE and not in src/train/sop.py on purpose: that module is the code that
                # produced the Tier-1 SOP results, and whether the projection is general to
                # Algorithm 1 or specific to this column is an open question (see
                # docs/TRANSCRIPTION.md). Leaving the shared module byte-identical keeps the
                # Tier-1 artifacts and the module that made them in correspondence whichever
                # way that ruling goes.
                with torch.no_grad():
                    sop.u.clamp_(-SOP_PROJECT, SOP_PROJECT)
                    sop.v.clamp_(-SOP_PROJECT, SOP_PROJECT)
            torch.cuda.synchronize(dev)                     # so the span is real, not queued
            t_comp += time.time() - t1                      # span 2: forward/backward/step
            losses += float(loss) * yb.size(0); seen += yb.size(0)
            correct += int((out.argmax(1) == yb).sum())
            step += 1
            if step % half == 0:
                k = step // half
                torch.save(dict(step=step, epoch=epoch, model=net.state_dict()),
                           os.path.join(run_dir, f"checkpoint_{k:03d}.pt"))
                wall = time.time() - t_run
                rec = dict(point=k, epoch=epoch, step=step, train_loss=losses / max(seen, 1),
                           train_acc=correct / max(seen, 1), lr=opt.param_groups[0]["lr"],
                           seconds=round(wall, 1), loader_s=round(t_load, 1),
                           compute_s=round(t_comp, 1))
                if sop is not None:
                    # Without this the SOP arm is unfalsifiable. u and v start at std 1e-8, so
                    # a run in which they never move is numerically CE plus a constant and
                    # looks perfectly healthy from loss and accuracy alone. The checkpoints
                    # store only the network, so it cannot be checked afterwards -- it has to
                    # be recorded while it happens.
                    with torch.no_grad():
                        u, v = sop.u.detach(), sop.v.detach()
                        rec["sop_u_absmean"] = float(u.abs().mean())
                        rec["sop_v_absmean"] = float(v.abs().mean())
                        rec["sop_u_absmax"] = float(u.abs().max())
                        rec["sop_v_absmax"] = float(v.abs().max())
                        rec["sop_u_frac_at_bound"] = float((u.abs() >= SOP_PROJECT).float().mean())
                        rec["sop_v_frac_at_bound"] = float((v.abs() >= SOP_PROJECT).float().mean())
                        rec["sop_u_frac_moved"] = float((u.abs() > 1e-6).float().mean())
                        rec["sop_v_frac_moved"] = float((v.abs() > 1e-6).float().mean())
                mf.write(json.dumps(rec) + "\n"); mf.flush()
                if sop is not None:
                    print(f"    SOP u |mean| {rec['sop_u_absmean']:.3e} max {rec['sop_u_absmax']:.3e} "
                          f"moved {100*rec['sop_u_frac_moved']:.1f}% at-bound {100*rec['sop_u_frac_at_bound']:.2f}% | "
                          f"v |mean| {rec['sop_v_absmean']:.3e} moved {100*rec['sop_v_frac_moved']:.1f}%",
                          flush=True)
                frac = t_load / max(t_load + t_comp, 1e-9)
                print(f"  point {k:02d}/{CKPT_PER_EPOCH*EPOCHS} epoch {epoch} "
                      f"loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} "
                      f"| wall {wall/60:.1f}m loader {t_load/60:.1f}m compute {t_comp/60:.1f}m "
                      f"-> loader {100*frac:.0f}%", flush=True)
            t0 = time.time()
        print(f"[tier2] epoch {epoch} done in {(time.time()-t_epoch)/60:.1f} min "
              f"(loader {t_load/60:.1f}m, compute {t_comp/60:.1f}m)", flush=True)

    payload = dict(meta)
    payload.update(status="completed", wall_seconds=round(time.time() - t_run, 1))
    tmp = os.path.join(run_dir, "TERMINAL.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, os.path.join(run_dir, "TERMINAL.json"))
    mf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
