"""Tier-2 forward pass over the 80 checkpoints, per tier2_amendment.md §3.

Registered axes, none of them chosen here:

* **R_ID**  — error on the clean 10k test set, all 14 classes.
* **R_tail** — error on the clean test restricted to the bottom-k classes by NOISY-label
  training frequency, k = 4, frozen at sealing to **[3, 4, 10, 12]**
  (Chiffon, Sweater, Shawl, Vest). Computed from the noisy labels only, so the
  validation-free premise is preserved: no clean label enters the definition of the subset,
  only the evaluation of it.
* **R_OOD** — two pools, msp and energy scores as in the G1 protocol. Near = C1M-C-local,
  far = ISIC2019, both from the sealed pool artifact.

Selectors E (T-grid), C1 and effective rank need the training-side quantities, so the pass
also stores logits and penultimate features on the **C1 split**: N = 5,000 stratified by
noisy label over 14 classes, per-class quota floor(5000/14) = 357 with remainder 2 assigned
by ascending class index, rng seed 20260813 — the amendment's §5 rule, which is the same
draw_split the registered battery uses, so it is imported rather than reimplemented.

**Stored logits are the raw f(x).** SOP's sparse term s_i = u_i^2 - v_i^2 is training-only
and must not appear in what the analysis reads. The amendment requires it, and
``train.sop.assert_raw_logits`` exists to enforce it by recomputation rather than by
convention. Here the point is structural: this script never constructs u or v at all, so
there is nothing to add. The forward model is a plain ResNet-50 with a 14-way head.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

EXTRACTED = os.path.join(ROOT, "data", "clothing1m_official", "extracted")
POOLS = os.path.join(ROOT, "results", "ood_pools_tier2", "ood_pools_c1m.npz")
RUNS = os.path.join(ROOT, "results", "tier2")
OUT = os.path.join(ROOT, "results", "forward_tier2")
N_CLASSES = 14
RES = 224
BATCH = 256
K_TAIL = [3, 4, 10, 12]                      # frozen at sealing, tier2_followup.md §3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POOL_NAMES = ("C1M-C-local", "ISIC2019")


def find(name: str) -> str:
    for dp, _d, fs in os.walk(EXTRACTED):
        if name in fs:
            return os.path.join(dp, name)
    raise SystemExit(f"{name} not found")


def key_path(k: str) -> str:
    return os.path.join(EXTRACTED, k[len("images/"):] if k.startswith("images/") else k)


def load_labels() -> Dict[str, object]:
    from c1_noisyval_report import SPLIT_N, SPLIT_SEED, draw_split
    clean_keys = [l.strip() for l in open(find("clean_test_key_list.txt")) if l.strip()]
    clean = {}
    for line in open(find("clean_label_kv.txt")):
        p = line.split()
        if len(p) >= 2:
            clean[p[0]] = int(p[1])
    y_test = np.asarray([clean[k] for k in clean_keys], np.int64)

    noisy_keys = [l.strip() for l in open(find("noisy_train_key_list.txt")) if l.strip()]
    noisy = {}
    for line in open(find("noisy_label_kv.txt")):
        p = line.split()
        if len(p) >= 2:
            noisy[p[0]] = int(p[1])
    y_noisy = np.asarray([noisy[k] for k in noisy_keys], np.int64)

    split = draw_split(y_noisy, N_CLASSES, SPLIT_N, SPLIT_SEED)
    counts = np.bincount(y_noisy[split], minlength=N_CLASSES)
    per, rem = SPLIT_N // N_CLASSES, SPLIT_N - (SPLIT_N // N_CLASSES) * N_CLASSES
    expect = np.array([per + (1 if c < rem else 0) for c in range(N_CLASSES)])
    if not np.array_equal(counts, expect):
        raise SystemExit(f"C1 split composition {counts.tolist()} != {expect.tolist()}")
    return dict(clean_keys=clean_keys, y_test=y_test, noisy_keys=noisy_keys,
                y_noisy=y_noisy, split=split, y_split=y_noisy[split])


def load_images(paths: List[str], tag: str) -> np.ndarray:
    from PIL import Image
    out = np.zeros((len(paths), RES, RES, 3), np.uint8)
    for i, p in enumerate(paths):
        out[i] = np.asarray(Image.open(p).convert("RGB").resize((RES, RES), Image.BILINEAR),
                            np.uint8)
        if (i + 1) % 5000 == 0:
            print(f"    {tag}: {i+1:,}/{len(paths):,}", flush=True)
    return out


def infer(net, arr, dev, want_feats):
    import torch
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    L, F = [], []
    hook = {}
    h = net.avgpool.register_forward_hook(
        lambda _m, _i, o: hook.__setitem__("f", o.detach().flatten(1))) if want_feats else None
    try:
        with torch.no_grad():
            for i in range(0, len(arr), BATCH):
                b = torch.from_numpy(np.ascontiguousarray(arr[i:i + BATCH]))
                b = b.to(dev).permute(0, 3, 1, 2).float().div_(255)
                b = (b - mean) / std
                out = net(b)
                L.append(out.float().cpu().numpy())
                if want_feats:
                    F.append(hook["f"].float().cpu().numpy())
    finally:
        if h is not None:
            h.remove()
    return (np.concatenate(L).astype(np.float32),
            np.concatenate(F).astype(np.float32) if want_feats else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args(argv)

    import torch, torchvision
    from provenance import code_stamp
    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to run the forward pass from a dirty tree")

    lab = load_labels()
    print(f"[fwd] clean test {len(lab['y_test']):,}  C1 split {len(lab['split']):,}", flush=True)

    pools = np.load(POOLS)
    for n in POOL_NAMES:
        print(f"[fwd] pool {n}: {pools[n].shape}", flush=True)

    x_test = load_images([key_path(k) for k in lab["clean_keys"]], "clean test")
    x_split = load_images([key_path(lab["noisy_keys"][i]) for i in lab["split"]], "C1 split")

    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    net = torchvision.models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, N_CLASSES)
    net = net.to(dev).eval()

    os.makedirs(OUT, exist_ok=True)
    for rid in a.runs:
        rdir = os.path.join(RUNS, rid)
        cks = sorted(f for f in os.listdir(rdir) if f.startswith("checkpoint_"))
        odir = os.path.join(OUT, rid)
        os.makedirs(odir, exist_ok=True)
        np.savez(os.path.join(odir, "labels.npz"), y_test=lab["y_test"],
                 split_idx=lab["split"], y_split_noisy=lab["y_split"],
                 tail_classes=np.asarray(K_TAIL, np.int64))
        for ck in cks:
            pt = int(ck[len("checkpoint_"):-3])
            od = os.path.join(odir, f"pt{pt:03d}")
            if os.path.isfile(os.path.join(od, "OK.json")):
                continue
            os.makedirs(od, exist_ok=True)
            sd = torch.load(os.path.join(rdir, ck), map_location=dev)
            net.load_state_dict(sd["model"])
            net.eval()
            lt, _ = infer(net, x_test, dev, False)
            ls, fs = infer(net, x_split, dev, True)
            np.save(os.path.join(od, "logits_test.npy"), lt)
            np.save(os.path.join(od, "logits_split.npy"), ls)
            np.save(os.path.join(od, "feats_split.npy"), fs)
            for n in POOL_NAMES:
                lp, _ = infer(net, pools[n], dev, False)
                np.save(os.path.join(od, f"logits_{n}.npy"), lp)
            with open(os.path.join(od, "OK.json"), "w") as fh:
                json.dump(dict(point=pt, epoch=sd["epoch"], step=sd["step"],
                               test=list(lt.shape), split=list(ls.shape),
                               feats=list(fs.shape)), fh)
            print(f"  {rid} pt{pt:03d} (ep{sd['epoch']}): test {lt.shape} split {ls.shape}",
                  flush=True)
        print(f"[fwd] {rid}: {len(cks)} checkpoints done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
