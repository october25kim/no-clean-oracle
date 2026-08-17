"""Recon forward pass over the wave-1/wave-2 checkpoints.  [EXPLORATORY-UNVERIFIED-PROVENANCE]

Nothing this produces may enter registered artifacts, paper claims, or be pooled with any
registered result.  Outputs live under ``results/exploratory_c1m/forward/`` only.

Four runs x 20 epochs = 80 checkpoints.  Each is evaluated on two populations:

* the **clean test** set, 10,526 images -- the ID axis, and the worst-class axis over its
  14 classes;
* the **NA split**, 5,000 noisy-labelled training images drawn by the registered
  ``draw_split`` -- the NA selector's evaluation sample, and the sample the effective rank
  is computed on.

The OOD axis is omitted: this reconnaissance has no OOD pool, so there is no third axis and
no two-world certificate to look for.  A1-A5 analogs at delta = 0.10 run over ID and WC only.

Four deviations from the registered forward pass, recorded here because they change what the
numbers mean and none of them is visible in the output files:

1. **The grid is all 20 epochs, not 24 of 120.**  The registered frame retains every fifth
   checkpoint of 120; the recon stores every epoch of 20 and uses all of them.  So w_delta
   is |F_delta| / 20 here, against |F_delta| / 24 in the registered frame.  The two are not
   comparable as fractions of a fixed budget, and the recon's grid is also nearly six times
   coarser in training time per step.

2. **Effective rank is computed on 5,000 samples, not on the full training set.**  The
   registered pass computes it over the whole training forward; 1M x 2048 float32 features
   per checkpoint is ~8 GB, which is not a thing to write 80 times.  The NA split is reused
   as the feature sample.  n = 5,000 against d = 2,048 leaves the covariance estimable but
   not comfortably so, and the resulting rank is a different estimator from the registered
   one, not the same estimator on less data.

3. **The NA draw's remainder branch is live for the first time.**  ``draw_split`` splits
   n_total across n_classes and hands the remainder to the lowest class indices.  With the
   registered 10 classes, 5000 / 10 divides exactly and the branch is dead; with 14 classes
   the remainder is 2, so classes 0 and 1 contribute 358 samples and the other twelve
   contribute 357.  A 0.28% imbalance, immaterial to any conclusion, but it is the first
   execution of that code path and it should be on the record rather than discovered later.

4. **The worst-class axis is granular.**  The smallest clean-test class has 297 members, so
   worst-class accuracy moves in steps of 1/297 = 0.337%.  Differences finer than that
   between checkpoints are quantisation, not signal.

The per-epoch arrays are named ``logits_split`` / ``feats_split``, deliberately *not* the
registered ``logits_train`` / ``feats_train``.  The registered battery indexes its train
logits with absolute indices into the full training set; these arrays are already restricted
to the 5,000 drawn rows.  Had the names matched, pointing the registered battery at this
directory would have silently mis-indexed instead of failing, and the classification barrier
would have depended on nobody making that mistake.  The names make it fail.
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

DATA = os.path.join(ROOT, "data", "exploratory", "clothing1m_kaggle")
RUNS = os.path.join(ROOT, "results", "exploratory_c1m")
OUT = os.path.join(RUNS, "forward")
TAG = "EXPLORATORY-UNVERIFIED-PROVENANCE"
N_CLASSES = 14
BATCH = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _atomic_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def load_split() -> Dict[str, np.ndarray]:
    """The NA split and the clean test set, with the draw's own asserts left in place."""
    from c1_noisyval_report import SPLIT_N, SPLIT_SEED, draw_split

    tr = np.load(os.path.join(DATA, "clothing1m.npz"))
    te = np.load(os.path.join(DATA, "clothing10k_test.npz"))
    noisy = np.asarray(tr["arr_1"], np.int64)
    idx = draw_split(noisy, N_CLASSES, SPLIT_N, SPLIT_SEED)

    counts = np.bincount(noisy[idx], minlength=N_CLASSES)
    per, rem = SPLIT_N // N_CLASSES, SPLIT_N - (SPLIT_N // N_CLASSES) * N_CLASSES
    expect = np.array([per + (1 if c < rem else 0) for c in range(N_CLASSES)])
    if not np.array_equal(counts, expect):
        raise SystemExit(f"NA split composition {counts.tolist()} != {expect.tolist()}")

    y_test = np.asarray(te["arr_1"], np.int64)
    if np.bincount(y_test, minlength=N_CLASSES).min() == 0:
        raise SystemExit("a clean-test class is empty; the worst-class axis is undefined")

    return dict(split_idx=idx, y_split_noisy=noisy[idx], y_test=y_test,
                x_split=np.asarray(tr["arr_0"])[idx], x_test=np.asarray(te["arr_0"]))


def infer(net, x: np.ndarray, dev: str, want_feats: bool):
    """Logits, and optionally penultimate features, in stored order and float32."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    feats: List[np.ndarray] = []
    logits: List[np.ndarray] = []

    hooked = {}
    if want_feats:
        h = net.avgpool.register_forward_hook(
            lambda _m, _i, o: hooked.__setitem__("f", o.detach().flatten(1)))
    try:
        with torch.no_grad():
            for i in range(0, len(x), BATCH):
                b = torch.from_numpy(np.ascontiguousarray(x[i:i + BATCH]))
                b = b.to(dev).permute(0, 3, 1, 2).float().div_(255)
                b = (b - mean) / std                      # deterministic; no augmentation
                out = net(b)
                logits.append(out.float().cpu().numpy())
                if want_feats:
                    feats.append(hooked["f"].float().cpu().numpy())
    finally:
        if want_feats:
            h.remove()
    return (np.concatenate(logits).astype(np.float32),
            np.concatenate(feats).astype(np.float32) if want_feats else None)


def forward_run(run_id: str, gpu: int, sp: Dict[str, np.ndarray]) -> dict:
    import torch
    import torchvision

    run_dir = os.path.join(RUNS, run_id)
    ckpts = sorted(f for f in os.listdir(run_dir) if f.startswith("checkpoint_ep"))
    if not ckpts:
        raise SystemExit(f"{run_id}: no checkpoints")

    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    net = torchvision.models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, N_CLASSES)
    net = net.to(dev).eval()

    out_run = os.path.join(OUT, run_id)
    os.makedirs(out_run, exist_ok=True)
    np.savez(os.path.join(out_run, "labels.npz"),
             split_idx=sp["split_idx"], y_split_noisy=sp["y_split_noisy"],
             y_test=sp["y_test"])

    done = []
    for name in ckpts:
        ep = int(name[len("checkpoint_ep"):-len(".pt")])
        out_ep = os.path.join(out_run, f"ep{ep:03d}")
        marker = os.path.join(out_ep, "OK.json")
        if os.path.isfile(marker):
            done.append(ep)
            continue
        os.makedirs(out_ep, exist_ok=True)
        sd = torch.load(os.path.join(run_dir, name), map_location=dev)
        net.load_state_dict(sd["model"])
        net.eval()

        lg_test, _ = infer(net, sp["x_test"], dev, want_feats=False)
        lg_split, ft_split = infer(net, sp["x_split"], dev, want_feats=True)

        np.save(os.path.join(out_ep, "logits_test.npy"), lg_test)
        np.save(os.path.join(out_ep, "logits_split.npy"), lg_split)
        np.save(os.path.join(out_ep, "feats_split.npy"), ft_split)
        # The marker is written last and only after every array is on disk, so an
        # interrupted epoch is resumed rather than silently treated as complete.
        _atomic_json(marker, dict(epoch=ep, classification=TAG,
                                  logits_test=list(lg_test.shape),
                                  logits_split=list(lg_split.shape),
                                  feats_split=list(ft_split.shape)))
        done.append(ep)
        print(f"  {run_id} ep{ep:03d}: test {lg_test.shape} split {lg_split.shape} "
              f"feats {ft_split.shape}", flush=True)

    return dict(run_id=run_id, epochs=done, n_checkpoints=len(ckpts), classification=TAG)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args(argv)

    from provenance import code_stamp
    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to run a forward pass from a dirty tree")

    os.makedirs(OUT, exist_ok=True)
    sp = load_split()
    print(f"[forward] NA split {len(sp['split_idx'])}, clean test {len(sp['y_test'])}, "
          f"{N_CLASSES} classes  [{TAG}]", flush=True)

    summary = [forward_run(r, a.gpu, sp) for r in a.runs]
    _atomic_json(os.path.join(OUT, f"summary_{'_'.join(a.runs)}.json"),
                 dict(runs=summary, classification=TAG, code_stamp=stamp,
                      grid="all 20 epochs (registered frame retains 24 of 120)",
                      effective_rank_sample="NA split, n=5000 (registered: full train)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
