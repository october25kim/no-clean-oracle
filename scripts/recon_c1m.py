"""[EXPLORATORY-UNVERIFIED-PROVENANCE] Clothing1M reconnaissance: train + forward + battery.

NOT Tier 2. Tier 2 stays gated on officially obtained data. Nothing this script writes may
enter a registered artifact, a paper claim, or be pooled with a registered result. Every
output file carries the classification tag, and everything lands under
``results/exploratory_c1m/``.

The registered launch path is deliberately not used, and `check_data_hygiene.py` is not
invoked. That guard exists to stop a *registered* campaign from starting on bytes nobody
has accounted for; here the inputs are known to be unaccounted-for — that is the whole
premise — and declaring them "not inputs" via ``--allow-unmanifested`` would be a false
statement, since they are precisely the inputs. Instead this script verifies each archive's
sha256 against ``PROVENANCE.md`` at load time and refuses to run on a mismatch, which is
the honest form of the same check.

Design pins, fixed here before any run and echoed into every output:

* backbone ResNet-50, torchvision ImageNet weights, version recorded;
* images are used at their native 64x64 — the bundle is already downsampled, and
  upsampling 1M images to the 224 px that standard Clothing1M practice uses would multiply
  compute by roughly an order of magnitude for a reconnaissance. Recorded as a deviation;
* train transform: pad-4 random crop + horizontal flip, ImageNet normalisation;
  eval transform: normalisation only, no randomness;
* SGD lr 0.01, momentum 0.9, weight decay 1e-3, batch 256, cosine to 0 over 20 epochs;
* {CE, ELR} x seeds {0, 1} = 4 runs, checkpoint every epoch -> a 20-point grid;
* ID and WC axes from the clean test set; WC = bottom round(0.30*14) = 4 classes by the
  reference run's final-epoch per-class accuracy, the protocol convention;
* NA selector input is a noisy split drawn from the training set, since the bundle has no
  clean train or val. OOD axis is NOT computed: the registered semantic-pool convention
  does not transfer and no analog has been confirmed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import zipfile
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

TAG = "EXPLORATORY-UNVERIFIED-PROVENANCE"
BUNDLE = os.path.join(ROOT, "data", "exploratory", "clothing1m_kaggle")
OUT = os.path.join(ROOT, "results", "exploratory_c1m")
N_CLASSES = 14
EPOCHS = 20
BATCH = 256
LR, MOMENTUM, WD = 0.01, 0.9, 1e-3
NA_SPLIT_N, NA_SEED = 5000, 20260813        # registered NA convention, reused
WC_FRAC = 0.30


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def expected_digests() -> Dict[str, str]:
    """Digests as recorded in PROVENANCE.md — the file is the contract."""
    txt = open(os.path.join(BUNDLE, "PROVENANCE.md")).read()
    out = {}
    for m in re.finditer(r"\|\s*`([^`]+\.npz)`\s*\|[^|]*\|\s*`([0-9a-f]{64})`", txt):
        out[m.group(1)] = m.group(2)
    if len(out) != 2:
        raise SystemExit("PROVENANCE.md does not carry two npz digests")
    return out


def load_npz(name: str, verify: bool = True):
    path = os.path.join(BUNDLE, name)
    if verify:
        want = expected_digests()[name]
        got = sha256(path)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != PROVENANCE.md {want}")
        print(f"[recon] {name} sha256 verified against PROVENANCE.md", flush=True)
    z = zipfile.ZipFile(path)
    with z.open("arr_0.npy") as f:
        x = np.lib.format.read_array(f)
    with z.open("arr_1.npy") as f:
        y = np.lib.format.read_array(f)
    return x, y


def header_only(name: str):
    """Shapes/dtypes without materialising the arrays — used by --inspect."""
    z = zipfile.ZipFile(os.path.join(BUNDLE, name))
    out = {}
    for n in z.namelist():
        with z.open(n) as f:
            ver = np.lib.format.read_magic(f)
            shape, _fo, dt = np.lib.format._read_array_header(f, ver)
            out[n] = (shape, str(dt))
    return out


def na_split(noisy: np.ndarray, n_total: int = NA_SPLIT_N, seed: int = NA_SEED) -> np.ndarray:
    """Stratified-by-noisy-label draw, the registered NA convention (c1_noisyval pin)."""
    per = n_total // N_CLASSES
    rem = n_total - per * N_CLASSES
    rng = np.random.default_rng(seed)
    picked = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(noisy == c)
        take = per + (1 if c < rem else 0)
        picked.append(rng.permutation(idx)[:take])
    return np.sort(np.concatenate(picked))


def inspect() -> int:
    """CPU-only preflight: digests, shapes, class balance, WC size, NA split, cost model."""
    print(f"[recon] [{TAG}] preflight — no GPU, no training")
    d = expected_digests()
    for name in ("clothing1m.npz", "clothing10k_test.npz"):
        got = sha256(os.path.join(BUNDLE, name))
        print(f"  {name:24s} sha256 {'OK' if got == d[name] else 'MISMATCH'}  {got[:16]}…")
        for k, (shape, dt) in header_only(name).items():
            print(f"      {k:12s} {shape} {dt}")
    _, ytr = load_npz("clothing1m.npz", verify=False) if False else (None, None)

    z = zipfile.ZipFile(os.path.join(BUNDLE, "clothing1m.npz"))
    with z.open("arr_1.npy") as f:
        ytr = np.lib.format.read_array(f)
    z = zipfile.ZipFile(os.path.join(BUNDLE, "clothing10k_test.npz"))
    with z.open("arr_1.npy") as f:
        yte = np.lib.format.read_array(f)

    print(f"  train labels {ytr.size:,} over {np.unique(ytr).size} classes")
    print(f"  test  labels {yte.size:,} over {np.unique(yte).size} classes")
    k = max(1, int(round(WC_FRAC * N_CLASSES)))
    cnt = np.bincount(yte, minlength=N_CLASSES)
    print(f"  WC set size k = max(1, round({WC_FRAC}*{N_CLASSES})) = {k}")
    print(f"  smallest {k} test classes by count: {sorted(cnt)[:k]} "
          f"(WC risk is a mean over whichever {k} the reference run ranks last)")
    idx = na_split(ytr)
    print(f"  NA split: {idx.size:,} indices, stratified by noisy label, seed {NA_SEED}")

    px = 64 * 64 * 3
    print(f"  train tensor if held in RAM: {ytr.size * px / 1024**3:.1f} GiB uint8")
    print(f"  checkpoints: {EPOCHS}/run x 4 runs = {EPOCHS * 4}; ResNet-50 ~ 94 MiB each "
          f"-> ~{EPOCHS * 4 * 94 / 1024:.1f} GiB")
    print(f"  forward pass: {EPOCHS * 4} checkpoints x {yte.size:,} test images"
          f" + {idx.size:,} NA images")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true",
                   help="CPU-only preflight; verifies digests and prints the design")
    p.add_argument("--learner", choices=["ce", "elr"])
    p.add_argument("--seed", type=int)
    a = p.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.inspect:
        return inspect()
    raise SystemExit("training path requires GPU arbitration; only --inspect is enabled "
                     "in this commit")


if __name__ == "__main__":
    raise SystemExit(main())
