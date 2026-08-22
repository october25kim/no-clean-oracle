"""Build the Tier-2 OOD pools: C1M-C-local (near) and ISIC2019 (far).

Both are pinned in ``docs/tier2_followup.md`` and ratified as 44-L1 / 46-L1.

**C1M-C-local** follows the CIFAR-C-local generation convention exactly, as the amendment
requires: the same four corruptions, the same severity 3, the same 2,000 images per
corruption, generated with the same ``imagecorruptions`` library. The convention is copied
from ``src/data/ood_pools.py`` rather than restated, so the two pools cannot drift apart.
Corruptions are applied at 224 -- the model's input resolution -- exactly as CIFAR-C-local
applied them at CIFAR's native 32.

As with CIFAR-C-local this is a **local substitution**, not an official corruption benchmark,
and any report using it carries that sentence.

**ISIC2019** is the far pool: 25,331 dermoscopic images from
``ISIC_2019_Training_Input_preprocessed``, whose short side is already 224. The count is the
corrected one -- an earlier registration said 50,662, which double-counted the raw and
preprocessed copies of the same images.

The library versions are pinned to what the sealed CIFAR-C-local provenance records
(``imagecorruptions`` 1.1.2, ``opencv-python-headless`` 4.10.0.84) and asserted at run time,
because "the same convention" is a claim about the toolchain as much as the parameters. A
different version could produce a differently-corrupted pool that still looks entirely valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data.ood_pools import (CIFARC_CORRUPTIONS, CIFARC_PER_CORRUPTION,   # noqa: E402
                            CIFARC_SEVERITY)

EXTRACTED = os.path.join(ROOT, "data", "clothing1m_official", "extracted")
ISIC = "/data/workspace/sanghoon/fedcore2/data/isic2019/ISIC_2019_Training_Input_preprocessed"
OUT = os.path.join(ROOT, "results", "ood_pools_tier2")
RES = 224
SUBSAMPLE_SEED = 20260822
CORRUPTION_SEED = 20260822
POOL_NEAR = "C1M-C-local"
POOL_FAR = "ISIC2019"

# From the sealed CIFAR-C-local provenance (results/ood_pools/provenance_cifar10.json).
REQUIRE = {"imagecorruptions": "1.1.2", "opencv-python-headless": "4.10.0.84"}

C1M_C_LOCAL_REPORT_SENTENCE = (
    "The covariate OOD pool named C1M-C-local is a local substitution built to the same "
    "convention as CIFAR-C-local: a fixed 4-corruption x 2,000-image subsample at severity 3, "
    "generated with the imagecorruptions library (v1.1.2) on a seeded subsample of the "
    "Clothing1M clean test set at 224 px; it is NOT an official corruption benchmark and its "
    "numbers are not comparable to published corruption-benchmark results."
)


def find(name: str) -> str:
    for dp, _d, fs in os.walk(EXTRACTED):
        if name in fs:
            return os.path.join(dp, name)
    raise SystemExit(f"{name} not found under {EXTRACTED}")


def key_path(k: str) -> str:
    return os.path.join(EXTRACTED, k[len("images/"):] if k.startswith("images/") else k)


def assert_versions() -> Dict[str, str]:
    import importlib.metadata as md
    got = {}
    for pkg, want in REQUIRE.items():
        try:
            v = md.version(pkg)
        except Exception:
            raise SystemExit(f"{pkg} is not installed; the sealed convention requires {want}")
        got[pkg] = v
        if v != want:
            raise SystemExit(
                f"{pkg} is {v}, the sealed CIFAR-C-local provenance records {want}. Refusing: "
                f"'the same generation convention' is a claim about the toolchain, and a "
                f"different version can produce a differently-corrupted pool that still looks "
                f"valid.")
    return got


def load_resized(paths: List[str], res: int, tag: str) -> np.ndarray:
    from PIL import Image
    out = np.zeros((len(paths), res, res, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((res, res), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)
        if (i + 1) % 5000 == 0:
            print(f"    {tag}: {i+1:,}/{len(paths):,}", flush=True)
    return out


def main() -> int:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from provenance import code_stamp
    stamp = code_stamp()
    if stamp.get("git_available") and stamp.get("git_tree_dirty"):
        raise SystemExit("R2: refusing to build pools from a dirty tree")

    versions = assert_versions()
    print(f"[pools] toolchain verified: {versions}")
    import imagecorruptions

    os.makedirs(OUT, exist_ok=True)
    keys = [l.strip() for l in open(find("clean_test_key_list.txt")) if l.strip()]
    print(f"[pools] clean test: {len(keys):,} images")

    # --- C1M-C-local -------------------------------------------------------------
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    chunks, index = [], {}
    for corr in CIFARC_CORRUPTIONS:
        sel = rng.choice(len(keys), size=CIFARC_PER_CORRUPTION, replace=False)
        imgs = load_resized([key_path(keys[i]) for i in sel], RES, f"{corr} source")
        np.random.seed(CORRUPTION_SEED)          # imagecorruptions uses global np.random
        out = np.zeros_like(imgs)
        for j in range(len(imgs)):
            out[j] = imagecorruptions.corrupt(imgs[j], corruption_name=corr,
                                              severity=CIFARC_SEVERITY)
        index[corr] = [int(x) for x in sel]
        chunks.append(out)
        print(f"[pools] {corr}: {out.shape} severity {CIFARC_SEVERITY}", flush=True)
    near = np.concatenate(chunks)

    # --- ISIC2019 ----------------------------------------------------------------
    isic = sorted(f for f in os.listdir(ISIC) if f.lower().endswith(".jpg"))
    print(f"[pools] ISIC2019: {len(isic):,} images")
    far = load_resized([os.path.join(ISIC, f) for f in isic], RES, "ISIC2019")

    path = os.path.join(OUT, "ood_pools_c1m.npz")
    np.savez(path, **{POOL_NEAR: near, POOL_FAR: far})
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    prov = dict(
        classification="TIER2-VERIFIED-OFFICIAL-DATA", code_stamp=stamp, sha256=h,
        resolution=RES,
        shapes={POOL_NEAR: list(near.shape), POOL_FAR: list(far.shape)},
        near=dict(name=POOL_NEAR, corruptions=list(CIFARC_CORRUPTIONS),
                  severity=CIFARC_SEVERITY, per_corruption=CIFARC_PER_CORRUPTION,
                  subsample_seed=SUBSAMPLE_SEED, corruption_seed=CORRUPTION_SEED,
                  library="imagecorruptions", library_version=versions["imagecorruptions"],
                  backend="opencv-python-headless",
                  backend_version=versions["opencv-python-headless"],
                  source="Clothing1M clean test set, official release",
                  NOT_official=C1M_C_LOCAL_REPORT_SENTENCE,
                  source_indices=index),
        far=dict(name=POOL_FAR, n=len(isic),
                 source="ISIC_2019_Training_Input_preprocessed (short side already 224)",
                 caveat=("dermoscopy carries distinctive low-level statistics that plausibly "
                         "make OOD detection EASIER; that direction under-counts "
                         "incompatibility and is therefore conservative for an existence "
                         "check")))
    with open(os.path.join(OUT, "provenance_c1m.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    print(f"[pools] wrote {os.path.relpath(path, ROOT)}  sha256 {h[:16]}…")
    print(f"[pools] near {near.shape}  far {far.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
