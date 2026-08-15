"""Build the FROZEN in-memory OOD evaluation pools ONCE (CPU-side; no training GPU).

Pools per ID dataset -> results/ood_pools/ood_pools_{cifar10,cifar100}.npz:
  - svhn          : SVHN test images (semantic shift; torchvision).
  - cross_cifar   : the opposite CIFAR test set (semantic shift; local).
  - CIFAR-C-local : a fixed 4-corruption x 2,000 covariate subsample at severity 3,
                    generated with the OFFICIAL imagecorruptions library (the same
                    corruption functions used to build official CIFAR-10-C/100-C),
                    applied to a seeded local subsample of the clean CIFAR test set.

CIFAR-C SUBSTITUTION (sealed provenance). This is NOT official CIFAR-10-C/CIFAR-100-C
and its numbers are NOT comparable to official CIFAR-C results. Downloading the ~2.9 GB
official tar per dataset to keep a 16k-image SECONDARY (covariate) pool is
disproportionate; the primary OOD endpoint is energy-vs-semantic (SVHN), which is exact.
The corruption library name+version and all parameters/seeds are recorded in the npz
meta and a sidecar provenance json.

FROZEN: existing npz files are never regenerated silently (pass --force to override,
which you should not do — G2+ must reuse these exact files). SHA-256 is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data.ood_pools import (  # noqa: E402
    CIFARC_CORRUPTIONS, CIFARC_PER_CORRUPTION, CIFARC_SEVERITY, POOL_CIFARC, cache_path,
)

OUT = os.path.join(ROOT, "results", "ood_pools")
DATA_ROOT = os.environ.get("CIFAR_DATA_ROOT", "/data/workspace/sanghoon/fedcore2/data")
SUBSAMPLE_SEED = 20260811
CORRUPTION_SEED = 20260811


def _clean_test_images(dataset: str) -> np.ndarray:
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root=DATA_ROOT, train=False, download=False).data  # (N,32,32,3) uint8


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cifar_c_local(dataset: str) -> np.ndarray:
    """4 corruptions x CIFARC_PER_CORRUPTION images at severity 3 via imagecorruptions."""
    import imagecorruptions as ic
    imgs = _clean_test_images(dataset)
    idx = np.random.default_rng(SUBSAMPLE_SEED).choice(len(imgs), size=CIFARC_PER_CORRUPTION, replace=False)
    base = imgs[np.sort(idx)]
    np.random.seed(CORRUPTION_SEED)                      # imagecorruptions uses global np.random
    out = []
    for name in CIFARC_CORRUPTIONS:
        corr = np.stack([ic.corrupt(base[i], corruption_name=name, severity=CIFARC_SEVERITY)
                         for i in range(len(base))], axis=0)
        out.append(corr.astype(np.uint8))
    return np.concatenate(out, axis=0)


def provenance() -> dict:
    return dict(
        cifar_c_local=dict(
            name="CIFAR-C-local",
            NOT_official="This is NOT official CIFAR-10-C/CIFAR-100-C; numbers are not comparable.",
            library="imagecorruptions", library_version=md.version("imagecorruptions"),
            backend="opencv-python-headless", backend_version=md.version("opencv-python-headless"),
            corruptions=list(CIFARC_CORRUPTIONS), severity=CIFARC_SEVERITY,
            per_corruption=CIFARC_PER_CORRUPTION, subsample_seed=SUBSAMPLE_SEED,
            corruption_seed=CORRUPTION_SEED),
        svhn=dict(source="torchvision.SVHN(split=test)"),
        cross_cifar=dict(source="opposite CIFAR clean test set"),
    )


def build(dataset: str, force: bool) -> None:
    import torchvision
    os.makedirs(OUT, exist_ok=True)
    path = cache_path(OUT, dataset)
    if os.path.isfile(path) and not force:
        print(f"{dataset}: FROZEN artifact exists ({path}); skipping (use --force to override)")
        return
    other = "cifar100" if dataset == "cifar10" else "cifar10"
    cross = _clean_test_images(other)
    svhn = torchvision.datasets.SVHN(root=os.path.join(DATA_ROOT, "svhn"), split="test", download=True)
    svhn_imgs = np.transpose(svhn.data, (0, 2, 3, 1)).astype(np.uint8)
    cifar_c = build_cifar_c_local(dataset)
    prov = provenance()
    pools = {"svhn": svhn_imgs, "cross_cifar": cross, POOL_CIFARC: cifar_c,
             "__meta__": np.asarray(json.dumps(prov))}
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **pools)
    os.replace(tmp, path)
    sha = _sha256(path)
    with open(os.path.join(OUT, f"provenance_{dataset}.json"), "w") as fh:
        json.dump(dict(dataset=dataset, sha256=sha, shapes={k: list(v.shape) for k, v in pools.items()
                                                            if k != "__meta__"}, **prov), fh, indent=2)
    with open(os.path.join(OUT, f"ood_pools_{dataset}.sha256"), "w") as fh:
        fh.write(f"{sha}  ood_pools_{dataset}.npz\n")
    print(f"{dataset}: svhn {svhn_imgs.shape} cross_cifar {cross.shape} {POOL_CIFARC} {cifar_c.shape}")
    print(f"  imagecorruptions {prov['cifar_c_local']['library_version']} "
          f"(cv2-headless {prov['cifar_c_local']['backend_version']}), severity {CIFARC_SEVERITY}, "
          f"seed {SUBSAMPLE_SEED}; FROZEN sha256 {sha[:16]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="override frozen artifacts (do not use)")
    a = p.parse_args(argv)
    for ds in ("cifar10", "cifar100"):
        build(ds, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
