"""In-memory OOD evaluation pools (built once per worker, no per-epoch disk I/O).

Semantic pools (primary): SVHN test, and the opposite CIFAR. Covariate pool:
``CIFAR-C-local`` — a fixed 4-corruption x 2,000 subsample at severity 3 generated
with the official imagecorruptions library on a seeded local subsample. It is NOT
official CIFAR-10-C/100-C and is never reported as such (see build_ood_pools.py).
The npz is a FROZEN artifact under results/ood_pools/; all phases reuse the exact file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

POOL_CIFARC = "CIFAR-C-local"                     # display + key name; never "CIFAR-10-C"
CIFARC_CORRUPTIONS: List[str] = ["gaussian_noise", "defocus_blur", "brightness", "contrast"]
CIFARC_SEVERITY = 3
CIFARC_PER_CORRUPTION = 2000

# Sealed by the owner: any report, table, or figure that uses this pool must carry
# this sentence verbatim. make_report.py emits it rather than re-wording it.
CIFARC_LOCAL_REPORT_SENTENCE = (
    "The covariate OOD pool named CIFAR-C-local is a local substitution for the official "
    "CIFAR-10-C / CIFAR-100-C benchmark: a fixed 4-corruption x 2,000-image subsample at "
    "severity 3, generated with the imagecorruptions library (v1.1.2) on a seeded local "
    "CIFAR test subsample; it is NOT official CIFAR-C and its numbers are not comparable "
    "to published CIFAR-C results."
)


@dataclass
class OODPools:
    pools: Dict[str, np.ndarray]                  # uint8 (N,32,32,3) per named pool
    meta: dict = field(default_factory=dict)      # sealed provenance from the npz

    @property
    def names(self) -> List[str]:
        return list(self.pools)


def cache_path(cache_dir: str, id_dataset: str) -> str:
    return os.path.join(cache_dir, f"ood_pools_{id_dataset}.npz")


def load_ood_pools(cache_dir: str, id_dataset: str) -> OODPools:
    path = cache_path(cache_dir, id_dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"OOD pool cache missing: {path}. Build it first with scripts/build_ood_pools.py")
    d = np.load(path, allow_pickle=False)
    meta = {}
    pools = {}
    for k in d.files:
        if k == "__meta__":
            meta = json.loads(str(d[k]))
        else:
            pools[k] = d[k]
    return OODPools(pools=pools, meta=meta)
