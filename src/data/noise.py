"""Deterministic, seeded label-noise injection for the G1 checkpoint-conflict audit.

Convention (documented + unit-tested). The *noise rate* ``eta`` is the probability
that a clean label is corrupted to a DIFFERENT class, so the row-stochastic
transition matrix ``T`` (T[i, j] = P(observed=j | clean=i)) has diagonal ``1 - eta``
and its off-diagonal row mass sums to ``eta``. Empirical flip rates therefore match
``eta`` directly (see ``tests/test_noise.py``).

Noise types
-----------
- ``symmetric``   : off-diagonal mass spread UNIFORMLY over the other C-1 classes,
                    T[i, j] = eta / (C - 1) for j != i.
- ``asymmetric``  : a single deterministic class->class flip map at rate ``eta``.
                    CIFAR-10 uses the standard DivideMix mapping
                    (truck->automobile, bird->airplane, deer->horse, cat->dog);
                    CIFAR-100 flips each class to the next class WITHIN its 20
                    superclasses (5 classes each), circularly.

Only pure NumPy is used so this module (and its test) run with no torch/GPU. The
corrupted labels for a given (dataset, noise_config, seed) are generated once and
saved to disk, so CE and ELR see the IDENTICAL corrupted labels.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---- standard asymmetric maps ------------------------------------------------

# CIFAR-10 label order: 0 airplane 1 automobile 2 bird 3 cat 4 deer
#                       5 dog 6 frog 7 horse 8 ship 9 truck
CIFAR10_ASYM_FLIP: Dict[int, int] = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
#   truck->automobile, bird->airplane, cat->dog, dog->cat, deer->horse (DivideMix/ELR).


def cifar100_superclass_map() -> List[List[int]]:
    """The 20 CIFAR-100 superclasses as lists of their 5 fine-label indices.

    Fixed grouping from the CIFAR-100 spec (fine labels are contiguous within a
    superclass in this canonical listing used across the noisy-label literature).
    """
    groups = [
        [4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92],
        [9, 10, 16, 28, 61], [0, 51, 53, 57, 83], [22, 39, 40, 86, 87],
        [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], [3, 42, 43, 88, 97],
        [12, 17, 37, 68, 76], [23, 33, 49, 60, 71], [15, 19, 21, 31, 38],
        [34, 63, 64, 66, 75], [26, 45, 77, 79, 99], [2, 11, 35, 46, 98],
        [27, 29, 44, 78, 93], [36, 50, 65, 74, 80], [47, 52, 56, 59, 96],
        [8, 13, 48, 58, 90], [41, 69, 81, 85, 89],
    ]
    return groups


def cifar100_asym_flip() -> Dict[int, int]:
    """Each class -> next class within its superclass (circular)."""
    flip: Dict[int, int] = {}
    for grp in cifar100_superclass_map():
        for k, c in enumerate(grp):
            flip[c] = grp[(k + 1) % len(grp)]
    return flip


# ---- transition matrices -----------------------------------------------------

def symmetric_T(n_classes: int, eta: float) -> np.ndarray:
    T = np.full((n_classes, n_classes), eta / (n_classes - 1), dtype=np.float64)
    np.fill_diagonal(T, 1.0 - eta)
    return T


def asymmetric_T(n_classes: int, eta: float, flip: Dict[int, int]) -> np.ndarray:
    T = np.eye(n_classes, dtype=np.float64)
    for i, j in flip.items():
        T[i, i] = 1.0 - eta
        T[i, j] = eta
    return T


def build_transition_matrix(dataset: str, noise_type: str, eta: float) -> np.ndarray:
    n_classes = 10 if dataset == "cifar10" else 100
    if noise_type == "symmetric":
        return symmetric_T(n_classes, eta)
    if noise_type == "asymmetric":
        flip = CIFAR10_ASYM_FLIP if dataset == "cifar10" else cifar100_asym_flip()
        return asymmetric_T(n_classes, eta, flip)
    raise ValueError(f"unknown noise_type {noise_type!r}")


# ---- injection ---------------------------------------------------------------

@dataclass(frozen=True)
class NoiseConfig:
    dataset: str          # "cifar10" | "cifar100"
    noise_type: str       # "symmetric" | "asymmetric"
    eta: float
    seed: int

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.noise_type}_{self.eta:g}_seed{self.seed}"


def inject_label_noise(clean_labels: np.ndarray, cfg: NoiseConfig) -> np.ndarray:
    """Return corrupted labels drawn per-sample from row ``T[clean]`` under a
    seeded RNG. Deterministic for a fixed ``cfg``."""
    clean = np.asarray(clean_labels, dtype=np.int64)
    T = build_transition_matrix(cfg.dataset, cfg.noise_type, cfg.eta)
    rng = np.random.default_rng(_seed_int(cfg))
    noisy = clean.copy()
    # draw each sample's observed label from its clean-conditioned categorical
    u = rng.random(clean.shape[0])
    cdf = np.cumsum(T, axis=1)               # (C, C)
    rows = cdf[clean]                        # (N, C)
    noisy = (u[:, None] < rows).argmax(axis=1).astype(np.int64)
    return noisy


def _seed_int(cfg: NoiseConfig) -> int:
    h = hashlib.sha256(f"g1|noise|{cfg.name}".encode()).hexdigest()[:8]
    return int(h, 16)


def empirical_flip_rate(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean = np.asarray(clean); noisy = np.asarray(noisy)
    return float((clean != noisy).mean())


def empirical_transition(clean: np.ndarray, noisy: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-normalized empirical T_hat[i, j] = P(noisy=j | clean=i)."""
    M = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(M, (clean, noisy), 1.0)
    row = M.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return M / row


# ---- persistence (identical corrupted labels for CE and ELR) -----------------

def save_noisy_labels(out_dir: str, cfg: NoiseConfig, clean: np.ndarray,
                      noisy: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg.name}.npz")
    tmp = f"{path}.tmp.{os.getpid()}.npz"   # np.savez appends .npz unless already present
    np.savez(tmp, clean=np.asarray(clean, np.int64), noisy=np.asarray(noisy, np.int64),
             flipped=(np.asarray(clean) != np.asarray(noisy)))
    os.replace(tmp, path)
    meta = dict(name=cfg.name, dataset=cfg.dataset, noise_type=cfg.noise_type,
                eta=cfg.eta, seed=cfg.seed, seed_int=_seed_int(cfg),
                empirical_flip_rate=empirical_flip_rate(clean, noisy), n=int(len(clean)))
    with open(os.path.join(out_dir, f"{cfg.name}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


def load_or_make_noisy_labels(out_dir: str, cfg: NoiseConfig,
                              clean: np.ndarray) -> np.ndarray:
    path = os.path.join(out_dir, f"{cfg.name}.npz")
    if os.path.isfile(path):
        return np.load(path)["noisy"]
    noisy = inject_label_noise(clean, cfg)
    save_noisy_labels(out_dir, cfg, clean, noisy)
    return noisy
