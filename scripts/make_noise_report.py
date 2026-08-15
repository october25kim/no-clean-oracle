"""Noise-injection report (Execution-Order step 1 gate): show that the seeded,
deterministic label corruption matches the pre-declared config on the REAL CIFAR
train labels, and persist the noisy-label masks that CE and ELR will both consume.

Runs on host with no torch: CIFAR labels are unpickled directly.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data.noise import (  # noqa: E402
    NoiseConfig, CIFAR10_ASYM_FLIP, build_transition_matrix, cifar100_asym_flip,
    empirical_flip_rate, empirical_transition, inject_label_noise, save_noisy_labels,
)

DATA_ROOT = os.environ.get("CIFAR_DATA_ROOT", "/data/workspace/sanghoon/fedcore2/data")
MASK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "noise_masks")
CONFIGS = [("symmetric", 0.2), ("symmetric", 0.4), ("symmetric", 0.6), ("asymmetric", 0.4)]


def load_clean_labels(dataset: str) -> np.ndarray:
    if dataset == "cifar10":
        labels = []
        for b in range(1, 6):
            with open(os.path.join(DATA_ROOT, "cifar-10-batches-py", f"data_batch_{b}"), "rb") as fh:
                labels += pickle.load(fh, encoding="bytes")[b"labels"]
        return np.array(labels, dtype=np.int64)
    with open(os.path.join(DATA_ROOT, "cifar-100-python", "train"), "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    return np.array(d[b"fine_labels"], dtype=np.int64)


def main() -> int:
    print(f"{'='*84}\nG1 NOISE-INJECTION REPORT (seed 0, real CIFAR train labels)\n{'='*84}")
    print("  'exp.overall' = expected overall corruption from the transition matrix")
    print("  (symmetric: = eta;  CIFAR-10 asymmetric: only 5 of 10 classes are flip")
    print("   sources so overall = 0.4*5/10 = 0.20;  CIFAR-100 asymmetric: all 100 flip)")
    print(f"\n{'dataset':9} {'noise':11} {'eta':>5} {'N':>7} {'exp.overall':>11} {'emp.flip':>9} {'|err|':>7} {'ok':>4}")
    all_ok = True
    for dataset in ("cifar10", "cifar100"):
        clean = load_clean_labels(dataset)
        nC = 10 if dataset == "cifar10" else 100
        for ntype, eta in CONFIGS:
            cfg = NoiseConfig(dataset, ntype, eta, seed=0)
            noisy = inject_label_noise(clean, cfg)
            fr = empirical_flip_rate(clean, noisy)
            T = build_transition_matrix(dataset, ntype, eta)
            exp_overall = float(1.0 - np.mean(np.diag(T)))   # balanced classes
            err = abs(fr - exp_overall)
            ok = err < 0.01
            all_ok &= ok
            save_noisy_labels(MASK_DIR, cfg, clean, noisy)
            print(f"{dataset:9} {ntype:11} {eta:5.2f} {len(clean):7d} {exp_overall:11.3f} {fr:9.4f} {err:7.4f} {str(ok):>4}")
    # asymmetric structure spot-check
    print(f"\n{'-'*74}\nAsymmetric flip-target verification (eta=0.4)\n{'-'*74}")
    c10 = load_clean_labels("cifar10")
    That = empirical_transition(c10, inject_label_noise(c10, NoiseConfig("cifar10", "asymmetric", 0.4, 0)), 10)
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for i, j in CIFAR10_ASYM_FLIP.items():
        print(f"  CIFAR-10  {names[i]:>10} -> {names[j]:<10}  P(flip)={That[i, j]:.3f}  (nominal 0.400)")
    c100 = load_clean_labels("cifar100")
    That100 = empirical_transition(c100, inject_label_noise(c100, NoiseConfig("cifar100", "asymmetric", 0.4, 0)), 100)
    flip = cifar100_asym_flip()
    mean_cell = float(np.mean([That100[i, flip[i]] for i in flip]))
    print(f"  CIFAR-100 within-superclass circular flip: mean P(flip)={mean_cell:.3f} over 100 classes (nominal 0.400)")
    print(f"\nMasks saved under: {MASK_DIR}  (CE and ELR consume the SAME .npz per cell)")
    print(f"\nSTEP-1 NOISE GATE: {'PASS' if all_ok else 'FAIL'}  (all empirical flip rates within 0.01 of config)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
