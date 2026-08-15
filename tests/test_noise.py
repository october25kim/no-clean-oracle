"""(a) Noise-injection: empirical flip rates + transition matrices match config.

Assertions use aggregate statistics (overall flip rate, diagonal mean, mean over
flip cells) which are tight at these sample sizes; per-cell max-abs is only checked
for CIFAR-10 (10x10), since a 100x100 CIFAR-100 matrix has O(10^4) sparse cells whose
worst-cell sampling error is large at any feasible N.
"""
import numpy as np
import pytest

from data.noise import (
    NoiseConfig, build_transition_matrix, cifar100_asym_flip, CIFAR10_ASYM_FLIP,
    empirical_flip_rate, empirical_transition, inject_label_noise,
)

PC = {"cifar10": 5000, "cifar100": 2000}   # per-class sample counts


def _balanced(dataset):
    nC = 10 if dataset == "cifar10" else 100
    return np.repeat(np.arange(nC), PC[dataset]), nC


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("eta", [0.2, 0.4, 0.6])
def test_symmetric_flip_rate_and_structure(dataset, eta):
    clean, nC = _balanced(dataset)
    noisy = inject_label_noise(clean, NoiseConfig(dataset, "symmetric", eta, seed=0))
    That = empirical_transition(clean, noisy, nC)
    # overall flip rate matches eta
    assert abs(empirical_flip_rate(clean, noisy) - eta) < 0.01
    # diagonal ~ 1-eta ; off-diagonal ~ uniform eta/(C-1)
    assert abs(np.mean(np.diag(That)) - (1 - eta)) < 0.01
    off = That[~np.eye(nC, dtype=bool)]
    assert abs(off.mean() - eta / (nC - 1)) < 0.005
    if dataset == "cifar10":
        assert np.max(np.abs(That - build_transition_matrix(dataset, "symmetric", eta))) < 0.02


def test_asymmetric_cifar10_targets():
    dataset, eta = "cifar10", 0.4
    clean, nC = _balanced(dataset)
    noisy = inject_label_noise(clean, NoiseConfig(dataset, "asymmetric", eta, seed=1))
    That = empirical_transition(clean, noisy, nC)
    for i, j in CIFAR10_ASYM_FLIP.items():
        assert abs(That[i, j] - eta) < 0.02, (i, j, That[i, j])
        assert abs(That[i, i] - (1 - eta)) < 0.02
    for i in set(range(10)) - set(CIFAR10_ASYM_FLIP):   # classes that are not a source stay put
        assert That[i, i] > 1 - 0.005


def test_asymmetric_cifar100_within_superclass():
    dataset, eta = "cifar100", 0.4
    clean, nC = _balanced(dataset)
    noisy = inject_label_noise(clean, NoiseConfig(dataset, "asymmetric", eta, seed=2))
    That = empirical_transition(clean, noisy, nC)
    flip = cifar100_asym_flip()
    assert len(flip) == 100
    flip_cells = np.array([That[i, flip[i]] for i in flip])
    assert abs(flip_cells.mean() - eta) < 0.01                  # aggregate is tight
    assert np.mean(np.abs(flip_cells - eta) < 0.05) > 0.95      # nearly all cells close
    # every flip target lies inside the source's own superclass
    from data.noise import cifar100_superclass_map
    grp_of = {c: gi for gi, g in enumerate(cifar100_superclass_map()) for c in g}
    assert all(grp_of[i] == grp_of[flip[i]] for i in flip)


def test_determinism_and_ce_elr_identical():
    clean, _ = _balanced("cifar10")
    cfg = NoiseConfig("cifar10", "symmetric", 0.4, seed=7)
    a = inject_label_noise(clean, cfg)
    b = inject_label_noise(clean, cfg)      # same cfg -> identical (CE and ELR view)
    assert np.array_equal(a, b)
    c = inject_label_noise(clean, NoiseConfig("cifar10", "symmetric", 0.4, seed=8))
    assert not np.array_equal(a, c)         # different seed -> different corruption
