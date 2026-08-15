"""ELR loss: EMA target update, objective value, and checkpoint round-trip.

torch lives only in the training container and the analysis host has no torch, so
this module skips on the host. The training image has no pytest either; to run it
there, install pytest into a throwaway container first:
  docker run --rm --user 1000:1000 -v <repo>:<repo> -w <repo> fedcore-c400r:latest \
      sh -c "pip install -q pytest && python -m pytest tests/test_elr.py -q"
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from train.elr import ELRLoss          # noqa: E402
from train.trainer import build_loss   # noqa: E402


def _logits(n=6, c=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g, requires_grad=True)


def test_target_bank_matches_hand_computed_ema():
    lam, beta = 3.0, 0.7
    loss_fn = ELRLoss(n_train=10, n_classes=4, lam=lam, beta=beta)
    logits = _logits()
    idx = torch.arange(6)

    p = F.softmax(logits, 1).clamp(1e-4, 1 - 1e-4).detach()
    expected = (1.0 - beta) * (p / p.sum(1, keepdim=True))      # from a zero-initialised bank

    loss_fn(logits, torch.zeros(6, dtype=torch.long), idx)
    assert torch.allclose(loss_fn.target[idx], expected, atol=1e-7)

    # a second step on the same samples applies the EMA again
    before = loss_fn.target[idx].clone()
    loss_fn(logits, torch.zeros(6, dtype=torch.long), idx)
    assert torch.allclose(loss_fn.target[idx], beta * before + expected, atol=1e-7)

    # untouched rows stay exactly zero
    assert torch.count_nonzero(loss_fn.target[6:]) == 0


def test_objective_is_ce_plus_lambda_times_regularizer():
    lam, beta = 3.0, 0.7
    loss_fn = ELRLoss(n_train=6, n_classes=4, lam=lam, beta=beta)
    logits = _logits()
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    idx = torch.arange(6)

    out = loss_fn(logits, y, idx)

    p = F.softmax(logits, 1).clamp(1e-4, 1 - 1e-4)
    reg = torch.log(1.0 - (loss_fn.target[idx] * p).sum(1)).mean()
    assert torch.allclose(out, F.cross_entropy(logits, y) + lam * reg, atol=1e-6)
    assert reg.item() < 0.0                      # <t, p> in (0, 1) => log(1 - .) < 0
    out.backward()
    assert torch.isfinite(logits.grad).all()


def test_zero_lambda_reduces_to_cross_entropy():
    loss_fn = ELRLoss(n_train=6, n_classes=4, lam=0.0, beta=0.7)
    logits = _logits()
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    assert torch.allclose(loss_fn(logits, y, torch.arange(6)), F.cross_entropy(logits, y),
                          atol=1e-6)


def test_state_round_trip_restores_the_target_bank():
    a = ELRLoss(n_train=6, n_classes=4, lam=3.0, beta=0.7)
    a(_logits(), torch.zeros(6, dtype=torch.long), torch.arange(6))
    b = ELRLoss(n_train=6, n_classes=4, lam=3.0, beta=0.7)
    b.load_state(a.state())
    assert torch.allclose(a.target, b.target)

    b.load_state(None)                           # a checkpoint without ELR state is a no-op
    assert torch.allclose(a.target, b.target)

    with pytest.raises(ValueError):              # refuse a bank trained under other settings
        ELRLoss(n_train=6, n_classes=4, lam=7.0, beta=0.9).load_state(a.state())


def test_missing_indices_and_missing_hyperparameters_are_hard_errors():
    with pytest.raises(ValueError):
        ELRLoss(6, 4)(_logits(), torch.zeros(6, dtype=torch.long), None)
    with pytest.raises(ValueError):
        build_loss("elr", 6, 4, {"beta": 0.7})   # no silent CIFAR-10 default for lambda


def test_build_loss_passes_the_configured_hyperparameters():
    fn = build_loss("elr", 50000, 100, {"lambda": 7.0, "beta": 0.9, "reference": "x"})
    assert (fn.lam, fn.beta, fn.target.shape) == (7.0, 0.9, (50000, 100))
