"""Unit checks pinned by the extension instruction for GCE and SOP.

torch lives only in the training container; this module skips on the analysis host.
Both checks are limit checks against CE, because CE is the one learner whose behaviour
is already certified by the G1/B2 chain — anchoring the new learners to it is stronger
than asserting their formulas against themselves.
"""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from train.gce import GCELoss        # noqa: E402
from train.sop import SOPLoss        # noqa: E402
from train.trainer import CELoss     # noqa: E402


def _batch(n=64, c=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, c, generator=g, requires_grad=True)
    y = torch.randint(0, c, (n,), generator=g)
    idx = torch.arange(n)
    return logits, y, idx


def test_gce_q_to_zero_limit_reproduces_cross_entropy():
    """(1 - p^q)/q -> -log p as q -> 0."""
    logits, y, _ = _batch()
    ce = float(CELoss()(logits, y))
    prev = None
    for q in (1e-2, 1e-3, 1e-4, 1e-5):
        gce = float(GCELoss(64, 10, q=q)(logits, y))
        err = abs(gce - ce)
        assert prev is None or err < prev, f"q={q} did not move closer to CE"
        prev = err
    assert abs(float(GCELoss(64, 10, q=1e-5)(logits, y)) - ce) < 1e-3


def test_gce_at_pinned_q_is_bounded_and_differentiable():
    logits, y, _ = _batch()
    loss = GCELoss(64, 10, q=0.7)(logits, y)
    assert 0.0 <= float(loss) <= 1.0 / 0.7      # (1 - p^q)/q with p in (0,1]
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_sop_with_zero_u_v_and_zero_lr_matches_a_ce_step_on_theta():
    """u = v = 0 makes SOP's gradient wrt theta identical to CE's.

    The loss VALUE differs by SOP's MSE term, which at u = v = 0 reduces to
    ||onehot(argmax f) - onehot(y)||_F^2 / B. That term is built from a DETACHED
    prediction, so it contributes no gradient to theta -- which is why the parameter
    step matches CE exactly while the scalar does not.
    """
    logits, y, idx = _batch()
    sop = SOPLoss(64, 10, lr_u=0.0, lr_v=0.0)
    with torch.no_grad():
        sop.u.zero_(); sop.v.zero_()

    l_sop = sop(logits, y, idx)
    g_sop = torch.autograd.grad(l_sop, logits, retain_graph=True)[0]

    logits2 = logits.detach().clone().requires_grad_(True)
    l_ce = CELoss()(logits2, y)
    g_ce = torch.autograd.grad(l_ce, logits2)[0]

    assert torch.allclose(g_sop, g_ce, atol=1e-6), "theta gradient diverges from CE"
    assert abs(float(g_sop.norm()) - float(g_ce.norm())) < 1e-6

    # the residual is exactly the detached MSE term, and it is theta-independent
    hard = F.one_hot(logits.detach().argmax(1), 10).float()
    label = F.one_hot(y, 10).float()
    mse = float(F.mse_loss(hard, label, reduction="sum") / len(label))
    assert abs(float(l_sop) - (float(l_ce) + mse)) < 1e-5


def test_sop_zero_lr_leaves_u_and_v_untouched():
    logits, y, idx = _batch()
    sop = SOPLoss(64, 10, lr_u=0.0, lr_v=0.0)
    with torch.no_grad():
        sop.u.zero_(); sop.v.zero_()
    before_u, before_v = sop.u.detach().clone(), sop.v.detach().clone()
    sop.zero_grad()
    sop(logits, y, idx).backward()
    sop.step()
    assert torch.equal(sop.u.detach(), before_u)
    assert torch.equal(sop.v.detach(), before_v)


def test_sop_parameter_shapes_and_no_weight_decay():
    sop = SOPLoss(50000, 100, lr_u=1.0, lr_v=10.0)
    assert tuple(sop.u.shape) == (50000, 1)      # u is (N, 1), not (N, C)
    assert tuple(sop.v.shape) == (50000, 100)
    lrs = [g["lr"] for g in sop.optimizer.param_groups]
    wds = [g["weight_decay"] for g in sop.optimizer.param_groups]
    moms = [g.get("momentum", 0.0) for g in sop.optimizer.param_groups]
    assert lrs == [1.0, 10.0]
    assert wds == [0.0, 0.0], "u and v must be excluded from weight decay"
    assert all(m == 0.0 for m in moms)


def test_sop_refuses_to_default_its_alphas():
    from train.trainer import build_loss
    with pytest.raises(ValueError) as e:
        build_loss("sop", 100, 10, {"alpha_u": 1.0})     # alpha_v missing
    assert "alpha_v" in str(e.value)
    with pytest.raises(ValueError):
        build_loss("gce", 100, 10, {})                   # q missing


def test_raw_logits_assertion_catches_an_added_sparse_term():
    from train.sop import assert_raw_logits
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 10)
    x = torch.randn(16, 8)
    with torch.no_grad():
        raw = model(x)
    assert_raw_logits(model, x, raw)                     # clean case passes
    with pytest.raises(AssertionError):
        assert_raw_logits(model, x, raw + 0.01)          # sparse term added -> caught
