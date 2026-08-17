"""The recon frame must differ from the registered one in exactly two ways, and no others.

[EXPLORATORY-UNVERIFIED-PROVENANCE] for what it supports; the tests themselves are ordinary.

``ReconRunFrame`` overrides three methods of ``analysis.corrected.RunFrame`` so that the
axis set and the grid length come from the instance rather than from module constants. The
risk in a subclass like this is not that the overrides are wrong -- they are three lines
each -- but that an inherited method still reaches a module constant behind ``self`` and
silently mixes a 24-point denominator into a 20-point frame. These tests pin that: every
inherited quantity is checked against a value computed by hand from the two ghat curves.

The last test guards the other direction, that adding the subclass has not perturbed the
registered three-axis path, since that is the code the adjudicated battery ran.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analysis.corrected import AXES, CKPT_GRID, RunFrame, axis_frame   # noqa: E402
from recon_frame import RECON_AXES, RECON_GRID, ReconRunFrame          # noqa: E402

# V-shaped risks: a nonzero IQR, so neither axis is fail-closed and ghat is defined.
R_ID = np.array([.60, .52, .45, .38, .32, .27, .23, .20, .18, .17,
                 .19, .22, .26, .30, .34, .38, .42, .46, .50, .54])
R_WC = np.array([.75, .68, .60, .53, .47, .42, .38, .35, .33, .32,
                 .34, .37, .41, .45, .49, .53, .57, .61, .65, .69])


@pytest.fixture
def frame() -> ReconRunFrame:
    return ReconRunFrame(run_id="probe",
                         axes={"ID": axis_frame("ID", R_ID), "WC": axis_frame("WC", R_WC)})


@pytest.fixture
def hand() -> np.ndarray:
    return np.maximum(axis_frame("ID", R_ID).ghat, axis_frame("WC", R_WC).ghat)


def test_registered_frame_cannot_hold_two_axes():
    """The reason the subclass exists: a two-axis frame is not representable upstream."""
    reg = RunFrame(run_id="probe",
                   axes={"ID": axis_frame("ID", R_ID), "WC": axis_frame("WC", R_WC)})
    with pytest.raises(KeyError, match="OOD"):
        reg.max_g()


def test_max_g_is_the_elementwise_max_over_present_axes(frame, hand):
    assert np.allclose(frame.max_g(), hand)
    assert len(frame.max_g()) == len(RECON_GRID) == 20


def test_rho_star_is_the_min_of_max_g(frame, hand):
    assert frame.rho_star_le() == pytest.approx(float(hand.min()), abs=1e-12)


@pytest.mark.parametrize("delta", [0.10, 0.50, 1.00])
def test_w_delta_uses_the_twenty_point_grid(frame, delta):
    """The denominator is the recon's own grid, not the registered 24."""
    F = frame.feasible_set(delta)
    assert frame.w_delta(delta) == pytest.approx(len(F) / 20, abs=1e-12)
    if len(F):                      # the registered denominator is a different number
        assert frame.w_delta(delta) != pytest.approx(len(F) / len(CKPT_GRID), abs=1e-12)


def test_feasible_set_agrees_with_the_threshold_read(frame, hand):
    for delta in (0.10, 0.50, 1.00):
        assert frame.feasible_set(delta).tolist() == np.flatnonzero(hand <= delta).tolist()


def test_inherited_eta_and_baseline_reach_the_overrides(frame, hand):
    """eta and uniform_baseline are not overridden; they must still see 2 axes and 20 points."""
    i = 9
    assert frame.J(i) == pytest.approx(float(hand[i]), abs=1e-12)
    assert frame.eta(i) == pytest.approx(float(hand[i] - hand.min()), abs=1e-12)
    ub = frame.uniform_baseline(0.50)
    assert ub["p_unif"] == pytest.approx(frame.w_delta(0.50), abs=1e-12)
    assert ub["expected_joint_regret"] == pytest.approx(float(hand.mean()), abs=1e-12)


def test_fail_closed_axis_withholds_the_run():
    """A flat risk has an exactly-zero IQR; D-7's rule must still hold in the subclass."""
    flat = np.full(20, 0.4)
    f = ReconRunFrame(run_id="probe",
                      axes={"ID": axis_frame("ID", flat), "WC": axis_frame("WC", R_WC)},
                      excluded_axes=["ID"])
    assert not f.scorable
    assert np.isnan(f.max_g()).all() and len(f.max_g()) == 20
    assert np.isnan(f.rho_star_le()) and np.isnan(f.w_delta(0.1))
    assert f.feasible_set(0.1).tolist() == []


def test_registered_three_axis_path_is_unperturbed():
    """The adjudicated battery's frame must behave exactly as before."""
    risk = np.concatenate([np.linspace(.6, .2, 12), np.linspace(.22, .5, 12)])
    reg = RunFrame(run_id="reg", axes={a: axis_frame(a, risk) for a in AXES})
    assert len(reg.max_g()) == len(CKPT_GRID) == 24
    assert reg.w_delta(0.5) == pytest.approx(len(reg.feasible_set(0.5)) / 24, abs=1e-12)
    assert RECON_AXES == ("ID", "WC") and AXES == ("ID", "WC", "OOD")
