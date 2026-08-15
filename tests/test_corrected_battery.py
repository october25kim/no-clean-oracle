"""A-battery synthetic fixtures (pin 23).

Every case is hand-constructed. None of the numbers here come from a real log: no value,
range, IQR, epoch, or summary is copied or approximated from the audited runs. That is the
point of the fixture suite — it must be possible to check the battery's arithmetic without
the checker having seen an outcome.

The list is exactly the one pinned: exact ties; IQR = 0; singleton and empty feasible
sets; minimax selection correctness; eta >= 0 within one estimand layer; the exact uniform
baseline |F_delta|/24; and absence of leakage in the fold partition.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.corrected import (AXES, CKPT_GRID, AxisFrame, RunFrame, axis_frame,
                                classify, corrected_oracle, denominator,
                                effective_rank, random_folds, stratified_folds)

N = len(CKPT_GRID)


def _run(id_curve, wc_curve, ood_curve, floor=0.0) -> RunFrame:
    axes = {a: axis_frame(a, c, floor)
            for a, c in zip(AXES, (id_curve, wc_curve, ood_curve))}
    return RunFrame(run_id="fixture", axes=axes,
                    excluded_axes=[a for a, f in axes.items() if f.excluded])


def _flat(v):
    return np.full(N, float(v))


def _v(center, slope=0.1):
    """A V-shaped risk curve: unique minimum at ``center``, and real spread.

    Curves with a single dip on an otherwise flat base are NOT usable here: their IQR is
    exactly zero, so the fail-closed rule withholds them, which is correct behaviour and
    was caught by these fixtures on first run. Every curve below therefore has a
    non-degenerate mid-50%.
    """
    return np.abs(np.arange(N) - center) * float(slope)


# --- exact ties -------------------------------------------------------------------

def test_oracle_takes_the_earliest_index_on_an_exact_tie():
    y = np.full(N, 0.5)
    y[3] = 0.2
    y[9] = 0.2                              # bit-identical minimum, later in the grid
    assert corrected_oracle(y) == 3


def test_tie_rule_is_not_an_accident_of_float_comparison():
    y = np.full(N, 1.0)
    for i in (5, 6, 7):
        y[i] = 0.25
    assert corrected_oracle(y) == 5


# --- IQR = 0, the D-7 case --------------------------------------------------------

def test_exactly_zero_iqr_is_fail_closed_not_zero_regret():
    d, excluded = denominator(_flat(0.3), floor=0.0)
    assert excluded is True and d == 0.0

    f = axis_frame("ID", _flat(0.3))
    assert f.excluded is True
    assert np.isnan(f.ghat).all(), "a flat axis must not be scored, least of all as perfect"


def test_a_fail_closed_axis_withholds_every_joint_quantity():
    r = _run(_flat(0.3), _v(4), _v(4, 0.05))
    assert r.scorable is False and r.excluded_axes == ["ID"]
    assert np.isnan(r.rho_star_le())
    assert r.feasible_set(0.10).size == 0
    assert np.isnan(r.w_delta(0.10))
    assert classify(r.rho_star_le(), {}, 0.10) == "unscorable"


def test_a_floor_makes_the_flat_axis_scorable_again_as_a_sensitivity():
    d, excluded = denominator(_flat(0.3), floor=1e-3)
    assert excluded is False and d == 1e-3
    f = axis_frame("ID", _flat(0.3), floor=1e-3)
    assert np.isfinite(f.ghat).all() and np.allclose(f.ghat, 0.0)


def test_near_zero_but_nonzero_iqr_is_not_fail_closed():
    y = _flat(0.3).copy()
    y[:12] += 1e-9                          # tiny but real spread
    d, excluded = denominator(y, floor=0.0)
    assert excluded is False and d > 0.0


# --- feasible sets: singleton and empty -------------------------------------------

def test_singleton_feasible_set():
    # one epoch is simultaneously best on all three axes; the V slope is steep enough
    # that its immediate neighbours already exceed delta
    y = _v(7)
    r = _run(y, y.copy(), y.copy())
    F = r.feasible_set(0.10)
    assert F.tolist() == [7]
    assert r.w_delta(0.10) == pytest.approx(1.0 / N)
    assert r.rho_star_le() == pytest.approx(0.0)


def test_empty_feasible_set_means_incompatible():
    # each axis is minimized where the others are worst
    r = _run(_v(2), _v(11), _v(20))
    assert r.feasible_set(0.10).size == 0
    assert r.w_delta(0.10) == 0.0
    assert r.rho_star_le() > 0.10
    assert classify(r.rho_star_le(), {"E_tau1": 5.0}, 0.10) == "incompatible"


def test_empty_feasible_set_iff_rho_exceeds_delta():
    for centers in [(2, 11, 20), (7, 7, 7), (5, 6, 7), (0, 12, 23), (9, 10, 11)]:
        r = _run(_v(centers[0]), _v(centers[1]), _v(centers[2]))
        rho, F = r.rho_star_le(), r.feasible_set(0.10)
        assert (F.size == 0) == (rho > 0.10)


# --- taxonomy ---------------------------------------------------------------------

def test_indeterminate_band_overrides_both_sides():
    assert classify(0.10 + 0.02, {"E_tau1": 0.0}, 0.10) == "indeterminate"
    assert classify(0.10 - 0.02, {"E_tau1": 9.0}, 0.10) == "indeterminate"
    assert classify(0.10 + 0.03, {"E_tau1": 0.0}, 0.10) == "incompatible"


def test_only_registered_selectors_gate_the_taxonomy():
    # a post-registered selector succeeding must NOT produce compatible-solved
    assert classify(0.01, {"LW_N": 0.0}, 0.10) == "compatible-unsolved"
    assert classify(0.01, {"E_tau1": 0.05}, 0.10) == "compatible-solved"
    assert classify(0.01, {"E_tau1": 0.5, "NA": 0.4, "ER_argmax": 0.3},
                    0.10) == "compatible-unsolved"


# --- minimax selection ------------------------------------------------------------

def test_minimax_point_is_the_argmin_of_the_axiswise_max():
    a = np.linspace(0.0, 1.0, N)
    b = np.linspace(1.0, 0.0, N)
    c = _v(11, 0.001)                       # spread, but negligible against a and b
    r = _run(a, b, c)
    m = r.max_g()
    assert int(np.argmin(m)) == int(np.argmin(np.maximum(r.axes["ID"].ghat,
                                                         r.axes["WC"].ghat)))
    assert r.rho_star_le() == pytest.approx(float(np.min(m)))


def test_max_g_is_the_pointwise_max_over_axes():
    r = _run(_v(3), _v(8, 0.07), _v(15, 0.06))
    stacked = np.vstack([r.axes[a].ghat for a in AXES])
    assert np.allclose(r.max_g(), stacked.max(axis=0))


# --- eta >= 0 within one estimand layer -------------------------------------------

def test_eta_is_nonnegative_for_every_grid_point_within_a_layer():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = _run(rng.random(N), rng.random(N), rng.random(N))
        rho = r.rho_star_le()
        for t in range(N):
            assert r.eta(t) >= -1e-12, "J - rho must be >= 0 inside one layer"
        assert min(r.J(t) for t in range(N)) == pytest.approx(rho)


def test_eta_is_zero_exactly_at_the_minimax_point():
    r = _run(_v(6), _v(6, 0.08), _v(6, 0.07))
    t = int(np.argmin(r.max_g()))
    assert r.eta(t) == pytest.approx(0.0)


# --- exact uniform baseline -------------------------------------------------------

def test_p_unif_equals_feasible_fraction_exactly():
    # a plateau of six jointly-optimal epochs, then a rise steep enough that the seventh
    # is already outside delta -- and with a non-degenerate IQR, so nothing fails closed
    y = np.concatenate([np.zeros(6), np.linspace(1.0, 5.0, N - 6)])
    r = _run(y, y.copy(), y.copy())
    base = r.uniform_baseline(0.10)
    assert base["p_unif"] == pytest.approx(6 / N)
    assert base["p_unif"] == pytest.approx(len(r.feasible_set(0.10)) / N)


def test_expected_uniform_joint_regret_is_the_mean_of_max_g():
    r = _run(_v(3), _v(8, 0.07), _v(15, 0.06))
    assert r.uniform_baseline(0.10)["expected_joint_regret"] == pytest.approx(
        float(np.mean(r.max_g())))


# --- fold partition: no leakage ---------------------------------------------------

def test_stratified_folds_partition_without_overlap_or_loss():
    y = np.repeat(np.arange(10), 100)
    f = stratified_folds(y, 5, 20260814)
    assert f.size == y.size and set(np.unique(f)) == set(range(5))
    counts = [np.flatnonzero(f == q).size for q in range(5)]
    assert sum(counts) == y.size
    assert len(set.union(*[set(np.flatnonzero(f == q)) for q in range(5)])) == y.size
    for q in range(5):
        others = set().union(*[set(np.flatnonzero(f == p)) for p in range(5) if p != q])
        assert not (set(np.flatnonzero(f == q)) & others), "folds must be disjoint"


def test_stratified_folds_balance_each_class():
    y = np.repeat(np.arange(10), 100)
    f = stratified_folds(y, 5, 20260814)
    for c in range(10):
        per = [np.sum((y == c) & (f == q)) for q in range(5)]
        assert max(per) - min(per) <= 1


def test_fold_partition_is_fixed_by_seed_not_by_call():
    y = np.repeat(np.arange(10), 50)
    assert np.array_equal(stratified_folds(y, 5, 20260814),
                          stratified_folds(y, 5, 20260814))
    assert not np.array_equal(stratified_folds(y, 5, 20260814),
                              stratified_folds(y, 5, 1))


def test_random_folds_partition_a_pool_without_loss():
    f = random_folds(1000, 5, 20260814)
    assert f.size == 1000 and set(np.unique(f)) == set(range(5))
    assert sum(np.flatnonzero(f == q).size for q in range(5)) == 1000


# --- effective rank details (spec D.5) --------------------------------------------

def test_effective_rank_zero_spectrum_fails_closed():
    assert np.isnan(effective_rank(np.ones((32, 8))))   # zero variance after centering


def test_effective_rank_of_an_isotropic_cloud_approaches_the_dimension():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 8))
    assert 6.0 < effective_rank(x) <= 8.0


def test_effective_rank_of_a_rank_one_cloud_is_near_one():
    rng = np.random.default_rng(2)
    d = rng.normal(size=8)
    x = rng.normal(size=(2000, 1)) * d
    assert effective_rank(x) == pytest.approx(1.0, abs=1e-6)
