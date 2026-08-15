"""(d) Oracle-epoch / cross-regret / NCR / bootstrap on synthetic trajectories."""
import numpy as np

from analysis.ncr import (
    analyze_run, aggregate_cell_ncr, bca_ci, moving_average, oracle_epoch, verdict,
)


def test_oracle_epoch_with_smoothing_suppresses_spike():
    # single-epoch spike-down at t=2; sustained low plateau near the end (t=5,6,7)
    r = np.array([1.0, 0.9, 0.05, 0.9, 0.7, 0.32, 0.26, 0.34], dtype=float)
    t_sm, t_raw = oracle_epoch(r, smooth_w=3)
    assert t_raw == 2          # raw argmin hits the 1-epoch spike
    assert t_sm != 2 and t_sm >= 5   # smoothing rejects the spike -> sustained valley


def test_cross_regret_known_trajectories():
    # Two objectives with oracles at different epochs -> nonzero off-diagonal regret.
    # ID: min at t=2 ; OOD: min at t=6 ; tail: min at t=4
    T = 9
    R_ID = np.array([1.0, 0.6, 0.20, 0.30, 0.45, 0.55, 0.60, 0.62, 0.63])
    R_tail = np.array([1.0, 0.9, 0.70, 0.50, 0.30, 0.45, 0.55, 0.60, 0.62])
    R_OOD = np.array([1.0, 0.95, 0.90, 0.80, 0.60, 0.40, 0.20, 0.35, 0.45])
    a = analyze_run({"ID": R_ID, "tail": R_tail, "OOD": R_OOD}, smooth_w=1)
    assert a.t_star["ID"] == 2 and a.t_star["tail"] == 4 and a.t_star["OOD"] == 6
    # diagonal regret is exactly 0
    assert np.allclose(np.diag(a.CR), 0.0)
    # deploying OOD's oracle (t=6) costs ID: R_ID(6)-R_ID(2) = 0.60-0.20 = 0.40
    assert abs(a.CR[0, 2] - 0.40) < 1e-9
    # NCR normalizes by IQR of the measured objective's trajectory
    assert a.NCR[0, 2] > 0 and np.isfinite(a.NCR[0, 2])


def test_bca_ci_covers_and_orders():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=8)
    m, lo, hi = bca_ci(x, n_boot=2000, seed=0)
    assert lo <= m <= hi
    # degenerate (identical) samples -> point interval
    m2, lo2, hi2 = bca_ci([0.3, 0.3, 0.3])
    assert lo2 == hi2 == m2 == 0.3


def test_verdict_pass_and_kill():
    # Build 8 cells. PASS case: strong ID<->OOD off-diagonal conflict in all cells.
    hi_conflict = []
    for _ in range(8):
        ncrs = []
        for s in range(3):
            m = np.zeros((3, 3))
            m[0, 2] = m[2, 0] = 0.5   # ID<->OOD conflict well above 0.10
            ncrs.append(m)
        hi_conflict.append(aggregate_cell_ncr(ncrs))
    aggs = {f"cifar10_symmetric_0.{i}_ce": a for i, a in enumerate(hi_conflict)}
    v = verdict(aggs, thresh=0.10)
    assert v["verdict"] == "PASS", v

    # KILL case: every off-diagonal straddles 0 across the 3 seeds -> its BCa CI
    # contains 0, so all off-diagonal CIs overlap 0 in every cell.
    kill = []
    for _ in range(8):
        # seed values [-0.02, 0, +0.02] for each off-diagonal -> mean 0, CI covers 0
        ncrs = [np.full((3, 3), v) for v in (-0.02, 0.0, 0.02)]
        for m in ncrs:
            np.fill_diagonal(m, 0.0)
        kill.append(aggregate_cell_ncr(ncrs))
    aggs_k = {f"cifar100_symmetric_0.{i}_elr": a for i, a in enumerate(kill)}
    vk = verdict(aggs_k, thresh=0.10)
    assert vk["verdict"] == "KILL", vk
