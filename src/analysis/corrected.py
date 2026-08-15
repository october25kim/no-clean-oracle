"""Corrected-frame primitives for the remediation battery (spec v2, plan v2 §1).

Everything here is [CORRECTED-PRIMARY]. The frozen G1/G2 machinery in ``ncr.py`` and
``selection.py`` is left untouched so it keeps reproducing the record it already produced;
this module is the parallel implementation the corrected analyses use.

Four differences from the frozen frame, each pinned rather than chosen:

* the candidate set is the 24 retained checkpoints, not the 120 logged epochs;
* the oracle is the argmin of the **raw** risk on that grid (earliest tie), not the argmin
  of a smoothed curve read on the raw one;
* the denominator is ``d_a = max{IQR_{t in T} R_a(t), eps_a}`` (spec C.2), and an
  **exactly** zero IQR is fail-closed — the run is excluded on that axis with a flag,
  never scored 0.0, which is the D-7 defect;
* the axis is named WC (worst-class), because both clean test sets are class-balanced and
  no frequency tail exists (F3).

``rho_star_le`` is the empirical joint compatibility radius, ``min_t max_a ghat_a(t)``
(spec C.3), and carries no general sign of bias: per-axis LE optimism inflates each
``ghat`` while the outer min over 24 candidates deflates the compromise. Nothing here
converts it to a population claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AXES: Tuple[str, ...] = ("ID", "WC", "OOD")
CKPT_GRID: Tuple[int, ...] = tuple(e for e in range(120) if (e + 1) % 5 == 0)
DELTAS: Tuple[float, ...] = (0.05, 0.10, 0.20)
DELTA_PRIMARY = 0.10
INDETERMINATE_BAND = 0.025
FLOORS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)
REGISTERED_SELECTORS: Tuple[str, ...] = ("E_tau1", "NA", "ER_argmax")
RNG_SEED = 20260814
K_CV = 5


def corrected_oracle(risk: np.ndarray) -> int:
    """argmin of the raw risk over the candidate grid; earliest index on ties.

    ``np.argmin`` already returns the first minimum, so the tie rule is the default
    rather than an extra step -- but it is the pinned rule, so it is asserted by test
    rather than left to a library detail.
    """
    return int(np.argmin(np.asarray(risk, dtype=np.float64)))


def iqr(x: np.ndarray) -> float:
    q75, q25 = np.percentile(np.asarray(x, dtype=np.float64), [75, 25])
    return float(q75 - q25)


def denominator(risk: np.ndarray, floor: float = 0.0) -> Tuple[float, bool]:
    """``d_a = max{IQR(R_a), eps_a}`` and whether the axis is fail-closed.

    Returns ``(d_a, excluded)``. ``excluded`` is True only when the IQR is **exactly**
    zero and no floor is in force: that is the case D-7 used to score as a perfect 0.0
    normalized regret. With a floor the axis is scorable, which is why the floor grid is
    a sensitivity and not the primary.
    """
    scale = iqr(risk)
    if scale == 0.0 and floor <= 0.0:
        return 0.0, True
    return max(scale, floor), False


@dataclass
class AxisFrame:
    """One run, one axis, in the corrected frame."""
    axis: str
    risk: np.ndarray                 # raw risk on the 24-point grid
    t_star: int                      # index into the grid
    R_star: float
    d: float
    excluded: bool                   # fail-closed on an exactly-zero IQR
    ghat: np.ndarray                 # normalized regret curve; all-NaN when excluded
    delta_raw: np.ndarray            # raw regret vector (spec C.2, always accompanies)


def axis_frame(axis: str, risk: Sequence[float], floor: float = 0.0) -> AxisFrame:
    y = np.asarray(risk, dtype=np.float64)
    t = corrected_oracle(y)
    d, excluded = denominator(y, floor)
    raw = y - y[t]
    g = np.full(y.shape, np.nan) if excluded else raw / d
    return AxisFrame(axis=axis, risk=y, t_star=t, R_star=float(y[t]), d=d,
                     excluded=excluded, ghat=g, delta_raw=raw)


@dataclass
class RunFrame:
    """A run's three axes plus the joint quantities built on them."""
    run_id: str
    axes: Dict[str, AxisFrame]
    excluded_axes: List[str] = field(default_factory=list)

    @property
    def scorable(self) -> bool:
        """A joint quantity needs every axis; one fail-closed axis withholds the run."""
        return not self.excluded_axes

    def max_g(self) -> np.ndarray:
        """``max_a ghat_a(t)`` over the grid; all-NaN if any axis is fail-closed."""
        if not self.scorable:
            return np.full(len(CKPT_GRID), np.nan)
        return np.max(np.vstack([self.axes[a].ghat for a in AXES]), axis=0)

    def rho_star_le(self) -> float:
        m = self.max_g()
        return float("nan") if not self.scorable else float(np.min(m))

    def feasible_set(self, delta: float) -> np.ndarray:
        """``F_delta`` as grid indices (spec C.4). Empty array if unscorable."""
        if not self.scorable:
            return np.asarray([], dtype=int)
        return np.flatnonzero(self.max_g() <= delta)

    def w_delta(self, delta: float) -> float:
        if not self.scorable:
            return float("nan")
        return float(len(self.feasible_set(delta)) / len(CKPT_GRID))

    def J(self, grid_index: int) -> float:
        """``J_s = max_a ghat_a(t_hat_s)`` (spec D.1)."""
        if not self.scorable:
            return float("nan")
        return float(max(self.axes[a].ghat[grid_index] for a in AXES))

    def eta(self, grid_index: int) -> float:
        """``eta_s = J_s - rho*``, valid only within one estimand layer (spec D.1)."""
        if not self.scorable:
            return float("nan")
        return float(self.J(grid_index) - self.rho_star_le())

    def uniform_baseline(self, delta: float) -> Dict[str, float]:
        """Exact chance baselines (spec D.3) -- no simulation.

        ``p_unif = w_delta`` exactly, and the expected joint regret of picking uniformly
        is the mean of ``max_a ghat_a(t)`` over the grid.
        """
        if not self.scorable:
            return dict(p_unif=float("nan"), expected_joint_regret=float("nan"))
        return dict(p_unif=self.w_delta(delta),
                    expected_joint_regret=float(np.mean(self.max_g())))


def classify(rho: float, selector_J: Dict[str, float], delta: float,
             band: float = INDETERMINATE_BAND) -> str:
    """The corrected run taxonomy (spec D.2, pins 2 and 3).

    The indeterminate band overrides both other verdicts: a run whose rho sits within
    ``band`` of delta is withheld rather than pushed to whichever side it happens to fall.
    Only the REGISTERED selectors decide solved/unsolved -- post-registered ones (LW-N,
    later TGS-N) are reported and never gate.
    """
    if not np.isfinite(rho):
        return "unscorable"
    if abs(rho - delta) <= band:
        return "indeterminate"
    if rho > delta:
        return "incompatible"
    registered = [selector_J[s] for s in REGISTERED_SELECTORS
                  if s in selector_J and np.isfinite(selector_J[s])]
    if registered and min(registered) <= delta:
        return "compatible-solved"
    return "compatible-unsolved"


def stratified_folds(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Class-stratified fold assignment for a labeled evaluation set (pin 10).

    Fixed once per dataset and reused by every run of that dataset, so the folds are a
    property of the evaluation sample rather than of any trajectory.
    """
    y = np.asarray(labels)
    out = np.empty(y.size, dtype=int)
    rng = np.random.default_rng(seed)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(idx)
        out[perm] = np.arange(perm.size) % k
    return out


def random_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Seeded random fold assignment within an unlabeled pool (pin 10)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def effective_rank(feats: np.ndarray) -> float:
    """The REGISTERED effective rank, with spec D.5's zero-spectrum correction.

    The spectrum is the eigenvalue distribution of the 512x512 covariance, exactly as
    ``scripts/b_representation_report.effective_rank`` computes it — that function defined
    the ER selector for the G2 record and is the pin. Taking the singular values of the
    centered data matrix instead is a different statistic, not a refactor: on real
    penultimate features the two differ by an order of magnitude, because squaring sharpens
    a decaying spectrum. This implementation was briefly written the other way and the
    divergence was caught by direct comparison before any real-log execution.

    One deliberate departure, from spec D.5: a spectrum that sums to zero returns NaN
    (fail-closed) rather than 0.0. A degenerate spectrum carries no distribution to take an
    entropy of, and returning a number the caller can rank is the same failure shape as
    D-7. ``0 log 0 := 0`` is unchanged.
    """
    x = np.asarray(feats, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / (len(x) - 1)
    w = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = w.sum()
    if total <= 0.0:
        return float("nan")              # spec D.5 fail-closed; registered returned 0.0
    p = w / total
    p = p[p > 0.0]                       # 0 log 0 := 0
    return float(np.exp(-np.sum(p * np.log(p))))
