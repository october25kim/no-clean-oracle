"""Core quantities for the G1 checkpoint-conflict audit (pure NumPy, no GPU).

Given per-epoch risk trajectories R_j(t) for objectives j in {ID, tail, OOD}, this
module computes: oracle epochs t*_j (argmin of a 3-epoch moving average, plus raw),
the cross-regret matrix CR[j', j] = R_{j'}(t*_j) - R*_{j'}, its scale-free
normalization NCR[j', j] = CR / IQR_t(R_{j'}), last-epoch regret, epoch gaps, and a
paired BCa bootstrap over seeds for cell-level aggregation. The pre-registered
decision rule (PASS / WEAK / KILL) is implemented verbatim in ``verdict``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

OBJECTIVES: Tuple[str, ...] = ("ID", "tail", "OOD")


def moving_average(x: np.ndarray, w: int = 3) -> np.ndarray:
    """Centered moving average with edge shrinkage (length-preserving)."""
    x = np.asarray(x, dtype=np.float64)
    if w <= 1 or x.size <= 1:
        return x.copy()
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    sm = np.convolve(xp, k, mode="same")[pad: pad + x.size]
    return sm


def oracle_epoch(risk: np.ndarray, smooth_w: int = 3) -> Tuple[int, int]:
    """Return (t_star_smoothed, t_star_raw), 0-indexed argmin of the risk curve."""
    risk = np.asarray(risk, dtype=np.float64)
    t_raw = int(np.argmin(risk))
    t_sm = int(np.argmin(moving_average(risk, smooth_w)))
    return t_sm, t_raw


def iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


@dataclass
class RunAnalysis:
    """Analysis of ONE run: trajectories dict {obj: R_j(t)} of equal length T."""
    risks: Dict[str, np.ndarray]
    smooth_w: int = 3
    t_star: Dict[str, int] = field(default_factory=dict)
    t_star_raw: Dict[str, int] = field(default_factory=dict)
    R_star: Dict[str, float] = field(default_factory=dict)
    CR: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    NCR: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    last_regret: Dict[str, float] = field(default_factory=dict)
    epoch_gap: Dict[str, int] = field(default_factory=dict)

    def compute(self) -> "RunAnalysis":
        objs = OBJECTIVES
        T = len(next(iter(self.risks.values())))
        for j in objs:
            assert len(self.risks[j]) == T, "objective trajectories must share length T"
            ts, tr = oracle_epoch(self.risks[j], self.smooth_w)
            self.t_star[j] = ts
            self.t_star_raw[j] = tr
            self.R_star[j] = float(self.risks[j][ts])
            self.last_regret[j] = float(self.risks[j][-1] - self.risks[j][ts])
        self.CR = np.zeros((3, 3))
        self.NCR = np.zeros((3, 3))
        for a, jp in enumerate(objs):          # row: objective we measure regret ON
            scale = iqr(self.risks[jp])
            for b, j in enumerate(objs):        # col: whose oracle checkpoint we deploy
                cr = float(self.risks[jp][self.t_star[j]] - self.R_star[jp])
                self.CR[a, b] = cr
                self.NCR[a, b] = cr / scale if scale > 0 else 0.0
        for a, jp in enumerate(objs):
            for b, j in enumerate(objs):
                if a < b:
                    self.epoch_gap[f"{jp}_{j}"] = abs(self.t_star[jp] - self.t_star[j])
        return self


def analyze_run(risks: Dict[str, Sequence[float]], smooth_w: int = 3) -> RunAnalysis:
    r = {k: np.asarray(v, dtype=np.float64) for k, v in risks.items()}
    return RunAnalysis(risks=r, smooth_w=smooth_w).compute()


# ---- BCa bootstrap over seeds ------------------------------------------------

def bca_ci(samples: Sequence[float], alpha: float = 0.05, n_boot: int = 10000,
           seed: int = 0) -> Tuple[float, float, float]:
    """Bias-corrected-and-accelerated bootstrap CI of the MEAN of ``samples``.

    Returns (mean, lo, hi). Falls back to a degenerate interval when n<2 or the
    statistic has zero variance.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    theta = float(x.mean())
    if n < 2 or np.allclose(x, x[0]):
        return theta, theta, theta
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    # bias-correction z0
    from scipy.stats import norm
    prop = float(np.mean(boot < theta))
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = norm.ppf(prop)
    # acceleration via jackknife
    idx = np.arange(n)
    jack = np.array([np.delete(x, i).mean() for i in idx])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo_q, hi_q = adj(zl), adj(zu)
    lo = float(np.percentile(boot, 100 * lo_q))
    hi = float(np.percentile(boot, 100 * hi_q))
    return theta, lo, hi


# ---- cell aggregation + pre-registered verdict -------------------------------

def aggregate_cell_ncr(run_ncrs: Sequence[np.ndarray], n_boot: int = 10000,
                       seed: int = 0) -> Dict[str, np.ndarray]:
    """Per-cell mean NCR matrix + BCa CI bounds, aggregating over seed runs."""
    stack = np.stack(run_ncrs, axis=0)              # (S, 3, 3)
    mean = stack.mean(axis=0)
    lo = np.zeros((3, 3)); hi = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            _, lo[a, b], hi[a, b] = bca_ci(stack[:, a, b], n_boot=n_boot, seed=seed)
    return dict(mean=mean, ci_lo=lo, hi=hi, ci_hi=hi)


def _offdiag_indices() -> List[Tuple[int, int]]:
    return [(a, b) for a in range(3) for b in range(3) if a != b]


def verdict(cell_aggs: Dict[str, Dict[str, np.ndarray]], thresh: float = 0.10) -> Dict:
    """Apply the pre-registered decision rule VERBATIM.

    PASS : in >=50% of cells, at least one off-diagonal NCR has 95% CI entirely
           above ``thresh``, AND the ID<->OOD direction shows conflict (CI_lo>thresh)
           in >=4 cells.
    KILL : in >70% of cells, ALL off-diagonal NCR CIs overlap 0.
    WEAK : conflict concentrated only in the tail objective, or only at noise 0.6.
    """
    objs = OBJECTIVES
    id_i, tail_i, ood_i = 0, 1, 2
    n_cells = len(cell_aggs)
    cells_with_conflict = 0
    id_ood_conflict_cells = 0
    cells_all_overlap0 = 0
    tail_only_cells = 0
    noise06_only = True
    any_conflict_outside_06 = False

    per_cell = {}
    for name, agg in cell_aggs.items():
        lo = agg["ci_lo"]; hi = agg["ci_hi"]
        offs = _offdiag_indices()
        above = [(a, b) for (a, b) in offs if lo[a, b] > thresh]
        overlaps0 = all(lo[a, b] <= 0.0 <= hi[a, b] for (a, b) in offs)
        has_conflict = len(above) > 0
        id_ood = (lo[id_i, ood_i] > thresh) or (lo[ood_i, id_i] > thresh)
        # "tail-only": every above-threshold conflict involves the tail objective
        tail_only = has_conflict and all(a == tail_i or b == tail_i for (a, b) in above)

        cells_with_conflict += int(has_conflict)
        id_ood_conflict_cells += int(id_ood)
        cells_all_overlap0 += int(overlaps0)
        tail_only_cells += int(tail_only)
        if has_conflict and "_0.6_" not in name and not name.endswith("_0.6"):
            any_conflict_outside_06 = True
        per_cell[name] = dict(
            max_offdiag=float(max(agg["mean"][a, b] for (a, b) in offs)),
            conflict_offdiag=[[objs[a], objs[b]] for (a, b) in above],
            all_overlap_zero=bool(overlaps0), id_ood_conflict=bool(id_ood))

    frac_conflict = cells_with_conflict / n_cells if n_cells else 0.0
    frac_overlap0 = cells_all_overlap0 / n_cells if n_cells else 0.0

    if frac_overlap0 > 0.70:
        v = "KILL"
    elif frac_conflict >= 0.50 and id_ood_conflict_cells >= 4:
        v = "PASS"
    elif cells_with_conflict > 0 and (tail_only_cells == cells_with_conflict
                                      or not any_conflict_outside_06):
        v = "WEAK"
    else:
        v = "WEAK"  # ambiguous middle ground re-scopes rather than proceeds
    return dict(
        verdict=v, n_cells=n_cells,
        cells_with_conflict=cells_with_conflict, frac_conflict=frac_conflict,
        id_ood_conflict_cells=id_ood_conflict_cells,
        cells_all_overlap_zero=cells_all_overlap0, frac_overlap_zero=frac_overlap0,
        tail_only_cells=tail_only_cells, threshold=thresh, per_cell=per_cell)
