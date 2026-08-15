"""Shared scoring for validation-free checkpoint-selection baselines.

Every baseline answers the same question -- "which epoch would you have picked?" --
so they are all scored the same way: the regret of deploying that epoch instead of
each objective's oracle epoch, normalized by the objective's own IQR so the three
scales are comparable. Keeping this in one place stops two baselines from being
scored by two subtly different rules.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from analysis.io import risk_trajectories
from analysis.ncr import OBJECTIVES, iqr, oracle_epoch


def regret_at_epoch(run, tail_classes: Sequence[int], stop_epoch: int,
                    smooth_w: int) -> Dict[str, dict]:
    """Cost of deploying ``stop_epoch`` instead of each objective's oracle epoch."""
    risks = risk_trajectories(run, tail_classes)
    out = {}
    for j in OBJECTIVES:
        y = risks[j]
        t_star = oracle_epoch(y, smooth_w)[0]
        scale = iqr(y)
        regret = float(y[stop_epoch] - y[t_star])
        out[j] = dict(oracle_epoch=int(t_star), oracle_risk=float(y[t_star]),
                      risk_at_stop=float(y[stop_epoch]), regret=regret,
                      normalized_regret=(regret / scale if scale > 0 else 0.0),
                      epoch_gap=int(stop_epoch - t_star))
    return out


def rank_agreement(statistic: np.ndarray, run, tail_classes: Sequence[int],
                   higher_is_better: bool) -> Dict[str, float]:
    """Kendall tau between a selector's per-epoch statistic and each true risk curve.

    A selector is useful to the extent it ORDERS epochs the way the risk does, not
    just that its argmax lands well. Reported as tau against risk, so a good
    higher-is-better statistic gives a NEGATIVE tau (high statistic <-> low risk); the
    sign is flipped here to "agreement", where positive means the selector ranks
    epochs in the same order the objective would.
    """
    from scipy.stats import kendalltau

    s = np.asarray(statistic, dtype=np.float64)
    risks = risk_trajectories(run, tail_classes)
    out = {}
    for j in OBJECTIVES:
        y = np.asarray(risks[j], dtype=np.float64)
        ok = ~np.isnan(s)
        tau = kendalltau(s[ok], y[ok]).correlation
        out[j] = float(-tau if higher_is_better else tau)
    return out
