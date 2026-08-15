"""Post-hoc OOD scores (MSP, energy) and AUROC — pure NumPy, no GPU.

Scores operate on logits. Higher score := more in-distribution (ID). AUROC is the
probability a random ID sample scores above a random OOD sample; ``1 - AUROC`` is the
R_OOD risk logged in the audit. AUROC is computed by the rank (Mann-Whitney) identity
and unit-tested against sklearn.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def msp_score(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability. Higher => more ID."""
    return softmax(logits).max(axis=1)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy = -T*logsumexp(logits/T). ID samples have LOWER energy, so the ID-high
    score returned is ``-energy = T*logsumexp(logits/T)``."""
    z = np.asarray(logits, dtype=np.float64) / T
    m = z.max(axis=1, keepdims=True)
    lse = (m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1)))
    return T * lse                       # higher => more ID


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC that ID scores exceed OOD scores, via the rank-sum identity
    (ties counted as 0.5). Matches sklearn.roc_auc_score with ID as the positive."""
    s_id = np.asarray(id_scores, dtype=np.float64)
    s_ood = np.asarray(ood_scores, dtype=np.float64)
    n1, n2 = s_id.size, s_ood.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    alls = np.concatenate([s_id, s_ood])
    order = alls.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, alls.size + 1)
    # average ranks for ties
    _assign_tie_ranks(alls, ranks)
    r_id = ranks[:n1].sum()
    u = r_id - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n2))


def _assign_tie_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    order = values.argsort(kind="mergesort")
    sv = values[order]
    i = 0
    n = sv.size
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def r_ood(id_logits: np.ndarray, ood_logits: np.ndarray, score: str = "energy",
          T: float = 1.0) -> float:
    """R_OOD = 1 - AUROC for the chosen post-hoc score."""
    fn = {"msp": msp_score, "energy": lambda z: energy_score(z, T)}[score]
    return 1.0 - auroc(fn(id_logits), fn(ood_logits))
