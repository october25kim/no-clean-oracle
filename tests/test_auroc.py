"""(c) AUROC computation validated against sklearn on synthetic scores."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eval.ood import auroc, energy_score, msp_score, r_ood


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_auroc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    id_scores = rng.normal(1.0, 1.0, size=500)
    ood_scores = rng.normal(0.0, 1.0, size=700)
    ours = auroc(id_scores, ood_scores)
    y = np.r_[np.ones(500), np.zeros(700)]
    s = np.r_[id_scores, ood_scores]
    ref = roc_auc_score(y, s)
    assert abs(ours - ref) < 1e-9


def test_auroc_with_ties_matches_sklearn():
    id_scores = np.array([1, 1, 2, 2, 3, 3], dtype=float)
    ood_scores = np.array([1, 2, 2, 3, 3, 3], dtype=float)
    ours = auroc(id_scores, ood_scores)
    y = np.r_[np.ones(6), np.zeros(6)]
    ref = roc_auc_score(y, np.r_[id_scores, ood_scores])
    assert abs(ours - ref) < 1e-9


def test_msp_and_energy_rank_id_above_ood():
    rng = np.random.default_rng(0)
    id_logits = rng.normal(0, 1, (200, 10)); id_logits[:, 0] += 6.0  # confident ID
    ood_logits = rng.normal(0, 1, (200, 10))                         # flat OOD
    assert auroc(msp_score(id_logits), msp_score(ood_logits)) > 0.9
    assert auroc(energy_score(id_logits), energy_score(ood_logits)) > 0.9
    assert 0.0 <= r_ood(id_logits, ood_logits, "energy") < 0.1
