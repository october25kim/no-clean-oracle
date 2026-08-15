"""Per-epoch evaluation producing ONE JSONL record (the audit's core log line).

All three objectives are evaluated every epoch on held-out data never seen in
training. R_ID = clean-test top-1 error; R_tail = mean per-class error over the
static tail set (primary) and the dynamic worst-30% (robustness); R_OOD = 1 - AUROC
for MSP and energy vs each pool. Cheap extras (train loss quantiles, prediction-flip
count, per-class train accuracy) are logged but not analyzed in this phase.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.ood import auroc, energy_score, msp_score
from eval.tail import per_class_error, r_tail_dynamic, r_tail_static


@torch.no_grad()
def _logits(model, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _logits_and_labels(model, loader: DataLoader, device: str):
    model.eval()
    logits, labels = [], []
    for xb, yb in loader:
        logits.append(model(xb.to(device)).float().cpu().numpy())
        labels.append(np.asarray(yb))
    return np.concatenate(logits), np.concatenate(labels)


def evaluate_epoch(model, device: str, clean_test_loader: DataLoader, n_classes: int,
                   tail_static: Optional[np.ndarray], ood_pool_loaders: Dict[str, DataLoader],
                   semantic_pools: List[str], covariate_pool: Optional[str],
                   energy_T: float = 1.0, tail_frac: float = 0.30) -> Dict:
    """Return the per-epoch metric record (a plain dict -> one JSONL line)."""
    id_logits, id_labels = _logits_and_labels(model, clean_test_loader, device)
    pred = id_logits.argmax(1)

    # R_ID
    r_id = float((pred != id_labels).mean())

    # per-class error on the clean test set (for tail objectives)
    correct = np.zeros(n_classes); total = np.zeros(n_classes)
    np.add.at(total, id_labels, 1.0)
    np.add.at(correct, id_labels[pred == id_labels], 1.0)
    pce = per_class_error(correct, total)
    r_tail_stat = float(r_tail_static(pce, tail_static)) if tail_static is not None else None
    r_tail_dyn = float(r_tail_dynamic(pce, tail_frac))

    # R_OOD = 1 - AUROC, per score x pool
    ood: Dict[str, float] = {}
    id_msp, id_energy = msp_score(id_logits), energy_score(id_logits, energy_T)
    for name, loader in ood_pool_loaders.items():
        ol = _logits(model, loader, device)
        ood[f"msp_{name}"] = 1.0 - auroc(id_msp, msp_score(ol))
        ood[f"energy_{name}"] = 1.0 - auroc(id_energy, energy_score(ol, energy_T))

    # primary endpoint: energy vs semantic pools (mean over semantic pools)
    sem = [ood[f"energy_{p}"] for p in semantic_pools if f"energy_{p}" in ood]
    r_ood_primary = float(np.mean(sem)) if sem else None

    rec = dict(
        R_ID=r_id, R_tail_static=r_tail_stat, R_tail_dynamic=r_tail_dyn,
        R_OOD_primary_energy_semantic=r_ood_primary,
        R_OOD=ood, per_class_test_error=pce.tolist(),
    )
    return rec


@torch.no_grad()
def train_diagnostics(per_sample_losses: np.ndarray, prev_pred: Optional[np.ndarray],
                      cur_pred: np.ndarray, train_labels: np.ndarray, n_classes: int) -> Dict:
    """Cheap extras logged but NOT analyzed this phase."""
    q = np.percentile(per_sample_losses, [10, 25, 50, 75, 90]).tolist()
    flips = None if prev_pred is None else int((cur_pred != prev_pred).sum())
    correct = np.zeros(n_classes); total = np.zeros(n_classes)
    np.add.at(total, train_labels, 1.0)
    np.add.at(correct, train_labels[cur_pred == train_labels], 1.0)
    with np.errstate(invalid="ignore"):
        pcta = np.where(total > 0, correct / total, np.nan)
    return dict(train_loss_quantiles={"p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4]},
                pred_flip_count=flips, per_class_train_acc=pcta.tolist())
